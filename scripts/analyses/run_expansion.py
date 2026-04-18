#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
"""Run ATLAS expansion analyses (E6, E7, SOFA decomposition, hospital-stratified race)
on the existing GOSSIS cohort_with_scores.csv — no new data download needed."""
import sys, json, warnings, numpy as np, pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")
DEMO_AXES = ["race_cat", "sex", "age_group", "diag_type"]

print(f"=== ATLAS Expansion — {TIMESTAMP} ===", flush=True)

data = pd.read_csv('experiments/exp_gossis/cohort_with_scores.csv')
print(f"Loaded {len(data)} stays", flush=True)

from src.evaluation.audit import (
    compute_score_conditional_mortality,
    clinical_threshold_audit,
    decompose_sofa_components,
    hospital_stratified_audit,
)
from src.evaluation import figures as fig_mod

# ── E6: Score-conditional mortality ─────────────────────────────────────────
print("\n--- E6: Score-conditional mortality ---", flush=True)
scm_df = compute_score_conditional_mortality(data, demo_axis="age_group")
scm_df.to_csv('experiments/exp_gossis/e6_score_conditional_mortality.csv', index=False)
print(f"  {len(scm_df)} rows", flush=True)

# Print summary: mortality rate spread for SOFA at high score values by age
for score in ["sofa", "apache2"]:
    sub = scm_df[scm_df["score"] == score]
    high_bins = sub["score_bin"].unique()[-3:]  # top 3 bins
    for b in high_bins:
        b_sub = sub[sub["score_bin"] == b]
        if len(b_sub) >= 2:
            spread = b_sub["mortality_rate"].max() - b_sub["mortality_rate"].min()
            print(f"  {score} bin={b}: mortality spread = {spread:.3f} "
                  f"(range {b_sub['mortality_rate'].min():.3f}–"
                  f"{b_sub['mortality_rate'].max():.3f})", flush=True)

fig_mod.plot_score_conditional_mortality(scm_df, demo_axis="age_group")

# ── E7: Clinical threshold analysis ─────────────────────────────────────────
print("\n--- E7: Clinical threshold analysis ---", flush=True)
thresh_df = clinical_threshold_audit(data, axes=DEMO_AXES)
thresh_df.to_csv('experiments/exp_gossis/e7_clinical_thresholds.csv', index=False)
print(f"  {len(thresh_df)} rows", flush=True)

# Print key results: sensitivity gap at primary thresholds
primary_thresh = {"sofa": 6, "qsofa": 2, "apache2": 20, "news2": 9}
for score, thresh in primary_thresh.items():
    sub = thresh_df[(thresh_df["score"] == score) &
                    (thresh_df["threshold"] == thresh) &
                    (thresh_df["axis"] == "age_group") &
                    (thresh_df["group"] != "overall")]
    if not sub.empty:
        sens_range = sub["sensitivity"].max() - sub["sensitivity"].min()
        spec_range = sub["specificity"].max() - sub["specificity"].min()
        print(f"  {score} ≥{thresh}: sens gap={sens_range:.3f}, "
              f"spec gap={spec_range:.3f}", flush=True)

fig_mod.plot_clinical_thresholds(thresh_df, axis="age_group")

# ── SOFA component decomposition ─────────────────────────────────────────────
print("\n--- SOFA component decomposition ---", flush=True)
comp_df = decompose_sofa_components(data, demo_axis="age_group")
if not comp_df.empty:
    comp_df.to_csv('experiments/exp_gossis/sofa_components.csv', index=False)
    gap_rows = comp_df[comp_df["group"] == "_gap"].sort_values("auroc", ascending=False)
    print("  Component gaps (age_group):")
    for _, row in gap_rows.iterrows():
        print(f"    {row['component']:15s}: AUROC gap = {row['auroc']:.3f}")
    fig_mod.plot_sofa_components(comp_df)
else:
    print("  No component data available (missing columns).")

# ── Hospital-stratified race analysis ────────────────────────────────────────
print("\n--- Hospital-stratified race analysis ---", flush=True)
if "hospitalid" in data.columns:
    hosp_df, hosp_summary = hospital_stratified_audit(data)
    hosp_df.to_csv('experiments/exp_gossis/hospital_stratified_race.csv', index=False)
    hosp_summary.to_csv('experiments/exp_gossis/hospital_stratified_summary.csv', index=False)
    print("  Hospital-stratified summary:")
    print(hosp_summary[["score", "within_hospital_gap_wmean",
                          "aggregate_gap", "simpsons_paradox_ratio"]].to_string(index=False))
    fig_mod.plot_hospital_stratified_race(hosp_df, hosp_summary)
else:
    print("  'hospitalid' column not found — skipping.", flush=True)
    hosp_df = pd.DataFrame()
    hosp_summary = pd.DataFrame()

# ── Re-run ASD with fixed error_threshold=0.3 ───────────────────────────────
print("\n--- Re-running ASD with error_threshold=0.3 ---", flush=True)
from src.evaluation.asd import adversarial_subgroup_discovery
asd_results = adversarial_subgroup_discovery(data, error_threshold=0.3)
with open('experiments/exp_gossis/e3_asd_results_v2.json', 'w') as f:
    def _serializable(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    import json as _json
    class NumpyEncoder(_json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    f.write(_json.dumps(asd_results, cls=NumpyEncoder, indent=2))
print("  ASD done.", flush=True)
if asd_results:
    fig_mod.plot_asd_results(asd_results)
    # Save updated figure as v2
    import shutil
    from pathlib import Path
    for suffix in [".png", ".pdf"]:
        src = Path("experiments/exp_gossis/figures/fig4_asd_error_concentration") / f"..{suffix}"
        # just overwrite fig4 in place since asd_results function does that

# ── Write expansion results to RESULTS_GOSSIS.md ────────────────────────────
with open('RESULTS_GOSSIS.md', 'a') as f:
    f.write(f"\n\n---\n## Expansion Analyses — {TIMESTAMP}\n\n")

    f.write("### E6: Score-Conditional Mortality\n")
    for score in ["sofa", "qsofa", "apache2", "news2"]:
        sub = scm_df[scm_df["score"] == score]
        if sub.empty:
            continue
        # Spread across all bins, all age groups
        overall_spread = sub.groupby("score_bin")["mortality_rate"].apply(
            lambda x: x.max() - x.min() if len(x) >= 2 else 0
        ).mean()
        f.write(f"- **{score.upper()}**: mean mortality rate spread across age "
                f"groups (same score bin) = {overall_spread:.3f}\n")

    f.write("\n### E7: Clinical Threshold Analysis\n")
    for score, thresh in primary_thresh.items():
        sub = thresh_df[(thresh_df["score"] == score) &
                        (thresh_df["threshold"] == thresh) &
                        (thresh_df["axis"] == "age_group") &
                        (thresh_df["group"] != "overall")]
        if not sub.empty:
            f.write(f"- **{score.upper()} ≥{thresh}**: sensitivity gap="
                    f"{sub['sensitivity'].max()-sub['sensitivity'].min():.3f}, "
                    f"specificity gap={sub['specificity'].max()-sub['specificity'].min():.3f}\n")

    if not comp_df.empty:
        f.write("\n### SOFA Component Attribution\n")
        gap_rows = comp_df[comp_df["group"] == "_gap"].sort_values("auroc", ascending=False)
        for _, row in gap_rows.iterrows():
            f.write(f"- **{row['component']}** ({row['column']}): AUROC gap = {row['auroc']:.3f}\n")

    if not hosp_summary.empty:
        f.write("\n### Hospital-Stratified Race Analysis\n")
        f.write(hosp_summary.to_markdown(index=False))
        f.write("\n")

print("\n=== EXPANSION COMPLETE ===", flush=True)
