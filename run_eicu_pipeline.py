#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
# Part of ATLAS
"""Pipeline for eICU-CRD: cohort -> scores -> E1-E5 experiments."""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.eicu_adapter import extract_eicu_cohort, compute_eicu_scores
from src.data.config import (FIGURES_DIR, EXPERIMENTS_DIR, RANDOM_SEED,
                              MIN_SUBGROUP_SIZE, BOOTSTRAP_ITERATIONS)
from src.evaluation.audit import prespecified_audit, intersectional_audit
from src.evaluation.asd import adversarial_subgroup_discovery
from src.training.train_gru import train_gru_model
from src.evaluation.rsb import compute_rsb, compute_ml_improvement
from src.evaluation import figures as fig_mod

np.random.seed(RANDOM_SEED)

EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR = EXPERIMENTS_DIR / "exp_eicu"
EXP_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")
RESULTS_FILE = ROOT / "RESULTS_EICU.md"


def update_results(section: str, content: str):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"\n\n{content}")
    print(f"[RESULTS] Updated: {section}")


def main():
    start = time.time()

    # Initialize results file
    with open(RESULTS_FILE, "w") as f:
        f.write("# Results — ATLAS on eICU-CRD Demo\n\n")
        f.write(f"**Last Updated**: {TIMESTAMP}\n")
        f.write("**Dataset**: eICU Collaborative Research Database Demo v2.0.1\n")
        f.write("**Status**: In Progress\n")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1: Cohort + Scores
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PHASE 1: eICU-CRD Cohort Extraction & Score Computation")
    print("=" * 70)

    cohort = extract_eicu_cohort()
    data = compute_eicu_scores(cohort)
    data.to_csv(EXP_DIR / "cohort_with_scores.csv", index=False)

    # Demographic axes to audit (no insurance for eICU)
    # Override the audit to use only available axes
    demo_axes = ["race_cat", "sex", "age_group", "diag_type"]

    update_results("Cohort Summary", f"""## Cohort Summary
**Date**: {TIMESTAMP}
**N**: {len(data)} ICU stays from {data['hospitalid'].nunique()} hospitals
**Mortality rate**: {data['mortality'].mean():.3f}

### Demographics
| Attribute | Distribution |
|-----------|-------------|
| Race | {data['race_cat'].value_counts().to_dict()} |
| Sex | {data['sex'].value_counts().to_dict()} |
| Age (mean) | {data['age'].mean():.1f} +/- {data['age'].std():.1f} |
| Diagnosis type | {data['diag_type'].value_counts().to_dict()} |

### Score Distributions
| Score | Mean | Median | Range |
|-------|------|--------|-------|
| SOFA | {data['sofa'].mean():.1f} | {data['sofa'].median():.0f} | [{data['sofa'].min():.0f}, {data['sofa'].max():.0f}] |
| qSOFA | {data['qsofa'].mean():.1f} | {data['qsofa'].median():.0f} | [{data['qsofa'].min():.0f}, {data['qsofa'].max():.0f}] |
| APACHE-II | {data['apache2'].mean():.1f} | {data['apache2'].median():.0f} | [{data['apache2'].min():.0f}, {data['apache2'].max():.0f}] |
| NEWS2 | {data['news2'].mean():.1f} | {data['news2'].median():.0f} | [{data['news2'].min():.0f}, {data['news2'].max():.0f}] |
""")

    # ═══════════════════════════════════════════════════════════════════
    # E1: Pre-specified Subgroup Audit
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("E1: Pre-specified Subgroup Audit")
    print("=" * 70)

    n_boot = BOOTSTRAP_ITERATIONS
    audit_results, gaps_df = prespecified_audit(data, axes=demo_axes,
                                                n_boot=n_boot)
    audit_results.to_csv(EXP_DIR / "e1_audit_results.csv", index=False)
    gaps_df.to_csv(EXP_DIR / "e1_gaps.csv", index=False)

    update_results("E1 Results", f"""## E1: Pre-specified Subgroup Audit

### AUROC Gaps (max - min across subgroups)
{gaps_df.to_markdown(index=False)}

### Full Results
{audit_results.to_markdown(index=False)}
""")

    # ═══════════════════════════════════════════════════════════════════
    # E2: Intersectional Analysis
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("E2: Intersectional Analysis")
    print("=" * 70)

    inter_results, worst_subgroups = intersectional_audit(
        data, axes=demo_axes, min_n=MIN_SUBGROUP_SIZE, n_boot=n_boot)
    inter_results.to_csv(EXP_DIR / "e2_intersectional.csv", index=False)

    worst_text = ""
    for score_name, worst_df in worst_subgroups.items():
        worst_text += f"\n### {score_name.upper()} — Worst Subgroups\n"
        if len(worst_df) > 0:
            worst_text += worst_df[["subgroup", "auroc", "n", "prevalence"]].to_markdown(index=False)
        else:
            worst_text += "No subgroups met minimum size threshold.\n"

    update_results("E2 Results", f"""## E2: Intersectional Analysis
**Min subgroup size**: {MIN_SUBGROUP_SIZE}
**Total intersectional subgroups evaluated**: {len(inter_results)}
{worst_text}
""")

    # ═══════════════════════════════════════════════════════════════════
    # E3: Adversarial Subgroup Discovery
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("E3: Adversarial Subgroup Discovery")
    print("=" * 70)

    asd_results = adversarial_subgroup_discovery(data)

    with open(EXP_DIR / "e3_asd_results.json", "w") as f:
        serializable = {}
        for k, v in asd_results.items():
            sv = {**v}
            sv["top_features"] = [(fn, float(fi)) for fn, fi in sv["top_features"]]
            serializable[k] = sv
        json.dump(serializable, f, indent=2, default=str)

    asd_text = ""
    for score_name, res in asd_results.items():
        asd_text += f"\n### {score_name.upper()}\n"
        asd_text += f"- Error prediction AUROC: {res['error_prediction_auroc']}\n"
        asd_text += f"- Top features: {[f[0] for f in res['top_features'][:5]]}\n"
        for i, sg in enumerate(res["vulnerable_subgroups"]):
            asd_text += (f"- Subgroup {i+1}: n={sg['n']}, "
                        f"error_conc={sg['concentration_ratio']:.2f}, "
                        f"demographics={sg['demographics']}\n")

    update_results("E3 Results", f"""## E3: Adversarial Subgroup Discovery
{asd_text}
""")

    # ═══════════════════════════════════════════════════════════════════
    # E4-E5: GRU Training + RSB
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("E4-E5: GRU Training & RSB Quantification")
    print("=" * 70)

    import torch
    device = "cpu"
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                mem = torch.cuda.mem_get_info(i)
                free_gb = mem[0] / 1e9
                used_gb = (mem[1] / 1e9) - free_gb
                if used_gb < 0.5 and free_gb > 8:
                    t = torch.zeros(1, device=f"cuda:{i}")
                    del t
                    device = f"cuda:{i}"
                    print(f"Using GPU {i} ({free_gb:.1f} GB free)")
                    break
            except Exception:
                continue
    if device == "cpu":
        print("Using CPU")

    gru_result = train_gru_model(data, device=device, epochs=50)
    ml_preds = gru_result["predictions"]

    # Log training
    with open(ROOT / "TRAINING_LOG.md", "a") as f:
        f.write(f"\n\n## Run eICU-001 — {TIMESTAMP}\n")
        f.write(f"- **Dataset**: eICU-CRD Demo ({len(data)} stays)\n")
        f.write(f"- **Hardware**: {device}\n")
        f.write(f"- **Overall CV AUROC**: {gru_result['overall_auroc']:.4f}\n")
        f.write(f"- **Model params**: {gru_result['n_params']:,}\n")
        f.write(f"- **Fold metrics**: {gru_result['fold_metrics']}\n")

    # RSB
    print("\nComputing RSB gaps...")
    rsb_df = compute_rsb(data, ml_preds, axes=demo_axes, n_boot=min(n_boot, 500))
    rsb_df.to_csv(EXP_DIR / "e4_rsb.csv", index=False)

    print("Computing ML improvement...")
    improvement_df = compute_ml_improvement(data, ml_preds, axes=demo_axes)
    improvement_df.to_csv(EXP_DIR / "e5_ml_improvement.csv", index=False)

    update_results("E4-E5 Results", f"""## E4: Reference Standard Bias

### RSB Gap Summary
{rsb_df.groupby(['score', 'metric'])['rsb_gap'].mean().unstack().to_markdown()}

## E5: ML Fairness Improvement
### Mean improvement over classical scores
{improvement_df.groupby('score')['pct_improvement'].mean().to_markdown()}

### GRU Model Performance
- Overall CV AUROC: {gru_result['overall_auroc']:.4f}
- Model parameters: {gru_result['n_params']:,}
""")

    # ═══════════════════════════════════════════════════════════════════
    # Figures
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Generating Figures")
    print("=" * 70)

    if len(gaps_df) > 0:
        print("Figure 1: AUROC gap heatmap...")
        fig_mod.plot_auroc_gap_heatmap(gaps_df)

    print("Figure 2: AUROC by race...")
    fig_mod.plot_subgroup_performance(audit_results, axis="race_cat")

    print("Figure 3: Calibration curves by race...")
    fig_mod.plot_calibration_curves(data)

    if asd_results:
        print("Figure 4: ASD results...")
        fig_mod.plot_asd_results(asd_results)

    if len(rsb_df) > 0:
        print("Figure 5: RSB gap heatmap...")
        fig_mod.plot_rsb_gaps(rsb_df)

    if len(improvement_df) > 0:
        print("Figure 6: ML improvement...")
        fig_mod.plot_ml_improvement(improvement_df)

    print("Figure S1: Score distributions...")
    fig_mod.plot_score_distributions(data)

    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"eICU PIPELINE COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'=' * 70}")

    update_results("Pipeline Complete", f"""---
## Pipeline Complete
**Duration**: {elapsed/60:.1f} minutes
**Date**: {TIMESTAMP}
""")


if __name__ == "__main__":
    main()
