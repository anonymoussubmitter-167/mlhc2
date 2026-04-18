#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
"""Compute RSB and fairness metrics for GA-FAFT predictions.
Run after run_ga_faft.py completes and ga_faft_preds.npy is available.
"""
import sys, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

FIGURES = Path("paper/figures")
OUT = Path("experiments/exp_gossis")
AGE_ORDER = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

print("Loading data...", flush=True)
data = pd.read_csv("experiments/exp_gossis/cohort_with_scores.csv")
y = data["mortality"].values

gru_preds  = np.load(OUT / "ml_preds_full.npy")
faft_preds = np.load(OUT / "faft_preds.npy")
ga_preds   = np.load(OUT / "ga_faft_preds.npy")

print(f"GA-FAFT overall AUROC: {roc_auc_score(y, ga_preds):.4f}", flush=True)
print(f"FAFT overall AUROC:    {roc_auc_score(y, faft_preds):.4f}", flush=True)
print(f"GRU overall AUROC:     {roc_auc_score(y, gru_preds):.4f}", flush=True)

# ── Per-age AUROC ─────────────────────────────────────────────────────────────
print("\n=== Per-Age AUROC ===", flush=True)
age_groups = data["age_group"].values

model_names = ["SOFA", "APACHE-II", "GRU", "FAFT", "GA-FAFT"]
model_preds = [
    data["sofa"].values.astype(float),
    data["apache2"].values.astype(float),
    gru_preds, faft_preds, ga_preds,
]

rows = []
for name, preds in zip(model_names, model_preds):
    aurocs = []
    for age in AGE_ORDER:
        m = age_groups == age
        if m.sum() > 20 and y[m].sum() > 2:
            a = roc_auc_score(y[m], preds[m])
            aurocs.append(a)
        else:
            aurocs.append(np.nan)
    gap = np.nanmax(aurocs) - np.nanmin(aurocs)
    print(f"  {name}: gap={gap:.4f}  "
          + "  ".join(f"{a:.4f}" if not np.isnan(a) else "----" for a in aurocs),
          flush=True)
    rows.append({"model": name, "age_gap": gap, **{age: aurocs[i] for i, age in enumerate(AGE_ORDER)}})

age_df = pd.DataFrame(rows)
age_df.to_csv(OUT / "e_gafaft_age_auroc_full.csv", index=False)
print("  Saved e_gafaft_age_auroc_full.csv", flush=True)

# ── RSB computation ───────────────────────────────────────────────────────────
print("\n=== RSB Analysis ===", flush=True)

def compute_eod_for_label(preds, labels, groups_col, uniq_groups):
    """Equalized odds difference across groups."""
    thr = np.median(preds)
    y_hat = (preds >= thr).astype(int)
    tprs, fprs = [], []
    for g in uniq_groups:
        m = groups_col == g
        if m.sum() < 20 or labels[m].sum() < 2 or (1-labels[m]).sum() < 2:
            continue
        tpr = y_hat[m][labels[m] == 1].mean() if labels[m].sum() > 0 else 0
        fpr = y_hat[m][labels[m] == 0].mean() if (1-labels[m]).sum() > 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)
    if len(tprs) < 2:
        return np.nan
    return max(max(tprs)-min(tprs), max(fprs)-min(fprs))

def compute_calibration_gap(preds, labels, groups_col, uniq_groups, n_bins=10):
    """Mean calibration gap across groups."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(preds, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    group_cal = {}
    for g in uniq_groups:
        m = groups_col == g
        if m.sum() < 20:
            continue
        # Compute overall calibration error for this group
        cal_err = 0.0
        cnt = 0
        for b in range(n_bins):
            bm = m & (bin_idx == b)
            if bm.sum() < 5:
                continue
            acc = labels[bm].mean()
            conf = preds[bm].mean()
            cal_err += bm.sum() * abs(acc - conf)
            cnt += bm.sum()
        if cnt > 0:
            group_cal[g] = cal_err / cnt
    if len(group_cal) < 2:
        return np.nan
    vals = list(group_cal.values())
    return max(vals) - min(vals)

def compute_auroc_gap(preds, labels, groups_col, uniq_groups):
    """AUROC gap across groups."""
    aurocs = []
    for g in uniq_groups:
        m = groups_col == g
        if m.sum() < 20 or labels[m].sum() < 2 or (1-labels[m]).sum() < 2:
            continue
        try:
            aurocs.append(roc_auc_score(labels[m], preds[m]))
        except Exception:
            pass
    if len(aurocs) < 2:
        return np.nan
    return max(aurocs) - min(aurocs)

rsb_records = []
for score_name in ["sofa", "apache2"]:
    y_ref = data[score_name].values.astype(float)
    y_ref_binary = (y_ref > np.median(y_ref)).astype(float)

    for model_name, y_pred in [("GA-FAFT", ga_preds), ("FAFT", faft_preds), ("GRU", gru_preds)]:
        for demo, demo_col in [("age", "age_group"), ("sex", "sex"), ("race", "race_cat")]:
            groups = data[demo_col].fillna("Unknown").values
            uniq = np.unique(groups)

            # Against ground truth
            eod_gt    = compute_eod_for_label(y_pred, y, groups, uniq)
            cal_gt    = compute_calibration_gap(y_pred, y.astype(float), groups, uniq)
            auroc_gt  = compute_auroc_gap(y_pred, y, groups, uniq)

            # Against score
            eod_sc    = compute_eod_for_label(y_pred, y_ref_binary, groups, uniq)
            cal_sc    = compute_calibration_gap(y_pred, y_ref_binary, groups, uniq)
            auroc_sc  = compute_auroc_gap(y_pred, y_ref, groups, uniq)

            rsb_eod   = abs(eod_gt - eod_sc) if not (np.isnan(eod_gt) or np.isnan(eod_sc)) else np.nan
            rsb_cal   = abs(cal_gt - cal_sc) if not (np.isnan(cal_gt) or np.isnan(cal_sc)) else np.nan
            rsb_auroc = abs(auroc_gt - auroc_sc) if not (np.isnan(auroc_gt) or np.isnan(auroc_sc)) else np.nan

            rsb_records.append({
                "score": score_name, "model": model_name, "demo": demo,
                "rsb_eod": rsb_eod, "rsb_cal": rsb_cal, "rsb_auroc": rsb_auroc,
            })

rsb_df = pd.DataFrame(rsb_records)
rsb_df.to_csv(OUT / "e_gafaft_rsb_full.csv", index=False)

print("\n  RSB Summary (mean across demographics):")
for (score, model), sub in rsb_df.groupby(["score", "model"]):
    print(f"    {model} vs {score.upper()}: "
          f"EOD={sub['rsb_eod'].mean():.4f}  "
          f"Cal={sub['rsb_cal'].mean():.4f}  "
          f"AUROC={sub['rsb_auroc'].mean():.4f}")

# ── Figure: 5-model age AUROC comparison ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors_map = {
    "SOFA": "#d62728", "APACHE-II": "#ff7f0e",
    "GRU": "#1f77b4", "FAFT": "#9467bd", "GA-FAFT": "#2ca02c",
}
styles_map = {
    "SOFA": ("o-", 1.5, 5), "APACHE-II": ("s-", 1.5, 5),
    "GRU": ("^-", 2.0, 6), "FAFT": ("v-", 2.0, 6),
    "GA-FAFT": ("D-", 2.5, 7),
}

for name, preds in zip(model_names, model_preds):
    auroc_vals = []
    for age in AGE_ORDER:
        m = age_groups == age
        if m.sum() > 20 and y[m].sum() > 2:
            auroc_vals.append(roc_auc_score(y[m], preds[m]))
        else:
            auroc_vals.append(np.nan)
    gap = np.nanmax(auroc_vals) - np.nanmin(auroc_vals)
    fmt, lw, ms = styles_map[name]
    ax.plot(range(len(AGE_ORDER)), auroc_vals, fmt, color=colors_map[name],
            lw=lw, ms=ms, label=f"{name} (gap={gap:.3f})")

ax.set_xticks(range(len(AGE_ORDER)))
ax.set_xticklabels(AGE_ORDER, rotation=30, ha="right")
ax.set_ylim(0.55, 1.0)
ax.set_ylabel("AUROC")
ax.set_xlabel("Age group")
ax.set_title("Age-Stratified AUROC: Classical Scores vs. ML Models\n"
             "GA-FAFT (per-group pairwise AUROC loss, minimax fairness)")
ax.legend(fontsize=9, loc="lower left")
ax.axhline(0.5, ls=":", color="gray", lw=0.8, alpha=0.5)
fig.tight_layout()
for fmt in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig18_gafaft_age_auroc.{fmt}", bbox_inches="tight")
plt.close(fig)
print("\n  Saved fig18_gafaft_age_auroc", flush=True)

# ── Figure: AUROC gap bar chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
gaps = []
for name, preds in zip(model_names, model_preds):
    aurocs = []
    for age in AGE_ORDER:
        m = age_groups == age
        if m.sum() > 20 and y[m].sum() > 2:
            aurocs.append(roc_auc_score(y[m], preds[m]))
    gaps.append(max(aurocs) - min(aurocs) if len(aurocs) >= 2 else 0)

bar_colors = [colors_map[m] for m in model_names]
ax.bar(range(len(model_names)), gaps, color=bar_colors, alpha=0.85, edgecolor="white")
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names)
ax.set_ylabel("Age AUROC gap (max − min)")
ax.set_title("Age AUROC Gap by Model\n"
             "GA-FAFT reduces gap via per-group pairwise AUROC ranking loss")
for xi, gap in enumerate(gaps):
    ax.text(xi, gap + 0.002, f"{gap:.3f}", ha="center", fontsize=9)
ax.set_ylim(0, max(gaps) * 1.15)
fig.tight_layout()
for fmt in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig18b_gafaft_gap_bar.{fmt}", bbox_inches="tight")
plt.close(fig)
print("  Saved fig18b_gafaft_gap_bar", flush=True)

print("\n=== GA-FAFT RSB Analysis COMPLETE ===", flush=True)
ga_gap_sub = age_df[age_df["model"] == "GA-FAFT"]["age_gap"].values[0]
gru_gap_sub = age_df[age_df["model"] == "GRU"]["age_gap"].values[0]
faft_gap_sub = age_df[age_df["model"] == "FAFT"]["age_gap"].values[0]
reduction_vs_gru = (gru_gap_sub - ga_gap_sub) / gru_gap_sub * 100
reduction_vs_faft = (faft_gap_sub - ga_gap_sub) / faft_gap_sub * 100
print(f"\n  GA-FAFT age AUROC gap: {ga_gap_sub:.4f}", flush=True)
print(f"  GRU age AUROC gap:     {gru_gap_sub:.4f} ({reduction_vs_gru:.1f}% reduction with GA-FAFT)", flush=True)
print(f"  FAFT age AUROC gap:    {faft_gap_sub:.4f} ({reduction_vs_faft:.1f}% reduction with GA-FAFT)", flush=True)
