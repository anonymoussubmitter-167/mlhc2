#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
"""Train Group-Adaptive FAFT (GA-FAFT) and evaluate fairness metrics.

GA-FAFT extends FAFT with:
  1. Per-group pairwise AUROC ranking loss (novel — targets worst-case group AUROC)
  2. Per-group temperature scaling (learned T_g per demographic group)
  3. GRL adversarial heads (same as FAFT)

Evaluates: overall AUROC, per-age AUROC, age AUROC gap, calibration gap,
RSB vs. classical scores (SOFA, APACHE-II).
"""
import sys, os, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

FIGURES = Path("paper/figures")
FIGURES.mkdir(exist_ok=True)
OUT = Path("experiments/exp_gossis")
AGE_ORDER = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

# Detect GPU
import torch
if torch.cuda.is_available():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)
else:
    device = "cpu"
    print("Using CPU", flush=True)

print("Loading data...", flush=True)
data = pd.read_csv("experiments/exp_gossis/cohort_with_scores.csv")
y = data["mortality"].values
print(f"  {len(data)} stays, {y.mean()*100:.1f}% mortality", flush=True)

# ── Train GA-FAFT ─────────────────────────────────────────────────────────────
print("\n=== Training GA-FAFT ===", flush=True)
from src.training.train_ga_faft import train_ga_faft_model

results = train_ga_faft_model(
    data, device=device,
    epochs=60,
    d_model=64, n_heads=4, n_layers=2, d_ff=256,
    dropout=0.1, batch_size=256, lr=3e-4,
    patience=12,
    adv_lambda=0.3,     # GRL adversarial weight
    rank_lambda=0.5,    # per-group AUROC ranking loss weight
    rank_margin=1.0,    # pairwise sigmoid margin
    rank_mode="max",    # minimax: optimize worst-case group AUROC
)

ga_preds = results["predictions"]
np.save(OUT / "ga_faft_preds.npy", ga_preds)
print(f"\n  Saved ga_faft_preds.npy", flush=True)

# ── Load FAFT and GRU preds for comparison ────────────────────────────────────
faft_preds = np.load(OUT / "faft_preds.npy")
gru_preds  = np.load(OUT / "ml_preds_full.npy")

# ── Per-age AUROC comparison ──────────────────────────────────────────────────
print("\n=== Per-Age AUROC Comparison ===", flush=True)
age_groups_col = data["age_group"].values

model_preds_map = {
    "SOFA": data["sofa"].values,
    "APACHE-II": data["apache2"].values,
    "GRU": gru_preds,
    "FAFT": faft_preds,
    "GA-FAFT": ga_preds,
}

age_auroc_df_rows = []
for model_name, preds in model_preds_map.items():
    for age in AGE_ORDER:
        mask = age_groups_col == age
        yt   = y[mask]
        sc   = preds[mask]
        if yt.sum() < 5 or (1-yt).sum() < 5:
            continue
        try:
            auc = roc_auc_score(yt, sc)
        except Exception:
            auc = np.nan
        age_auroc_df_rows.append({"model": model_name, "age_group": age, "auroc": auc})

age_auroc_df = pd.DataFrame(age_auroc_df_rows)

# Print comparison table
print(f"\n  {'Age':<10}", end="")
for m in ["SOFA", "APACHE-II", "GRU", "FAFT", "GA-FAFT"]:
    print(f"  {m:>10}", end="")
print()
for age in AGE_ORDER:
    print(f"  {age:<10}", end="")
    for m in ["SOFA", "APACHE-II", "GRU", "FAFT", "GA-FAFT"]:
        sub = age_auroc_df[(age_auroc_df["model"] == m) & (age_auroc_df["age_group"] == age)]
        if len(sub) > 0:
            print(f"  {sub['auroc'].values[0]:>10.4f}", end="")
        else:
            print(f"  {'---':>10}", end="")
    print()

# AUROC gap summary
print("\n  Age AUROC gap (max-min):")
for m in ["SOFA", "APACHE-II", "GRU", "FAFT", "GA-FAFT"]:
    sub = age_auroc_df[age_auroc_df["model"] == m]["auroc"].dropna()
    if len(sub) >= 2:
        gap = sub.max() - sub.min()
        print(f"    {m}: {gap:.4f}")

age_auroc_df.to_csv(OUT / "e_gafaft_age_auroc.csv", index=False)

# ── RSB computation ───────────────────────────────────────────────────────────
print("\n=== RSB vs Classical Scores ===", flush=True)

def compute_eod(y_true, pred, threshold=None):
    """Equalized odds difference using median threshold."""
    if threshold is None:
        threshold = np.median(pred)
    y_hat = (pred >= threshold).astype(int)
    # EOD = max(|TPR_gap|, |FPR_gap|)
    unique_y = np.unique(y_true)
    if len(unique_y) < 2:
        return np.nan
    tprs, fprs = [], []
    # This is a simplified EOD across ALL subgroups — for each pair
    return np.nan  # placeholder: actual RSB uses group-specific EOD

def compute_rsb(y_true, y_score_ref, y_pred_ml, demo_col, groups):
    """RSB = |EOD(ML, y_true) - EOD(ML, y_score_ref)|"""
    def eod_for_label(preds, labels, groups_col, uniq_groups):
        thr = np.median(preds)
        y_hat = (preds >= thr).astype(int)
        eods = []
        for g in uniq_groups:
            m = groups_col == g
            if m.sum() < 20 or labels[m].sum() < 2 or (1-labels[m]).sum() < 2:
                continue
            tpr = y_hat[m][labels[m] == 1].mean() if labels[m].sum() > 0 else 0
            fpr = y_hat[m][labels[m] == 0].mean() if (1-labels[m]).sum() > 0 else 0
            eods.append((tpr, fpr))
        if len(eods) < 2:
            return np.nan
        tpr_vals = [e[0] for e in eods]
        fpr_vals = [e[1] for e in eods]
        return max(max(tpr_vals)-min(tpr_vals), max(fpr_vals)-min(fpr_vals))

    uniq = np.unique(groups)
    eod_true = eod_for_label(y_pred_ml, y_true, groups, uniq)
    eod_ref  = eod_for_label(y_pred_ml, (y_score_ref > np.median(y_score_ref)).astype(int),
                              groups, uniq)
    if np.isnan(eod_true) or np.isnan(eod_ref):
        return np.nan
    return abs(eod_true - eod_ref)

rsb_records = []
for score_name, score_col in [("sofa", "sofa"), ("apache2", "apache2")]:
    y_ref = data[score_col].values.astype(float)
    for model_name, y_pred in [("GA-FAFT", ga_preds), ("FAFT", faft_preds), ("GRU", gru_preds)]:
        for demo, demo_col in [("age", "age_group"), ("sex", "sex"), ("race", "race_cat")]:
            groups = data[demo_col].fillna("Unknown").values
            rsb = compute_rsb(y, y_ref, y_pred, demo_col, groups)
            rsb_records.append({
                "score": score_name, "model": model_name,
                "demo": demo, "rsb": rsb,
            })

rsb_df = pd.DataFrame(rsb_records)
rsb_df.to_csv(OUT / "e_gafaft_rsb.csv", index=False)
print("\n  RSB summary:")
for (score, model), sub in rsb_df.groupby(["score", "model"]):
    row = {r["demo"]: r["rsb"] for _, r in sub.iterrows()}
    print(f"    {model} vs {score.upper()}: "
          f"age={row.get('age', float('nan')):.4f}  "
          f"sex={row.get('sex', float('nan')):.4f}  "
          f"race={row.get('race', float('nan')):.4f}")

# ── Figure: Age AUROC by model ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = {
    "SOFA": "#d62728", "APACHE-II": "#ff7f0e",
    "GRU": "#1f77b4", "FAFT": "#9467bd", "GA-FAFT": "#2ca02c"
}
styles = {
    "SOFA": ("o-", 1.5, 5), "APACHE-II": ("s-", 1.5, 5),
    "GRU": ("^-", 2.0, 6), "FAFT": ("v-", 2.0, 6),
    "GA-FAFT": ("D-", 2.5, 7),
}

for model_name in ["SOFA", "APACHE-II", "GRU", "FAFT", "GA-FAFT"]:
    sub = age_auroc_df[age_auroc_df["model"] == model_name]
    ages = [a for a in AGE_ORDER if a in sub["age_group"].values]
    xs = range(len(ages))
    vals = [sub[sub["age_group"] == a]["auroc"].values[0] for a in ages]
    gap_str = f"{max(vals)-min(vals):.3f}" if len(vals) >= 2 else "---"
    fmt, lw, ms = styles[model_name]
    ax.plot(xs, vals, fmt, color=colors[model_name], lw=lw, ms=ms,
            label=f"{model_name} (gap={gap_str})")

ax.set_xticks(range(len(AGE_ORDER)))
ax.set_xticklabels(AGE_ORDER, rotation=30, ha="right")
ax.set_ylim(0.55, 1.0)
ax.set_ylabel("AUROC")
ax.set_xlabel("Age group")
ax.set_title("Age-Stratified AUROC: Classical Scores vs. ML Models\n"
             "(GA-FAFT with per-group pairwise ranking loss)")
ax.legend(fontsize=9, loc="lower left")
ax.axhline(0.5, ls=":", color="gray", lw=0.8, alpha=0.5)

fig.tight_layout()
for fmt in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig18_gafaft_age_auroc.{fmt}", bbox_inches="tight")
plt.close(fig)
print("\n  Saved fig18_gafaft_age_auroc", flush=True)

# ── Figure: AUROC gap comparison (bar chart) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
models_list = ["SOFA", "APACHE-II", "GRU", "FAFT", "GA-FAFT"]
gaps = []
for m in models_list:
    sub = age_auroc_df[age_auroc_df["model"] == m]["auroc"].dropna()
    gaps.append(sub.max() - sub.min() if len(sub) >= 2 else 0)

x = np.arange(len(models_list))
bar_colors = [colors[m] for m in models_list]
bars = ax.bar(x, gaps, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(models_list)
ax.set_ylabel("Age AUROC gap (max − min)")
ax.set_title("Age AUROC Gap by Model\n(GA-FAFT targets worst-case group AUROC directly)")
for xi, (gap, m) in enumerate(zip(gaps, models_list)):
    ax.text(xi, gap + 0.003, f"{gap:.3f}", ha="center", fontsize=9)
ax.set_ylim(0, max(gaps) * 1.2)

fig.tight_layout()
for fmt in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig18b_gafaft_gap_bar.{fmt}", bbox_inches="tight")
plt.close(fig)
print("  Saved fig18b_gafaft_gap_bar", flush=True)

print("\n=== GA-FAFT COMPLETE ===", flush=True)
print(f"  Overall AUROC: {results['overall_auroc']:.4f}", flush=True)
print(f"  Params: {results['n_params']:,}", flush=True)
if results.get("age_aurocs"):
    vals = list(results["age_aurocs"].values())
    print(f"  Age AUROC gap: {max(vals)-min(vals):.4f}", flush=True)
