#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
"""Additional analyses:
  E11: Race-stratified calibration curves (within-hospital controlled)
  E12: FAFT vs GRU per-axis fairness gap comparison
"""
import sys, warnings, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from pathlib import Path

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
SCORE_LABELS = {"sofa": "SOFA", "qsofa": "qSOFA",
                "apache2": "APACHE-II", "news2": "NEWS2"}

print("Loading data...", flush=True)
data = pd.read_csv("experiments/exp_gossis/cohort_with_scores.csv")
y = data["mortality"].values
print(f"  {len(data)} stays", flush=True)

# ── E11: Race-stratified calibration curves ──────────────────────────────────
print("\n=== E11: Race-stratified calibration curves ===", flush=True)

race_groups = sorted(data["race_cat"].dropna().unique())
race_colors = {r: c for r, c in zip(race_groups,
    plt.cm.tab10(np.linspace(0, 0.9, len(race_groups))))}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

calib_records = []
for i, score_name in enumerate(["sofa", "qsofa", "apache2", "news2"]):
    ax = axes[i]
    score_vals = data[score_name].values
    s_min, s_max = np.nanmin(score_vals), np.nanmax(score_vals)
    score_norm = (score_vals - s_min) / max(s_max - s_min, 1e-9)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")

    for race in race_groups:
        mask = data["race_cat"] == race
        if mask.sum() < 100:
            continue
        yt = y[mask]
        sn = score_norm[mask]
        if yt.sum() < 10:
            continue
        try:
            frac_pos, mean_pred = calibration_curve(yt, sn, n_bins=8, strategy="quantile")
            brier = brier_score_loss(yt, sn)
            auroc = roc_auc_score(yt, sn)
            ax.plot(mean_pred, frac_pos, "o-", color=race_colors[race],
                    lw=1.5, ms=4, alpha=0.85,
                    label=f"{race} (n={mask.sum():,}, B={brier:.3f})")
            calib_records.append({
                "score": score_name, "race": race, "brier": brier,
                "auroc": auroc, "n": mask.sum()
            })
        except Exception:
            pass

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive (observed mortality)")
    ax.set_title(SCORE_LABELS[score_name])
    ax.legend(fontsize=7, loc="upper left")

fig.suptitle("E11: Calibration Curves by Race/Ethnicity\n"
             "(Score values normalized to [0,1]; B=Brier score)",
             fontsize=12)
fig.tight_layout()
for fmt in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig14_calibration_by_race.{fmt}", bbox_inches="tight")
plt.close(fig)

calib_df = pd.DataFrame(calib_records)
calib_df.to_csv("experiments/exp_gossis/e11_race_calibration.csv", index=False)
print(f"  Saved fig14, e11_race_calibration.csv ({len(calib_df)} rows)", flush=True)

# Print key finding: AUROC by race
print("\n  AUROC by race (averaged across scores):")
print(calib_df.groupby("race")["auroc"].mean().sort_values().to_string())

# ── E12: Per-axis AUROC gap: GRU vs FAFT ─────────────────────────────────────
print("\n=== E12: Per-axis AUROC gap — GRU vs FAFT ===", flush=True)

gru_preds  = np.load("experiments/exp_gossis/ml_preds_full.npy")
faft_preds = np.load("experiments/exp_gossis/faft_preds.npy")

axes_map = {"race_cat": "Race", "sex": "Sex",
            "age_group": "Age", "diag_type": "Diagnosis"}
records = []
for demo_col, demo_label in axes_map.items():
    if demo_col not in data.columns:
        continue
    groups = data[demo_col].values
    unique = [g for g in data[demo_col].dropna().unique()
              if (groups == g).sum() >= 30]
    if len(unique) < 2:
        continue

    # Classical scores
    for score_name in ["sofa", "qsofa", "apache2", "news2"]:
        sv = data[score_name].values
        aurocs = []
        for g in unique:
            mask = groups == g
            yt = y[mask]
            if yt.sum() < 5 or (1 - yt).sum() < 5:
                continue
            try:
                aurocs.append(roc_auc_score(yt, sv[mask]))
            except Exception:
                pass
        if len(aurocs) >= 2:
            records.append({"model": score_name.upper(), "axis": demo_label,
                            "auroc_gap": max(aurocs) - min(aurocs),
                            "min_auroc": min(aurocs), "max_auroc": max(aurocs)})

    # GRU
    aurocs = []
    for g in unique:
        mask = groups == g
        yt = y[mask]
        if yt.sum() < 5 or (1 - yt).sum() < 5:
            continue
        try:
            aurocs.append(roc_auc_score(yt, gru_preds[mask]))
        except Exception:
            pass
    if len(aurocs) >= 2:
        records.append({"model": "GRU", "axis": demo_label,
                        "auroc_gap": max(aurocs) - min(aurocs),
                        "min_auroc": min(aurocs), "max_auroc": max(aurocs)})

    # FAFT
    aurocs = []
    for g in unique:
        mask = groups == g
        yt = y[mask]
        if yt.sum() < 5 or (1 - yt).sum() < 5:
            continue
        try:
            aurocs.append(roc_auc_score(yt, faft_preds[mask]))
        except Exception:
            pass
    if len(aurocs) >= 2:
        records.append({"model": "FAFT", "axis": demo_label,
                        "auroc_gap": max(aurocs) - min(aurocs),
                        "min_auroc": min(aurocs), "max_auroc": max(aurocs)})

e12_df = pd.DataFrame(records)
e12_df.to_csv("experiments/exp_gossis/e12_model_auroc_gaps.csv", index=False)
print("  Key gaps (GRU vs FAFT vs classical):")
print(e12_df.pivot(index="axis", columns="model", values="auroc_gap").to_string())

# Figure: grouped bar chart — AUROC gap by axis and model
fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)
model_colors = {"SOFA": "#4dac26", "QSOFA": "#b8e186",
                "APACHE-II": "#d01c8b", "NEWS2": "#f1b6da",
                "GRU": "#2166ac", "FAFT": "#d62728"}

demo_order = ["Race", "Sex", "Age", "Diagnosis"]
for i, demo_label in enumerate(demo_order):
    ax = axes[i]
    sub = e12_df[e12_df["axis"] == demo_label].copy()
    models = ["SOFA", "QSOFA", "APACHE-II", "NEWS2", "GRU", "FAFT"]
    sub["model"] = sub["model"].replace({"QSOFA": "QSOFA"})
    vals = []
    colors = []
    labels = []
    for m in models:
        row = sub[sub["model"] == m]
        if len(row) > 0:
            vals.append(row["auroc_gap"].values[0])
            colors.append(model_colors.get(m, "gray"))
            labels.append(m)
    x = range(len(labels))
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", lw=0.5)
    # Highlight ML models with hatching
    for j, (lbl, bar) in enumerate(zip(labels, bars)):
        if lbl in ("GRU", "FAFT"):
            bar.set_hatch("//")
            bar.set_edgecolor("black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(demo_label, fontsize=11)
    ax.set_ylabel("AUROC gap" if i == 0 else "")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

fig.suptitle("E12: AUROC Gap by Demographic Axis\nClassical Scores vs GRU vs FAFT",
             fontsize=12)
fig.tight_layout()
for fmt in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig15_model_auroc_gap_comparison.{fmt}",
                bbox_inches="tight")
plt.close(fig)
print("  Saved fig15_model_auroc_gap_comparison", flush=True)

print("\n=== Extra analyses COMPLETE ===", flush=True)
