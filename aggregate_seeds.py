#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
"""Aggregate multi-seed results and compute mean ± std for paper.

Seeds: 42 (primary), 123, 456
Models: GRU, FAFT, GA-FAFT
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

OUT   = Path("experiments/exp_gossis")
SEEDS = Path("experiments/exp_gossis/seeds")
AGE_ORDER = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

data = pd.read_csv(OUT / "cohort_with_scores.csv")
y    = data["mortality"].values.astype(float)

# ── Seed 42 (primary run) ─────────────────────────────────────────────────────
def compute_metrics(preds, data, y):
    overall = roc_auc_score(y, preds)
    per_age = {}
    for ag in AGE_ORDER:
        m = data["age_group"].values == ag
        yt, sc = y[m], preds[m]
        if yt.sum() < 5 or (1-yt).sum() < 5:
            continue
        per_age[ag] = roc_auc_score(yt, sc)
    gap = max(per_age.values()) - min(per_age.values())
    return overall, gap, per_age


seed42 = {
    "GRU":    np.load(OUT / "ml_preds_full.npy"),
    "FAFT":   np.load(OUT / "faft_preds.npy"),
    "GA-FAFT":np.load(OUT / "ga_faft_preds.npy"),
}

results = {}
for model, preds in seed42.items():
    ov, gap, per_age = compute_metrics(preds, data, y)
    results.setdefault(model, {"overall": [], "gap": [], "per_age": {}})
    results[model]["overall"].append(ov)
    results[model]["gap"].append(gap)
    for ag, auc in per_age.items():
        results[model]["per_age"].setdefault(ag, []).append(auc)

# ── Additional seeds ──────────────────────────────────────────────────────────
for seed in [123, 456]:
    for model in ["GRU", "FAFT", "GA-FAFT"]:
        fname = SEEDS / f"{model.lower().replace('-','')}_seed{seed}.npy"
        if not fname.exists():
            # Try alternative naming
            fname2 = SEEDS / f"{model.replace('-','').lower()}_preds_seed{seed}.npy"
            if not fname2.exists():
                print(f"  Missing: {fname}")
                continue
            fname = fname2
        preds = np.load(fname)
        ov, gap, per_age = compute_metrics(preds, data, y)
        results[model]["overall"].append(ov)
        results[model]["gap"].append(gap)
        for ag, auc in per_age.items():
            results[model]["per_age"].setdefault(ag, []).append(auc)

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n=== Multi-Seed Summary ===")
print(f"{'Model':<12} {'Seeds':>6}  {'AUROC':>14}  {'Age Gap':>14}")
rows = []
for model in ["GRU", "FAFT", "GA-FAFT"]:
    r = results[model]
    n = len(r["overall"])
    ov_mean = np.mean(r["overall"])
    ov_std  = np.std(r["overall"])
    gap_mean = np.mean(r["gap"])
    gap_std  = np.std(r["gap"])
    print(f"  {model:<10} n={n}  {ov_mean:.4f}±{ov_std:.4f}  {gap_mean:.4f}±{gap_std:.4f}")
    rows.append({"model": model, "n_seeds": n,
                 "overall_auroc_mean": ov_mean, "overall_auroc_std": ov_std,
                 "age_gap_mean": gap_mean, "age_gap_std": gap_std})

df = pd.DataFrame(rows)
df.to_csv(OUT / "multiseed_summary.csv", index=False)
print(f"\nSaved: {OUT}/multiseed_summary.csv")
