#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
"""Re-run RSB + ML improvement only, using saved predictions. Runs GRU then FAFT sequentially."""
import sys, warnings, numpy as np, pandas as pd
from datetime import datetime
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from src.evaluation.rsb import compute_rsb, compute_ml_improvement

DEMO_AXES = ["race_cat", "sex", "age_group", "diag_type"]
N_BOOT = 200
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")

print(f"=== RSB-only pass — {TIMESTAMP} ===", flush=True)
data = pd.read_csv("experiments/exp_gossis/cohort_with_scores.csv")
print(f"Loaded {len(data)} stays", flush=True)

# ── GRU ───────────────────────────────────────────────────────────────────────
print("\n--- GRU RSB (n_boot={}) ---".format(N_BOOT), flush=True)
gru_preds = np.load("experiments/exp_gossis/ml_preds_full.npy")

rsb_df = compute_rsb(data, gru_preds, axes=DEMO_AXES, n_boot=N_BOOT)
rsb_df.to_csv("experiments/exp_gossis/e4_rsb_full.csv", index=False)
print("GRU RSB saved.", flush=True)
print(rsb_df.groupby(["score","metric"])["rsb_gap"].mean().unstack().to_string(), flush=True)

print("\n--- GRU ML Improvement ---", flush=True)
imp_df = compute_ml_improvement(data, gru_preds, axes=DEMO_AXES)
imp_df.to_csv("experiments/exp_gossis/e5_ml_improvement_full.csv", index=False)
print("GRU ML improvement saved.", flush=True)
print(imp_df.groupby(["score","metric"])["pct_improvement"].mean().unstack().to_string(), flush=True)

# ── FAFT ──────────────────────────────────────────────────────────────────────
print("\n--- FAFT RSB (n_boot={}) ---".format(N_BOOT), flush=True)
faft_preds = np.load("experiments/exp_gossis/faft_preds.npy")

faft_rsb_df = compute_rsb(data, faft_preds, axes=DEMO_AXES, n_boot=N_BOOT)
faft_rsb_df.to_csv("experiments/exp_gossis/e4_rsb_faft.csv", index=False)
print("FAFT RSB saved.", flush=True)
print(faft_rsb_df.groupby(["score","metric"])["rsb_gap"].mean().unstack().to_string(), flush=True)

print("\n--- FAFT ML Improvement ---", flush=True)
faft_imp_df = compute_ml_improvement(data, faft_preds, axes=DEMO_AXES)
faft_imp_df.to_csv("experiments/exp_gossis/e5_ml_improvement_faft.csv", index=False)
print("FAFT ML improvement saved.", flush=True)
print(faft_imp_df.groupby(["score","metric"])["pct_improvement"].mean().unstack().to_string(), flush=True)

print("\n=== RSB-only COMPLETE ===", flush=True)
