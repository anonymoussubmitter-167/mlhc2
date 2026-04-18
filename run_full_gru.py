#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
"""Full-cohort GRU training — no subsample cap.
Trains on all 87,315 GOSSIS stays; re-computes RSB and ML improvement."""
import sys, warnings, numpy as np, pandas as pd, torch
from datetime import datetime
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")
DEMO_AXES = ["race_cat", "sex", "age_group", "diag_type"]

print(f"=== Full-Cohort GRU — {TIMESTAMP} ===", flush=True)

data = pd.read_csv('experiments/exp_gossis/cohort_with_scores.csv')
print(f"Loaded {len(data)} stays", flush=True)

from src.training.train_gru import train_gru_model
from src.evaluation.rsb import compute_rsb, compute_ml_improvement

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda:0"
print(f"Device: {device} (GPU 1)", flush=True)

# Full cohort — no subsample
n_gru = len(data)
print(f"Training GRU on FULL cohort ({n_gru} stays)...", flush=True)
gru_result = train_gru_model(data, device=device, epochs=50)

ml_preds = gru_result["predictions"]  # full-cohort CV predictions

print(f"Full-cohort GRU CV AUROC: {gru_result['overall_auroc']:.4f}", flush=True)

with open('TRAINING_LOG.md', 'a') as f:
    f.write(f"\n\n## Run GOSSIS-FULL — {TIMESTAMP}\n")
    f.write(f"- **Dataset**: GOSSIS (FULL, {n_gru} stays)\n")
    f.write(f"- **Device**: {device}\n")
    f.write(f"- **Overall CV AUROC**: {gru_result['overall_auroc']:.4f}\n")
    f.write(f"- **Fold metrics**: {gru_result['fold_metrics']}\n")

np.save('experiments/exp_gossis/ml_preds_full.npy', ml_preds)

print("Computing RSB (full-cohort)...", flush=True)
rsb_df = compute_rsb(data, ml_preds, axes=DEMO_AXES, n_boot=200)
rsb_df.to_csv('experiments/exp_gossis/e4_rsb_full.csv', index=False)

print("Computing ML improvement (full-cohort)...", flush=True)
improvement_df = compute_ml_improvement(data, ml_preds, axes=DEMO_AXES)
improvement_df.to_csv('experiments/exp_gossis/e5_ml_improvement_full.csv', index=False)

print("\n=== RSB Summary (full cohort) ===")
print(rsb_df.groupby(['score','metric'])['rsb_gap'].mean().unstack().to_string())
print("\n=== ML Improvement Summary (full cohort) ===")
print(improvement_df.groupby(['score','metric'])['pct_improvement'].mean().unstack().to_string())

with open('RESULTS_GOSSIS.md', 'a') as f:
    f.write(f"\n\n---\n## Full-Cohort GRU Results — {TIMESTAMP}\n\n")
    f.write(f"**GRU AUROC (full {n_gru} stays)**: {gru_result['overall_auroc']:.4f}\n")
    f.write(f"**Fold metrics**: {gru_result['fold_metrics']}\n\n")
    f.write("### E4 RSB (full cohort)\n")
    f.write(rsb_df.groupby(['score','metric'])['rsb_gap'].mean().unstack().to_markdown())
    f.write("\n\n### E5 ML Improvement (full cohort)\n")
    f.write(improvement_df.groupby(['score','metric'])['pct_improvement'].mean().unstack().to_markdown())

print("\n=== FULL-COHORT GRU COMPLETE ===", flush=True)
