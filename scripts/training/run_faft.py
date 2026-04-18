#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
"""Train FAFT (Fairness-Aware Feature Transformer) on full GOSSIS cohort.

FAFT architecture:
  - Feature tokenization (each scalar feature → learned d-dim token)
  - Transformer self-attention over feature tokens
  - [CLS] token → mortality head
  - Gradient reversal adversarial heads (age_group, race_cat)
    → encoder learns to predict mortality but resist demographic prediction

Computes RSB and ML-improvement metrics; saves results for paper update.
"""
import os, sys, warnings, numpy as np, pandas as pd, torch
from datetime import datetime
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")
DEMO_AXES = ["race_cat", "sex", "age_group", "diag_type"]

print(f"=== FAFT Training — {TIMESTAMP} ===", flush=True)

data = pd.read_csv("experiments/exp_gossis/cohort_with_scores.csv")
print(f"Loaded {len(data)} stays", flush=True)

from src.training.train_faft import train_faft_model
from src.evaluation.rsb import compute_rsb, compute_ml_improvement

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

print(f"Training FAFT on full cohort ({len(data)} stays)...", flush=True)
faft_result = train_faft_model(
    data,
    device=device,
    epochs=50,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    dropout=0.1,
    batch_size=256,
    lr=3e-4,
    patience=10,
    adv_lambda=0.3,
)

faft_preds = faft_result["predictions"]
print(f"Full-cohort FAFT CV AUROC: {faft_result['overall_auroc']:.4f}", flush=True)

# Save predictions
np.save("experiments/exp_gossis/faft_preds.npy", faft_preds)

# Log training run
with open("TRAINING_LOG.md", "a") as f:
    f.write(f"\n\n## FAFT Run — {TIMESTAMP}\n")
    f.write(f"- **Dataset**: GOSSIS (FULL, {len(data)} stays)\n")
    f.write(f"- **Device**: {device}\n")
    f.write(f"- **Architecture**: FAFT (d=64, heads=4, layers=2, ff=256, adv_lambda=0.3)\n")
    f.write(f"- **Overall CV AUROC**: {faft_result['overall_auroc']:.4f}\n")
    f.write(f"- **Fold metrics**: {faft_result['fold_metrics']}\n")

# RSB with FAFT predictions
print("Computing RSB (FAFT)...", flush=True)
faft_rsb_df = compute_rsb(data, faft_preds, axes=DEMO_AXES, n_boot=200)
faft_rsb_df.to_csv("experiments/exp_gossis/e4_rsb_faft.csv", index=False)

# ML improvement with FAFT predictions
print("Computing ML improvement (FAFT)...", flush=True)
faft_improvement_df = compute_ml_improvement(data, faft_preds, axes=DEMO_AXES)
faft_improvement_df.to_csv("experiments/exp_gossis/e5_ml_improvement_faft.csv", index=False)

print("\n=== RSB Summary (FAFT) ===", flush=True)
print(faft_rsb_df.groupby(["score","metric"])["rsb_gap"].mean().unstack().to_string())
print("\n=== ML Improvement Summary (FAFT) ===", flush=True)
print(faft_improvement_df.groupby(["score","metric"])["pct_improvement"].mean().unstack().to_string())

with open("RESULTS_GOSSIS.md", "a") as f:
    f.write(f"\n\n---\n## FAFT Results — {TIMESTAMP}\n\n")
    f.write(f"**FAFT AUROC (full {len(data)} stays)**: {faft_result['overall_auroc']:.4f}\n")
    f.write(f"**Architecture**: d_model=64, n_heads=4, n_layers=2, d_ff=256, adv_lambda=0.3\n")
    f.write(f"**Fold metrics**: {faft_result['fold_metrics']}\n\n")
    f.write("### E4 RSB (FAFT)\n")
    f.write(faft_rsb_df.groupby(["score","metric"])["rsb_gap"].mean().unstack().to_markdown())
    f.write("\n\n### E5 ML Improvement (FAFT)\n")
    f.write(faft_improvement_df.groupby(["score","metric"])["pct_improvement"].mean().unstack().to_markdown())

print("\n=== FAFT COMPLETE ===", flush=True)
