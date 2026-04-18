#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
"""Run E4-E5 + figures from saved GOSSIS E1-E3 data."""
import sys, json, warnings, numpy as np, pandas as pd, torch
from datetime import datetime
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")
DEMO_AXES = ["race_cat", "sex", "age_group", "diag_type"]

data = pd.read_csv('experiments/exp_gossis/cohort_with_scores.csv')
print(f'Loaded {len(data)} stays', flush=True)

# ── E4-E5: GRU + RSB ────────────────────────────────────────────
from src.training.train_gru import train_gru_model
from src.evaluation.rsb import compute_rsb, compute_ml_improvement

device = "cpu"
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        try:
            mem = torch.cuda.mem_get_info(i)
            free_gb, used_gb = mem[0]/1e9, (mem[1]-mem[0])/1e9
            if used_gb < 0.5 and free_gb > 8:
                torch.zeros(1, device=f"cuda:{i}")
                device = f"cuda:{i}"
                print(f"Using GPU {i} ({free_gb:.1f}GB free)", flush=True)
                break
        except Exception:
            continue
print(f"Device: {device}", flush=True)

# Sample 20K for GRU
n_gru = 20000
gru_data = data.sample(n=n_gru, random_state=42)
print(f"Training GRU on {n_gru} stays...", flush=True)

gru_result = train_gru_model(gru_data, device=device, epochs=50)

# Predictions for full data: use mean mortality for non-sampled
ml_preds = np.full(len(data), data["mortality"].mean())
ml_preds[gru_data.index] = gru_result["predictions"]

with open('TRAINING_LOG.md', 'a') as f:
    f.write(f"\n\n## Run GOSSIS-001 — {TIMESTAMP}\n")
    f.write(f"- **Dataset**: GOSSIS ({n_gru} stays)\n")
    f.write(f"- **Device**: {device}\n")
    f.write(f"- **Overall CV AUROC**: {gru_result['overall_auroc']:.4f}\n")
    f.write(f"- **Fold metrics**: {gru_result['fold_metrics']}\n")

print("Computing RSB...", flush=True)
rsb_df = compute_rsb(data, ml_preds, axes=DEMO_AXES, n_boot=500)
rsb_df.to_csv('experiments/exp_gossis/e4_rsb.csv', index=False)

print("Computing ML improvement...", flush=True)
improvement_df = compute_ml_improvement(data, ml_preds, axes=DEMO_AXES)
improvement_df.to_csv('experiments/exp_gossis/e5_ml_improvement.csv', index=False)

with open('RESULTS_GOSSIS.md', 'a') as f:
    f.write(f"\n\n## E4: Reference Standard Bias\n### RSB Gap Summary\n")
    f.write(rsb_df.groupby(['score','metric'])['rsb_gap'].mean().unstack().to_markdown())
    f.write(f"\n\n## E5: ML Fairness Improvement\n")
    f.write(improvement_df.groupby('score')['pct_improvement'].mean().to_markdown())
    f.write(f"\n\n### GRU Performance\n")
    f.write(f"- Overall CV AUROC: {gru_result['overall_auroc']:.4f}\n")
    f.write(f"- Training samples: {n_gru}\n")

print("E4-E5 done.", flush=True)

# ── Figures ──────────────────────────────────────────────────────
from src.evaluation import figures as fig_mod

print("Generating figures...", flush=True)
gaps_df = pd.read_csv('experiments/exp_gossis/e1_gaps.csv')
audit_results = pd.read_csv('experiments/exp_gossis/e1_audit_results.csv')

fig_mod.plot_auroc_gap_heatmap(gaps_df)
fig_mod.plot_subgroup_performance(audit_results, axis="race_cat")
fig_mod.plot_calibration_curves(data)

with open('experiments/exp_gossis/e3_asd_results.json') as f:
    asd_results = json.load(f)
if asd_results:
    fig_mod.plot_asd_results(asd_results)

if len(rsb_df) > 0:
    fig_mod.plot_rsb_gaps(rsb_df)
if len(improvement_df) > 0:
    fig_mod.plot_ml_improvement(improvement_df)
fig_mod.plot_score_distributions(data)
print("Figures done.", flush=True)

with open('RESULTS_GOSSIS.md', 'a') as f:
    f.write(f"\n\n---\n## Pipeline Complete\n**Date**: {TIMESTAMP}\n")

print("PIPELINE COMPLETE.", flush=True)
