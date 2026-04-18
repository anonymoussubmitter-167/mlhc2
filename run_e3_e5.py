#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
"""Run E3-E5 from saved E1/E2 cohort data."""
import sys, json, warnings, numpy as np, pandas as pd, torch
from datetime import datetime
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")

data = pd.read_csv('experiments/exp_eicu/cohort_with_scores.csv')
print(f'Loaded {len(data)} stays', flush=True)

# ── E3: ASD ──────────────────────────────────────────────────────
from src.evaluation.asd import adversarial_subgroup_discovery
print('Running E3: ASD...', flush=True)
asd_results = adversarial_subgroup_discovery(data)

with open('experiments/exp_eicu/e3_asd_results.json', 'w') as f:
    serializable = {}
    for k, v in asd_results.items():
        sv = {**v}
        sv['top_features'] = [(fn, float(fi)) for fn, fi in sv['top_features']]
        serializable[k] = sv
    json.dump(serializable, f, indent=2, default=str)

asd_text = ''
for score_name, res in asd_results.items():
    asd_text += f'\n### {score_name.upper()}\n'
    asd_text += f'- Error prediction AUROC: {res["error_prediction_auroc"]}\n'
    asd_text += f'- Top features: {[f[0] for f in res["top_features"][:5]]}\n'
    for i, sg in enumerate(res['vulnerable_subgroups']):
        asd_text += (f'- Subgroup {i+1}: n={sg["n"]}, '
                     f'error_conc={sg["concentration_ratio"]:.2f}\n')

with open('RESULTS_EICU.md', 'a') as f:
    f.write(f'\n\n## E3: Adversarial Subgroup Discovery\n{asd_text}\n')
print('E3 done.', flush=True)

# ── E4-E5: GRU + RSB ────────────────────────────────────────────
from src.training.train_gru import train_gru_model
from src.evaluation.rsb import compute_rsb, compute_ml_improvement

demo_axes = ["race_cat", "sex", "age_group", "diag_type"]

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
                print(f"Using GPU {i} ({free_gb:.1f} GB free)", flush=True)
                break
        except Exception:
            continue
if device == "cpu":
    print("Using CPU", flush=True)

print('Running E4-E5: GRU training...', flush=True)
gru_result = train_gru_model(data, device=device, epochs=50)
ml_preds = gru_result["predictions"]

with open('TRAINING_LOG.md', 'a') as f:
    f.write(f'\n\n## Run eICU-001 — {TIMESTAMP}\n')
    f.write(f'- **Dataset**: eICU-CRD Demo ({len(data)} stays)\n')
    f.write(f'- **Hardware**: {device}\n')
    f.write(f'- **Overall CV AUROC**: {gru_result["overall_auroc"]:.4f}\n')
    f.write(f'- **Model params**: {gru_result["n_params"]:,}\n')

print('Computing RSB gaps...', flush=True)
rsb_df = compute_rsb(data, ml_preds, axes=demo_axes, n_boot=500)
rsb_df.to_csv('experiments/exp_eicu/e4_rsb.csv', index=False)

print('Computing ML improvement...', flush=True)
improvement_df = compute_ml_improvement(data, ml_preds, axes=demo_axes)
improvement_df.to_csv('experiments/exp_eicu/e5_ml_improvement.csv', index=False)

with open('RESULTS_EICU.md', 'a') as f:
    f.write(f'\n\n## E4: Reference Standard Bias\n\n### RSB Gap Summary\n')
    f.write(rsb_df.groupby(['score', 'metric'])['rsb_gap'].mean().unstack().to_markdown())
    f.write(f'\n\n## E5: ML Fairness Improvement\n')
    f.write(f'### Mean improvement over classical scores\n')
    f.write(improvement_df.groupby('score')['pct_improvement'].mean().to_markdown())
    f.write(f'\n\n### GRU Model Performance\n')
    f.write(f'- Overall CV AUROC: {gru_result["overall_auroc"]:.4f}\n')
    f.write(f'- Model parameters: {gru_result["n_params"]:,}\n')

print('E4-E5 done.', flush=True)

# ── Figures ──────────────────────────────────────────────────────
from src.evaluation import figures as fig_mod
from src.evaluation.audit import prespecified_audit

print('Generating figures...', flush=True)
audit_results = pd.read_csv('experiments/exp_eicu/e1_audit_results.csv')
gaps_df = pd.read_csv('experiments/exp_eicu/e1_gaps.csv')

fig_mod.plot_auroc_gap_heatmap(gaps_df)
fig_mod.plot_subgroup_performance(audit_results, axis="race_cat")
fig_mod.plot_calibration_curves(data)
if asd_results:
    fig_mod.plot_asd_results(asd_results)
if len(rsb_df) > 0:
    fig_mod.plot_rsb_gaps(rsb_df)
if len(improvement_df) > 0:
    fig_mod.plot_ml_improvement(improvement_df)
fig_mod.plot_score_distributions(data)

print('All figures generated.', flush=True)

with open('RESULTS_EICU.md', 'a') as f:
    f.write(f'\n\n---\n## Pipeline Complete\n**Date**: {TIMESTAMP}\n')

print('PIPELINE COMPLETE.', flush=True)
