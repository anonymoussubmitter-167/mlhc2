#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
# Part of ATLAS
"""Main pipeline: cohort extraction -> score computation -> all experiments."""

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

# Project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.cohort import extract_cohort
from src.data.scores import compute_all_scores
from src.data.config import (FIGURES_DIR, EXPERIMENTS_DIR, RANDOM_SEED,
                              MIN_SUBGROUP_SIZE, BOOTSTRAP_ITERATIONS)
from src.evaluation.audit import prespecified_audit, intersectional_audit
from src.evaluation.asd import adversarial_subgroup_discovery
from src.training.train_gru import train_gru_model
from src.evaluation.rsb import compute_rsb, compute_ml_improvement
from src.evaluation import figures as fig_mod

np.random.seed(RANDOM_SEED)

# ── Output directories ──────────────────────────────────────────────────
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR = EXPERIMENTS_DIR / "exp_001"
EXP_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")


def update_results(section: str, content: str):
    """Append to RESULTS.md."""
    with open(ROOT / "RESULTS.md", "a") as f:
        f.write(f"\n\n{content}")
    print(f"[RESULTS] Updated: {section}")


def update_training_log(entry: str):
    """Append to TRAINING_LOG.md."""
    with open(ROOT / "TRAINING_LOG.md", "a") as f:
        f.write(f"\n\n{entry}")


def main():
    start = time.time()

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1: Cohort extraction + score computation
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PHASE 1: Cohort Extraction & Score Computation")
    print("=" * 70)

    cohort = extract_cohort()
    data = compute_all_scores(cohort)

    # Save
    data.to_csv(EXP_DIR / "cohort_with_scores.csv", index=False)

    update_results("Cohort Summary", f"""## Cohort Summary
**Date**: {TIMESTAMP}
**N**: {len(data)} ICU stays
**Mortality rate**: {data['mortality'].mean():.3f}

### Demographics
| Attribute | Distribution |
|-----------|-------------|
| Race | {data['race_cat'].value_counts().to_dict()} |
| Sex | {data['sex'].value_counts().to_dict()} |
| Age (mean) | {data['age'].mean():.1f} +/- {data['age'].std():.1f} |
| Insurance | {data['insurance_cat'].value_counts().to_dict()} |

### Score Distributions
| Score | Mean | Median | Range |
|-------|------|--------|-------|
| SOFA | {data['sofa'].mean():.1f} | {data['sofa'].median():.0f} | [{data['sofa'].min():.0f}, {data['sofa'].max():.0f}] |
| qSOFA | {data['qsofa'].mean():.1f} | {data['qsofa'].median():.0f} | [{data['qsofa'].min():.0f}, {data['qsofa'].max():.0f}] |
| APACHE-II | {data['apache2'].mean():.1f} | {data['apache2'].median():.0f} | [{data['apache2'].min():.0f}, {data['apache2'].max():.0f}] |
| NEWS2 | {data['news2'].mean():.1f} | {data['news2'].median():.0f} | [{data['news2'].min():.0f}, {data['news2'].max():.0f}] |
""")

    # ═══════════════════════════════════════════════════════════════════
    # E1: Pre-specified subgroup audit
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("E1: Pre-specified Subgroup Audit")
    print("=" * 70)

    # Reduce bootstrap for small datasets
    n_boot = min(BOOTSTRAP_ITERATIONS, 200) if len(data) < 1000 else BOOTSTRAP_ITERATIONS

    audit_results, gaps_df = prespecified_audit(data, n_boot=n_boot)
    audit_results.to_csv(EXP_DIR / "e1_audit_results.csv", index=False)
    gaps_df.to_csv(EXP_DIR / "e1_gaps.csv", index=False)

    update_results("E1 Results", f"""## E1: Pre-specified Subgroup Audit
**Reference**: RESEARCH_PLAN.md §4.1

### AUROC Gaps (max - min across subgroups)
{gaps_df.to_markdown(index=False) if hasattr(gaps_df, 'to_markdown') else gaps_df.to_string()}

### Full Results
{audit_results.to_markdown(index=False) if hasattr(audit_results, 'to_markdown') else audit_results.to_string()}
""")

    # ════════════════════════════════════════════════════════════════════
    # E2: Intersectional analysis
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("E2: Intersectional Analysis")
    print("=" * 70)

    # Use smaller min_n for demo dataset
    min_n = min(MIN_SUBGROUP_SIZE, max(20, len(data) // 10))
    inter_results, worst_subgroups = intersectional_audit(data, min_n=min_n,
                                                          n_boot=n_boot)
    inter_results.to_csv(EXP_DIR / "e2_intersectional.csv", index=False)

    worst_text = ""
    for score_name, worst_df in worst_subgroups.items():
        worst_text += f"\n### {score_name.upper()} — Worst Subgroups\n"
        if len(worst_df) > 0:
            worst_text += worst_df[["subgroup", "auroc", "n", "prevalence"]].to_markdown(index=False)
        else:
            worst_text += "No subgroups met minimum size threshold.\n"

    update_results("E2 Results", f"""## E2: Intersectional Analysis
**Reference**: RESEARCH_PLAN.md §4.2
**Min subgroup size**: {min_n}
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
        # Convert to serializable format
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
**Reference**: RESEARCH_PLAN.md §4.3
{asd_text}
""")

    # ═══════════════════════════════════════════════════════════════════
    # E4-E5: GRU Training + RSB Quantification
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("E4-E5: GRU Training & RSB Quantification")
    print("=" * 70)

    # Select GPU device — skip GPUs with existing processes
    import torch
    if torch.cuda.is_available():
        device = "cpu"  # fallback
        for i in range(torch.cuda.device_count()):
            try:
                mem = torch.cuda.mem_get_info(i)
                free_gb = mem[0] / 1e9
                total_gb = mem[1] / 1e9
                used_gb = total_gb - free_gb
                # Only use GPUs with <500MB in use (essentially empty)
                if used_gb < 0.5 and free_gb > 8:
                    # Quick test to make sure GPU works
                    t = torch.zeros(1, device=f"cuda:{i}")
                    del t
                    device = f"cuda:{i}"
                    print(f"Using GPU {i} ({free_gb:.1f} GB free)")
                    break
            except Exception:
                continue
        if device == "cpu":
            print("Using CPU (no suitable GPU found)")
    else:
        device = "cpu"

    gru_result = train_gru_model(data, device=device,
                                  epochs=50 if len(data) > 500 else 30)
    ml_preds = gru_result["predictions"]

    update_training_log(f"""## Run 001 — {TIMESTAMP}
- **Experiment**: E4-E5 GRU mortality prediction
- **Config**:
  - Model: MortalityGRU (hidden=128, layers=2, dropout=0.3)
  - LR: 1e-3, Schedule: ReduceLROnPlateau
  - Batch size: 256
  - Epochs: 50 (early stopping patience=10)
  - 5-fold CV
- **Data**: {len(data)} ICU stays, {data['mortality'].mean():.3f} mortality rate
- **Hardware**: {device}
- **Final Metrics**:
  - Overall CV AUROC: {gru_result['overall_auroc']:.4f}
  - Model params: {gru_result['n_params']:,}
- **Fold metrics**: {gru_result['fold_metrics']}
- **Status**: Completed
""")

    # RSB quantification
    print("\nComputing RSB gaps...")
    rsb_df = compute_rsb(data, ml_preds, n_boot=min(n_boot, 500))
    rsb_df.to_csv(EXP_DIR / "e4_rsb.csv", index=False)

    print("Computing ML improvement...")
    improvement_df = compute_ml_improvement(data, ml_preds)
    improvement_df.to_csv(EXP_DIR / "e5_ml_improvement.csv", index=False)

    update_results("E4-E5 Results", f"""## E4: Reference Standard Bias
**Reference**: RESEARCH_PLAN.md §4.4

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

    print("Figure 1: AUROC gap heatmap...")
    if len(gaps_df) > 0:
        fig_mod.plot_auroc_gap_heatmap(gaps_df)

    print("Figure 2: AUROC by race...")
    fig_mod.plot_subgroup_performance(audit_results, axis="race_cat")

    print("Figure 2b: AUROC by insurance...")
    fig_mod.plot_subgroup_performance(audit_results, axis="insurance_cat")

    print("Figure 3: Calibration curves by race...")
    fig_mod.plot_calibration_curves(data)

    print("Figure 4: ASD results...")
    if asd_results:
        fig_mod.plot_asd_results(asd_results)

    print("Figure 5: RSB gap heatmap...")
    if len(rsb_df) > 0:
        fig_mod.plot_rsb_gaps(rsb_df)

    print("Figure 6: ML improvement...")
    if len(improvement_df) > 0:
        fig_mod.plot_ml_improvement(improvement_df)

    print("Figure S1: Score distributions...")
    fig_mod.plot_score_distributions(data)

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'=' * 70}")
    print(f"Results: {EXP_DIR}")
    print(f"Figures: {FIGURES_DIR}")

    update_results("Pipeline Complete", f"""---
## Pipeline Complete
**Duration**: {elapsed/60:.1f} minutes
**Date**: {TIMESTAMP}
""")


if __name__ == "__main__":
    main()
