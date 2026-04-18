#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
"""Multi-seed robustness evaluation for GRU, FAFT, GA-FAFT.

Runs each model with seeds [123, 456] (seed 42 already in primary results).
Saves per-seed CSVs; aggregate_seeds.py then computes mean±std.
"""
import os, sys, warnings, argparse
import numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--model", choices=["gru", "faft", "gafaft", "all"], default="all")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

SEED = args.seed
OUT  = Path("experiments/exp_gossis/seeds")
OUT.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

AGE_ORDER = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

print(f"=== Multi-seed run: seed={SEED}, model={args.model} ===", flush=True)
data = pd.read_csv("experiments/exp_gossis/cohort_with_scores.csv")
y    = data["mortality"].values.astype(np.float32)
print(f"  {len(data)} stays, {y.mean()*100:.1f}% mortality", flush=True)


def age_auroc_gap(preds, data, y):
    """Returns (overall_auroc, age_auroc_gap, per_age_dict)."""
    overall = roc_auc_score(y, preds)
    per_age = {}
    for ag in AGE_ORDER:
        m = data["age_group"].values == ag
        if m.sum() < 20:
            continue
        yt, sc = y[m], preds[m]
        if yt.sum() < 5 or (1 - yt).sum() < 5:
            continue
        per_age[ag] = roc_auc_score(yt, sc)
    gap = max(per_age.values()) - min(per_age.values())
    return overall, gap, per_age


def save_seed_result(model_name, preds, data, y, seed):
    overall, gap, per_age = age_auroc_gap(preds, data, y)
    rows = [{"seed": seed, "model": model_name,
             "overall_auroc": overall, "age_auroc_gap": gap}]
    for ag, auc in per_age.items():
        rows[0][f"age_{ag}"] = auc
    df = pd.DataFrame(rows)
    fname = OUT / f"{model_name}_seed{seed}.csv"
    df.to_csv(fname, index=False)
    print(f"  → saved {fname}", flush=True)
    print(f"    overall={overall:.4f}  age_gap={gap:.4f}", flush=True)
    return overall, gap


# ── GRU ───────────────────────────────────────────────────────────────────────
if args.model in ("gru", "all"):
    print(f"\n--- GRU (seed={SEED}) ---", flush=True)
    from src.training.train_gru import train_gru_model
    res = train_gru_model(data, device=args.device,
                          hidden_dim=128, n_layers=2, lr=1e-3,
                          epochs=50, batch_size=256, patience=10,
                          cv_seed=SEED)
    preds = res["predictions"]
    np.save(OUT / f"gru_preds_seed{SEED}.npy", preds)
    save_seed_result("GRU", preds, data, y, SEED)


# ── FAFT ──────────────────────────────────────────────────────────────────────
if args.model in ("faft", "all"):
    print(f"\n--- FAFT (seed={SEED}) ---", flush=True)
    from src.training.train_faft import train_faft_model
    res = train_faft_model(data, device=args.device,
                           epochs=50, d_model=64, n_heads=4,
                           n_layers=2, d_ff=256, dropout=0.1,
                           batch_size=256, lr=3e-4, patience=12,
                           adv_lambda=0.3, cv_seed=SEED)
    preds = res["predictions"]
    np.save(OUT / f"faft_preds_seed{SEED}.npy", preds)
    save_seed_result("FAFT", preds, data, y, SEED)


# ── GA-FAFT ───────────────────────────────────────────────────────────────────
if args.model in ("gafaft", "all"):
    print(f"\n--- GA-FAFT (seed={SEED}) ---", flush=True)
    from src.training.train_ga_faft import train_ga_faft_model
    res = train_ga_faft_model(data, device=args.device,
                              epochs=60, d_model=64, n_heads=4,
                              n_layers=2, d_ff=256, dropout=0.1,
                              batch_size=256, lr=3e-4, patience=12,
                              adv_lambda=0.3, rank_lambda=0.5,
                              rank_margin=1.0, rank_mode="max",
                              cv_seed=SEED)
    preds = res["predictions"]
    np.save(OUT / f"gafaft_preds_seed{SEED}.npy", preds)
    save_seed_result("GA-FAFT", preds, data, y, SEED)

print("\n=== Done ===", flush=True)
