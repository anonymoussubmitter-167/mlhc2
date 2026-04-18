#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
"""New analyses for ATLAS paper expansion:
  E8: Age-stratified optimal threshold analysis
  E9: Subgroup calibration curves
  E10: FAFT feature attention heatmap (single-fold retrain)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

plt.rcParams.update({
    "font.size": 11, "font.family": "serif",
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

FIGURES = Path("paper/figures")
FIGURES.mkdir(exist_ok=True)

SCORE_LABELS = {"sofa": "SOFA", "qsofa": "qSOFA",
                "apache2": "APACHE-II", "news2": "NEWS2"}
AGE_ORDER = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
AGE_COLORS = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(AGE_ORDER)))

print("Loading data...", flush=True)
data = pd.read_csv("experiments/exp_gossis/cohort_with_scores.csv")
y = data["mortality"].values
print(f"  {len(data)} stays loaded", flush=True)

# ── E8: Age-Stratified Optimal Threshold Analysis ────────────────────────────
print("\n=== E8: Age-stratified optimal thresholds ===", flush=True)

def youden_threshold(y_true, score):
    """Return threshold that maximizes Youden's J = sensitivity + specificity - 1."""
    if y_true.sum() < 5 or (1 - y_true).sum() < 5:
        return np.nan, np.nan, np.nan
    fpr, tpr, thresholds = roc_curve(y_true, score)
    j = tpr - fpr
    idx = np.argmax(j)
    return thresholds[idx], tpr[idx], 1 - fpr[idx]  # threshold, sens, spec

records = []
for score_name in ["sofa", "qsofa", "apache2", "news2"]:
    score_vals = data[score_name].values
    # Overall optimal threshold
    thr_overall, sens_overall, spec_overall = youden_threshold(y, score_vals)

    for age in AGE_ORDER:
        mask = data["age_group"] == age
        if mask.sum() < 30:
            continue
        yt = y[mask]
        sv = score_vals[mask]
        thr, sens, spec = youden_threshold(yt, sv)
        try:
            auroc = roc_auc_score(yt, sv)
        except Exception:
            auroc = np.nan
        records.append({
            "score": score_name, "age_group": age,
            "optimal_threshold": thr,
            "sensitivity": sens, "specificity": spec,
            "j_stat": (sens + spec - 1) if not (np.isnan(sens) or np.isnan(spec)) else np.nan,
            "auroc": auroc,
            "n": mask.sum(),
            "mortality_rate": yt.mean(),
            "overall_threshold": thr_overall,
            "sens_at_overall": np.nan,  # filled below
            "spec_at_overall": np.nan,
        })
        # Sensitivity/specificity at the overall threshold
        if not np.isnan(thr_overall):
            y_bin = (sv >= thr_overall).astype(int)
            if yt.sum() > 0:
                records[-1]["sens_at_overall"] = y_bin[yt == 1].mean()
            if (1 - yt).sum() > 0:
                records[-1]["spec_at_overall"] = 1 - y_bin[yt == 0].mean()

threshold_df = pd.DataFrame(records)
threshold_df.to_csv("experiments/exp_gossis/e8_age_optimal_thresholds.csv", index=False)
print(f"  Saved e8_age_optimal_thresholds.csv ({len(threshold_df)} rows)", flush=True)

# Figure: optimal threshold vs age for each score
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, score_name in enumerate(["sofa", "qsofa", "apache2", "news2"]):
    ax = axes[i]
    sub = threshold_df[threshold_df["score"] == score_name].copy()
    sub["age_group"] = pd.Categorical(sub["age_group"], categories=AGE_ORDER, ordered=True)
    sub = sub.sort_values("age_group")

    ax2 = ax.twinx()
    ax2.bar(range(len(sub)), sub["mortality_rate"] * 100, alpha=0.2,
            color="gray", label="Mortality %")
    ax2.set_ylabel("Mortality rate (%)", color="gray", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="gray")

    ax.plot(range(len(sub)), sub["optimal_threshold"], "o-",
            color="#2166ac", lw=2, ms=7, label="Optimal threshold")
    ax.axhline(sub["overall_threshold"].iloc[0], color="#d62728",
               ls="--", lw=1.5, label=f"Overall optimal ({sub['overall_threshold'].iloc[0]:.1f})")

    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["age_group"], rotation=30, ha="right")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Optimal threshold")
    ax.set_title(SCORE_LABELS[score_name])
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle("E8: Age-Stratified Optimal Clinical Thresholds\n"
             "(Youden's J maximization; bars = group mortality rate)",
             fontsize=12)
fig.tight_layout()
for fmt in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig11_age_optimal_thresholds.{fmt}",
                bbox_inches="tight")
plt.close(fig)
print("  Saved fig11_age_optimal_thresholds", flush=True)

# ── E9: Subgroup Calibration Curves ─────────────────────────────────────────
print("\n=== E9: Subgroup calibration curves ===", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, score_name in enumerate(["sofa", "qsofa", "apache2", "news2"]):
    ax = axes[i]
    score_vals = data[score_name].values
    # Normalize to [0,1] via min-max for calibration curve
    s_min, s_max = np.nanmin(score_vals), np.nanmax(score_vals)
    score_norm = (score_vals - s_min) / max(s_max - s_min, 1e-9)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")

    for j, age in enumerate(AGE_ORDER):
        mask = data["age_group"] == age
        if mask.sum() < 50:
            continue
        yt = y[mask]
        sn = score_norm[mask]
        try:
            frac_pos, mean_pred = calibration_curve(yt, sn, n_bins=8, strategy="quantile")
            brier = brier_score_loss(yt, sn)
            ax.plot(mean_pred, frac_pos, "o-", color=AGE_COLORS[j],
                    lw=1.5, ms=4, alpha=0.85,
                    label=f"{age} (B={brier:.3f})")
        except Exception:
            pass

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction positive (observed mortality)")
    ax.set_title(SCORE_LABELS[score_name])
    ax.legend(fontsize=7, loc="upper left")

fig.suptitle("E9: Calibration Curves by Age Group\n"
             "(Score values normalized to [0,1]; B=Brier score)",
             fontsize=12)
fig.tight_layout()
for fmt in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig12_calibration_by_age.{fmt}",
                bbox_inches="tight")
plt.close(fig)
print("  Saved fig12_calibration_by_age", flush=True)

# ── E10: FAFT Attention Weights ──────────────────────────────────────────────
print("\n=== E10: FAFT attention weights (single fold retrain) ===", flush=True)

import torch
from sklearn.preprocessing import StandardScaler
from src.models.faft import FAFT
from src.training.train_faft import _prepare_features

X_raw, feat_cols = _prepare_features(data)
y_arr = data["mortality"].values.astype(np.float32)
age_groups = data["age_group"].values
race_groups = data["race_cat"].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

# Encode demographic labels
from sklearn.preprocessing import LabelEncoder
age_enc = LabelEncoder().fit(data["age_group"].fillna("Unknown"))
race_enc = LabelEncoder().fit(data["race_cat"].fillna("Unknown"))
n_age = len(age_enc.classes_)
n_race = len(race_enc.classes_)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"  Training single-fold FAFT on {device} for attention extraction...", flush=True)

model = FAFT(n_features=X_scaled.shape[1], n_age_groups=n_age, n_race_groups=n_race,
             d_model=64, n_heads=4, n_layers=2, d_ff=256).to(device)

# Quick train: 20 epochs, subset for speed
N_TRAIN = min(len(data), 20000)
rng = np.random.RandomState(42)
idx = rng.choice(len(data), N_TRAIN, replace=False)
X_t = torch.from_numpy(X_scaled[idx]).to(device)
y_t = torch.from_numpy(y_arr[idx]).to(device)
pos_w = torch.tensor([(1 - y_arr.mean()) / max(y_arr.mean(), 1e-6)]).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)

model.train()
for epoch in range(20):
    for start in range(0, N_TRAIN, 256):
        Xb = X_t[start:start+256]
        yb = y_t[start:start+256]
        logit, _, _ = model(Xb, alpha=0.0)
        loss = criterion(logit, yb)
        opt.zero_grad(); loss.backward(); opt.step()

print("  Training done. Extracting attention weights...", flush=True)

# Extract attention weights using hooks
attention_maps = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        # output[1] is the attention weight matrix when need_weights=True
        # But nn.MultiheadAttention returns (attn_output, attn_weights)
        pass
    return hook

# Re-register attention with need_weights=True by running forward with custom code
model.eval()

# We'll manually extract CLS attention by running the tokenizer + encoder
with torch.no_grad():
    X_sample = X_t[:500]  # sample for visualization
    tokens = model.tokenizer(X_sample)
    cls = model.cls_token.expand(500, -1, -1)
    seq = torch.cat([cls, tokens], dim=1)  # (500, 48, 64)

    # Layer 0 attention
    layer0 = model.encoder[0]
    h = layer0.norm1(seq)
    # Call attention with need_weights=True
    attn_out, attn_weights_l0 = layer0.attn(h, h, h, need_weights=True)
    # attn_weights_l0: (500, 48, 48) — each row is which tokens attend to
    # Row 0 = CLS token attending to all features
    cls_attn_l0 = attn_weights_l0[:, 0, 1:].cpu().numpy()  # (500, 47) - CLS → features
    mean_cls_attn = cls_attn_l0.mean(axis=0)  # (47,) average across samples

# Normalize and sort
attn_norm = mean_cls_attn / mean_cls_attn.sum()
top_idx = np.argsort(attn_norm)[::-1][:20]  # top 20 features by attention
top_feats = [feat_cols[i] for i in top_idx]
top_attn = attn_norm[top_idx]

# Attention by age group
age_attn = {}
for age in AGE_ORDER:
    age_mask = age_groups[idx[:500]] == age
    if age_mask.sum() < 10:
        continue
    age_attn[age] = cls_attn_l0[age_mask].mean(axis=0)

attn_df = pd.DataFrame({
    "feature": feat_cols,
    "mean_attention": attn_norm,
})
attn_df.to_csv("experiments/exp_gossis/e10_faft_attention.csv", index=False)

# Figure: CLS attention heatmap across age groups × top features
n_top = 15
top_feats_15 = [feat_cols[i] for i in top_idx[:n_top]]
top_idx_15 = top_idx[:n_top]

age_keys = [a for a in AGE_ORDER if a in age_attn]
attn_matrix = np.array([age_attn[a][top_idx_15] for a in age_keys])
attn_matrix = attn_matrix / attn_matrix.sum(axis=1, keepdims=True)  # row-normalize

import seaborn as sns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: overall top feature attention
ax1.barh(range(n_top), top_attn[:n_top][::-1], color="#2166ac", alpha=0.8)
ax1.set_yticks(range(n_top))
ax1.set_yticklabels(top_feats_15[::-1], fontsize=8)
ax1.set_xlabel("Mean [CLS] attention weight")
ax1.set_title("(a) Top-15 Features by\nFAFT [CLS] Attention (Layer 1)")

# Panel B: attention by age group
im = ax2.imshow(attn_matrix, aspect="auto", cmap="YlOrRd")
ax2.set_xticks(range(n_top))
ax2.set_xticklabels(top_feats_15, rotation=45, ha="right", fontsize=7)
ax2.set_yticks(range(len(age_keys)))
ax2.set_yticklabels(age_keys, fontsize=9)
ax2.set_title("(b) Attention by Age Group\n(row-normalized)")
plt.colorbar(im, ax=ax2, shrink=0.7)

fig.suptitle("E10: FAFT Feature Attention Analysis\n"
             "(Single-fold retrain on 20K sample; [CLS]→feature attention layer 1)",
             fontsize=11)
fig.tight_layout()
for fmt in ["pdf", "png"]:
    fig.savefig(FIGURES / f"fig13_faft_attention.{fmt}", bbox_inches="tight")
plt.close(fig)
print("  Saved fig13_faft_attention", flush=True)

print("\n=== New analyses COMPLETE ===", flush=True)
