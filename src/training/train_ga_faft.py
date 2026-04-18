# MIT License — Anonymous Authors, 2026
"""Training loop for Group-Adaptive Fairness-Aware Feature Transformer (GA-FAFT).

Novel training objective vs. FAFT:
  L = L_CE + λ_rank * L_rank(groups) + λ_adv * L_adv
  where L_rank = max_g[-AUC_g] (worst-case group AUROC via pairwise sigmoid)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

from src.models.ga_faft import GAFAFT


def _prepare_features(data):
    """Same feature exclusion as FAFT for fair comparison."""
    exclude = ["subject_id", "hadm_id", "stay_id", "intime", "outtime",
               "mortality", "race_cat", "sex", "age_group", "insurance_cat",
               "diag_type", "first_careunit", "sofa", "qsofa", "apache2",
               "news2", "race_raw", "los", "hospitalid"]
    feat_cols = [c for c in data.columns if c not in exclude
                 and data[c].dtype in [np.float64, np.int64,
                                        np.float32, np.int32]]
    X = data[feat_cols].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    X = X.fillna(0)
    return X.values.astype(np.float32), feat_cols


def _alpha_schedule(step: int, total_steps: int, max_alpha: float = 0.3) -> float:
    """Ganin et al. (2016) GRL schedule."""
    p = step / max(total_steps, 1)
    return max_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def _lambda_rank_schedule(step: int, warmup_steps: int,
                          max_lambda: float = 0.5) -> float:
    """Warm up ranking loss weight gradually (0 → max_lambda over warmup_steps).
    Keep at 0 during first warmup steps so CE loss establishes a good baseline
    before ranking loss kicks in. Prevents gradient instability at start.
    """
    if step < warmup_steps:
        return 0.0
    progress = (step - warmup_steps) / max(warmup_steps, 1)
    return min(max_lambda, max_lambda * progress)


def train_ga_faft_model(data,
                        device: str = "cpu",
                        epochs: int = 60,
                        d_model: int = 64,
                        n_heads: int = 4,
                        n_layers: int = 2,
                        d_ff: int = 256,
                        dropout: float = 0.1,
                        batch_size: int = 256,
                        lr: float = 3e-4,
                        patience: int = 12,
                        adv_lambda: float = 0.3,
                        rank_lambda: float = 0.5,
                        rank_margin: float = 1.0,
                        rank_mode: str = "max",
                        cv_seed: int = 42):
    """Train GA-FAFT with 5-fold cross-validation.

    Returns dict with keys:
        predictions, overall_auroc, fold_metrics, group_aurocs, n_params
    """
    X_raw, feat_cols = _prepare_features(data)
    y = data["mortality"].values.astype(np.float32)

    # Encode demographic labels
    age_enc  = LabelEncoder().fit(data["age_group"].fillna("Unknown"))
    race_enc = LabelEncoder().fit(data["race_cat"].fillna("Unknown"))
    age_labels  = age_enc.transform(data["age_group"].fillna("Unknown")).astype(np.int64)
    race_labels = race_enc.transform(data["race_cat"].fillna("Unknown")).astype(np.int64)
    n_age  = len(age_enc.classes_)
    n_race = len(race_enc.classes_)

    print(f"  Features: {X_raw.shape[1]}", flush=True)
    print(f"  Age groups ({n_age}): {list(age_enc.classes_)}", flush=True)
    print(f"  Race groups ({n_race}): {list(race_enc.classes_)}", flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    all_preds = np.zeros(len(data), dtype=np.float32)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_raw, y)):
        print(f"\n  Fold {fold+1}/5:", flush=True)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_raw[train_idx])
        X_val   = scaler.transform(X_raw[val_idx])

        y_train    = y[train_idx]
        y_val      = y[val_idx]
        age_train  = age_labels[train_idx]
        age_val    = age_labels[val_idx]
        race_train = race_labels[train_idx]
        race_val   = race_labels[val_idx]

        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(age_train),
            torch.from_numpy(race_train),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
            torch.from_numpy(age_val),
            torch.from_numpy(race_val),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=True)  # drop_last for stable batch stats
        val_loader   = DataLoader(val_ds, batch_size=batch_size)

        model = GAFAFT(
            n_features=X_train.shape[1],
            n_age_groups=n_age,
            n_race_groups=n_race,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            rank_margin=rank_margin,
            rank_max_pairs=512,
            rank_mode=rank_mode,
        ).to(device)

        pos_weight = torch.tensor(
            [(1 - y_train.mean()) / max(y_train.mean(), 1e-6)]
        ).to(device)
        mort_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        total_steps  = epochs * len(train_loader)
        warmup_steps = max(1, total_steps // 6)  # 1/6 of training for CE warm-up

        best_val_auroc = 0.0
        best_state     = None
        no_improve     = 0
        global_step    = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = {"total": 0, "ce": 0, "rank": 0, "adv": 0}
            n_batches = 0

            for Xb, yb, age_b, race_b in train_loader:
                Xb     = Xb.to(device)
                yb     = yb.to(device)
                age_b  = age_b.to(device)
                race_b = race_b.to(device)

                alpha  = _alpha_schedule(global_step, total_steps, adv_lambda)
                lam_r  = _lambda_rank_schedule(global_step, warmup_steps, rank_lambda)
                global_step += 1

                mort_logit, age_logit, race_logit = model(Xb, alpha=alpha)

                losses = model.compute_loss(
                    mort_logit, yb,
                    age_logit=age_logit if alpha > 0 else None,
                    race_logit=race_logit if alpha > 0 else None,
                    age_ids=age_b,
                    race_ids=race_b,
                    lambda_rank=lam_r,
                    lambda_adv=alpha,  # adv weight = GRL alpha (same schedule)
                )

                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss["total"] += losses["total"].item()
                epoch_loss["ce"]    += losses["ce"].item()
                epoch_loss["rank"]  += (losses["rank_age"] + losses["rank_race"]).item() / 2
                if alpha > 0:
                    epoch_loss["adv"] += (losses["adv_age"] + losses["adv_race"]).item() / 2
                n_batches += 1

            # Validation
            model.eval()
            val_preds = []
            with torch.no_grad():
                for Xb, yb, age_b, race_b in val_loader:
                    Xb = Xb.to(device)
                    mort_logit, _, _ = model(Xb, alpha=0.0)
                    val_preds.append(torch.sigmoid(mort_logit).cpu().numpy())
            val_preds = np.concatenate(val_preds)

            try:
                val_auroc = roc_auc_score(y_val, val_preds)
            except Exception:
                val_auroc = 0.0

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg = {k: v/n_batches for k, v in epoch_loss.items()}
                print(f"    Epoch {epoch+1:3d}: AUROC={val_auroc:.4f}  "
                      f"CE={avg['ce']:.4f}  rank={avg['rank']:.4f}  "
                      f"adv={avg['adv']:.4f}  λ_rank={lam_r:.3f}  α={alpha:.3f}",
                      flush=True)

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    Early stop at epoch {epoch+1}", flush=True)
                    break

        # Final predictions on val set
        model.load_state_dict(best_state)
        model.eval()
        val_preds = []
        with torch.no_grad():
            for Xb, yb, age_b, race_b in val_loader:
                Xb = Xb.to(device)
                mort_logit, _, _ = model(Xb, alpha=0.0)
                val_preds.append(torch.sigmoid(mort_logit).cpu().numpy())
        val_preds = np.concatenate(val_preds)
        all_preds[val_idx] = val_preds

        fold_auroc = roc_auc_score(y_val, val_preds)

        # Per-age-group AUROC for this fold's val set
        age_groups = data["age_group"].values[val_idx]
        AGE_ORDER = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
        grp_aurocs = {}
        for ag in AGE_ORDER:
            m = age_groups == ag
            if m.sum() > 20 and y_val[m].sum() > 2:
                try:
                    grp_aurocs[ag] = roc_auc_score(y_val[m], val_preds[m])
                except Exception:
                    pass

        fold_metrics.append({
            "fold": fold, "auroc": fold_auroc,
            "best_val_auroc": best_val_auroc,
            "age_aurocs": grp_aurocs,
        })
        print(f"  Fold {fold+1} AUROC: {fold_auroc:.4f}  "
              f"(best val={best_val_auroc:.4f})", flush=True)
        if grp_aurocs:
            vals = list(grp_aurocs.values())
            print(f"    Age AUROC gap: {max(vals)-min(vals):.4f}  "
                  f"(min={min(vals):.4f} → max={max(vals):.4f})", flush=True)

    overall_auroc = roc_auc_score(y, all_preds)

    # Overall per-age-group AUROC
    age_groups = data["age_group"].values
    AGE_ORDER = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    overall_age_aurocs = {}
    for ag in AGE_ORDER:
        m = age_groups == ag
        if m.sum() > 20 and y[m].sum() > 2:
            try:
                overall_age_aurocs[ag] = roc_auc_score(y[m], all_preds[m])
            except Exception:
                pass

    print(f"\n  GA-FAFT Overall AUROC: {overall_auroc:.4f}", flush=True)
    print(f"  GA-FAFT n_params: {model.n_params:,}", flush=True)
    print(f"  Per-age-group AUROCs:", flush=True)
    for ag, auc in overall_age_aurocs.items():
        print(f"    {ag}: {auc:.4f}", flush=True)
    if overall_age_aurocs:
        vals = list(overall_age_aurocs.values())
        print(f"  Age AUROC gap: {max(vals)-min(vals):.4f}", flush=True)

    return {
        "predictions":       all_preds,
        "overall_auroc":     overall_auroc,
        "fold_metrics":      fold_metrics,
        "age_aurocs":        overall_age_aurocs,
        "n_params":          model.n_params,
        "n_age_groups":      n_age,
        "n_race_groups":     n_race,
    }
