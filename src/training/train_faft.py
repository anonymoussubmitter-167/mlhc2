# MIT License — Anonymous Authors, 2026
"""Training loop for the Fairness-Aware Feature Transformer (FAFT)."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

from src.models.faft import FAFT


def _prepare_features(data):
    """Same exclusion list as GRU — ensures fair comparison."""
    import numpy as np
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


def _alpha_schedule(step: int, total_steps: int, max_alpha: float = 0.5) -> float:
    """Ganin et al. (2016) GRL schedule: gradually increase alpha."""
    p = step / max(total_steps, 1)
    return max_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def train_faft_model(data,
                     device: str = "cpu",
                     epochs: int = 50,
                     d_model: int = 64,
                     n_heads: int = 4,
                     n_layers: int = 2,
                     d_ff: int = 256,
                     dropout: float = 0.1,
                     batch_size: int = 256,
                     lr: float = 3e-4,
                     patience: int = 10,
                     adv_lambda: float = 0.3,
                     cv_seed: int = 42):
    """Train FAFT with 5-fold CV.

    Args:
        adv_lambda: max weight for adversarial loss (alpha schedule scales up to this)

    Returns dict with keys:
        predictions, overall_auroc, fold_metrics, n_age_groups, n_race_groups
    """
    X_raw, feat_cols = _prepare_features(data)
    y = data["mortality"].values.astype(np.float32)

    # Encode demographic labels for adversarial heads
    age_enc  = LabelEncoder().fit(data["age_group"].fillna("Unknown"))
    race_enc = LabelEncoder().fit(data["race_cat"].fillna("Unknown"))
    age_labels  = age_enc.transform(data["age_group"].fillna("Unknown")).astype(np.int64)
    race_labels = race_enc.transform(data["race_cat"].fillna("Unknown")).astype(np.int64)
    n_age   = len(age_enc.classes_)
    n_race  = len(race_enc.classes_)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    all_preds = np.zeros(len(data), dtype=np.float32)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_raw, y)):
        print(f"  Fold {fold+1}/5...", flush=True)

        # Scale features
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_raw[train_idx])
        X_val   = scaler.transform(X_raw[val_idx])

        y_train = y[train_idx]
        y_val   = y[val_idx]
        age_train  = age_labels[train_idx]
        race_train = race_labels[train_idx]

        # Datasets
        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(age_train),
            torch.from_numpy(race_train),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        # Model
        model = FAFT(
            n_features=X_train.shape[1],
            n_age_groups=n_age,
            n_race_groups=n_race,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        ).to(device)

        # Loss functions
        pos_weight = torch.tensor(
            [(1 - y_train.mean()) / max(y_train.mean(), 1e-6)]
        ).to(device)
        mort_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        age_criterion  = nn.CrossEntropyLoss()
        race_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        total_steps = epochs * len(train_loader)

        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0
        global_step   = 0

        for epoch in range(epochs):
            model.train()
            for Xb, yb, age_b, race_b in train_loader:
                Xb    = Xb.to(device)
                yb    = yb.to(device)
                age_b = age_b.to(device)
                race_b = race_b.to(device)

                alpha = _alpha_schedule(global_step, total_steps, adv_lambda)
                global_step += 1

                mort_logit, age_logit, race_logit = model(Xb, alpha=alpha)

                loss_mort = mort_criterion(mort_logit, yb)
                if alpha > 0 and age_logit is not None:
                    loss_age  = age_criterion(age_logit, age_b)
                    loss_race = race_criterion(race_logit, race_b)
                    loss = loss_mort + alpha * (loss_age + loss_race)
                else:
                    loss = loss_mort

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validation (no adversarial head)
            model.eval()
            val_loss = 0.0
            val_preds = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    mort_logit, _, _ = model(Xb, alpha=0.0)
                    val_loss += mort_criterion(mort_logit, yb).item() * len(yb)
                    val_preds.append(torch.sigmoid(mort_logit).cpu().numpy())

            val_loss /= len(y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Final val predictions with best weights
        model.load_state_dict(best_state)
        model.eval()
        val_preds = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                mort_logit, _, _ = model(Xb, alpha=0.0)
                val_preds.append(torch.sigmoid(mort_logit).cpu().numpy())
        val_preds = np.concatenate(val_preds)
        all_preds[val_idx] = val_preds

        try:
            fold_auroc = roc_auc_score(y_val, val_preds)
        except Exception:
            fold_auroc = float("nan")

        fold_metrics.append({"fold": fold, "auroc": fold_auroc,
                              "val_loss": best_val_loss})
        print(f"    AUROC: {fold_auroc:.4f}, Val loss: {best_val_loss:.4f}",
              flush=True)

    try:
        overall_auroc = roc_auc_score(y, all_preds)
    except Exception:
        overall_auroc = float("nan")

    print(f"  Overall CV AUROC: {overall_auroc:.4f}", flush=True)
    print(f"  Model params: {model.n_params:,}", flush=True)

    return {
        "predictions":   all_preds,
        "overall_auroc": overall_auroc,
        "fold_metrics":  fold_metrics,
        "n_age_groups":  n_age,
        "n_race_groups": n_race,
    }
