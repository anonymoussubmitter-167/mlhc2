# MIT License — Anonymous Authors, 2026
# Part of ATLAS
"""GRU training loop for mortality prediction."""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from ..models.gru import MortalityGRU
from ..data.config import RANDOM_SEED, EXPERIMENTS_DIR


def _prepare_features(data: pd.DataFrame) -> tuple:
    """Extract numeric features suitable for GRU input."""
    exclude = ["subject_id", "hadm_id", "stay_id", "intime", "outtime",
               "mortality", "race_cat", "sex", "age_group", "insurance_cat",
               "diag_type", "first_careunit", "sofa", "qsofa", "apache2",
               "news2", "race_raw", "los", "hospitalid"]
    feat_cols = [c for c in data.columns if c not in exclude
                 and data[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    X = data[feat_cols].copy()

    # Fill NaN with column median
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    X = X.fillna(0)

    return X.values.astype(np.float32), feat_cols


def train_gru_model(data: pd.DataFrame,
                    device: str = "cuda:1",
                    hidden_dim: int = 128,
                    n_layers: int = 2,
                    lr: float = 1e-3,
                    epochs: int = 50,
                    batch_size: int = 256,
                    patience: int = 10,
                    cv_seed: int = None,
                    ) -> dict:
    """Train GRU with 5-fold CV, return predictions and metrics."""
    X_raw, feat_cols = _prepare_features(data)
    y = data["mortality"].values.astype(np.float32)

    if not torch.cuda.is_available():
        device = "cpu"

    _cv_seed = cv_seed if cv_seed is not None else RANDOM_SEED
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=_cv_seed)
    all_preds = np.zeros(len(y))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_raw, y)):
        print(f"  Fold {fold+1}/5...")

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_raw[train_idx])
        X_val = scaler.transform(X_raw[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        # Tensors
        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Model
        model = MortalityGRU(
            input_dim=X_train.shape[1],
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        ).to(device)

        # Class-weighted loss
        pos_weight = torch.tensor([(1 - y_train.mean()) / max(y_train.mean(), 1e-6)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=patience // 2, factor=0.5)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(yb)

            # Validate
            model.eval()
            val_loss = 0
            val_preds = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    logits = model(Xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * len(yb)
                    val_preds.append(torch.sigmoid(logits).cpu().numpy())

            val_loss /= len(y_val)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Best model predictions on validation set
        model.load_state_dict(best_state)
        model.eval()
        val_preds = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                logits = model(Xb)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
        val_preds = np.concatenate(val_preds)
        all_preds[val_idx] = val_preds

        try:
            fold_auroc = roc_auc_score(y_val, val_preds)
        except ValueError:
            fold_auroc = np.nan
        fold_metrics.append({"fold": fold, "auroc": fold_auroc,
                             "val_loss": best_val_loss})
        print(f"    AUROC: {fold_auroc:.4f}, Val loss: {best_val_loss:.4f}")

    # Overall metrics
    try:
        overall_auroc = roc_auc_score(y, all_preds)
    except ValueError:
        overall_auroc = np.nan

    print(f"  Overall CV AUROC: {overall_auroc:.4f}")
    print(f"  Model params: {model.n_params:,}")

    return {
        "predictions": all_preds,
        "overall_auroc": overall_auroc,
        "fold_metrics": fold_metrics,
        "n_params": model.n_params,
        "feature_cols": feat_cols,
    }
