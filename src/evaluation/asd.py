# MIT License — Anonymous Authors, 2026
# Part of ATLAS
"""E3: Adversarial Subgroup Discovery — train XGBoost to predict where
classical scores make large errors, then extract vulnerable subgroups
from tree structure."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from ..data.config import RANDOM_SEED, MIN_SUBGROUP_SIZE

SCORE_NAMES = ["sofa", "qsofa", "apache2", "news2"]


def _build_asd_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for ASD: demographics + clinical context."""
    feat = pd.DataFrame(index=data.index)

    # Demographics (one-hot)
    for col in ["race_cat", "sex", "age_group", "insurance_cat", "diag_type"]:
        if col in data.columns:
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=False)
            feat = pd.concat([feat, dummies], axis=1)

    # Numeric features
    feat["age"] = data["age"]

    # Elixhauser-like comorbidity count from diagnosis codes if available
    if "n_diagnoses" in data.columns:
        feat["n_diagnoses"] = data["n_diagnoses"]

    # First care unit
    if "first_careunit" in data.columns:
        dummies = pd.get_dummies(data["first_careunit"], prefix="unit", drop_first=False)
        feat = pd.concat([feat, dummies], axis=1)

    # Ensure all columns are numeric
    feat = feat.apply(pd.to_numeric, errors="coerce").fillna(0)
    return feat


def _extract_leaf_subgroups(model: xgb.XGBClassifier, feature_names: list,
                            data: pd.DataFrame, X: np.ndarray,
                            errors: np.ndarray, top_k: int = 5) -> list:
    """Extract top-K leaf nodes with highest error concentration and
    characterize them by their decision path."""
    # Get leaf assignments
    leaf_ids = model.apply(X)  # shape: (n_samples, n_trees)

    # Use first tree for interpretability
    subgroups = []
    booster = model.get_booster()
    trees = booster.get_dump(with_stats=True)

    # Alternative: use feature importances + threshold analysis
    # Group by leaf in the first few trees, compute error concentration
    for tree_idx in range(min(3, len(trees))):
        tree_leaves = leaf_ids[:, tree_idx] if leaf_ids.ndim > 1 else leaf_ids
        unique_leaves = np.unique(tree_leaves)

        for leaf in unique_leaves:
            mask = tree_leaves == leaf
            n_in_leaf = mask.sum()
            if n_in_leaf < MIN_SUBGROUP_SIZE:
                continue
            error_rate = errors[mask].mean()
            base_error_rate = errors.mean()
            concentration = error_rate / max(base_error_rate, 1e-8)

            subgroups.append({
                "tree_idx": tree_idx,
                "leaf_id": int(leaf),
                "n": int(n_in_leaf),
                "error_rate": float(error_rate),
                "base_error_rate": float(base_error_rate),
                "concentration_ratio": float(concentration),
                "mask": mask,
            })

    # Sort by concentration ratio
    subgroups.sort(key=lambda x: x["concentration_ratio"], reverse=True)
    return subgroups[:top_k]


def _characterize_subgroup(data: pd.DataFrame, mask: np.ndarray,
                           demo_cols: list) -> dict:
    """Describe a subgroup by its demographic composition."""
    sub = data[mask]
    full = data
    desc = {}
    for col in demo_cols:
        if col not in data.columns:
            continue
        sub_dist = sub[col].value_counts(normalize=True)
        full_dist = full[col].value_counts(normalize=True)
        # Find overrepresented categories
        for cat in sub_dist.index:
            ratio = sub_dist[cat] / max(full_dist.get(cat, 0.01), 0.01)
            if ratio > 1.5:  # >50% overrepresentation
                desc[f"{col}={cat}"] = {
                    "subgroup_pct": float(sub_dist[cat]),
                    "population_pct": float(full_dist.get(cat, 0)),
                    "overrep_ratio": float(ratio),
                }
    desc["n"] = int(mask.sum())
    desc["mortality_rate"] = float(data.loc[mask, "mortality"].mean())
    return desc


def adversarial_subgroup_discovery(data: pd.DataFrame,
                                   error_threshold: float = 0.3,
                                   ) -> dict:
    """E3: For each score, train XGBoost on large errors, extract subgroups."""
    demo_cols = ["race_cat", "sex", "age_group", "insurance_cat", "diag_type"]
    X_df = _build_asd_features(data)
    feature_names = list(X_df.columns)
    X = X_df.values

    all_results = {}

    for score_name in SCORE_NAMES:
        print(f"\nASD for {score_name}...")

        # Score-to-probability mapping
        score_vals = data[score_name]
        mapping = data["mortality"].groupby(score_vals).mean()
        pred_prob = score_vals.map(mapping).fillna(data["mortality"].mean()).values
        actual = data["mortality"].values

        # Error = |predicted - actual|
        errors = np.abs(pred_prob - actual)
        # Binary target: large error
        large_error = (errors > error_threshold).astype(int)

        if large_error.sum() < 20 or (1 - large_error).sum() < 20:
            print(f"  Skipping {score_name}: insufficient error variation "
                  f"({large_error.sum()} large errors)")
            continue

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric="logloss",
        )

        # Cross-validated predictions for fair evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_probs = cross_val_predict(model, X, large_error, cv=cv,
                                      method="predict_proba")[:, 1]

        # Fit on full data for tree extraction
        model.fit(X, large_error)

        # Model performance
        try:
            auroc = roc_auc_score(large_error, cv_probs)
        except ValueError:
            auroc = np.nan

        # Feature importance
        importances = dict(zip(feature_names, model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1],
                              reverse=True)[:10]

        # Extract vulnerable subgroups from tree leaves
        subgroups = _extract_leaf_subgroups(model, feature_names, data,
                                            X, errors)

        # Characterize each subgroup
        for sg in subgroups:
            sg["demographics"] = _characterize_subgroup(data, sg["mask"],
                                                         demo_cols)
            del sg["mask"]  # Don't store the mask in results

        all_results[score_name] = {
            "error_prediction_auroc": float(auroc) if not np.isnan(auroc) else None,
            "top_features": top_features,
            "vulnerable_subgroups": subgroups,
            "n_large_errors": int(large_error.sum()),
            "error_rate": float(large_error.mean()),
        }

        print(f"  Error prediction AUROC: {auroc:.3f}")
        print(f"  Top features: {[f[0] for f in top_features[:5]]}")
        print(f"  Found {len(subgroups)} vulnerable subgroups")

    return all_results
