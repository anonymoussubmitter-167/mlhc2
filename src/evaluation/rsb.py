# MIT License — Anonymous Authors, 2026
# Part of ATLAS
"""E4-E5: Reference Standard Bias quantification.

Measures how much ML fairness evaluations are distorted when benchmarked
against inequitable classical scores vs. ground truth outcomes.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from ..data.config import BOOTSTRAP_ITERATIONS, RANDOM_SEED
from .audit import _score_to_prob, bootstrap_ci, _safe_auroc

SCORE_NAMES = ["sofa", "qsofa", "apache2", "news2"]
DEMO_AXES = ["race_cat", "sex", "age_group", "insurance_cat"]


def _equalized_odds_diff(y_true: np.ndarray, y_pred: np.ndarray,
                          groups: np.ndarray) -> float:
    """Compute equalized odds difference: max |TPR_a - TPR_b| + max |FPR_a - FPR_b|."""
    unique_groups = np.unique(groups[~pd.isna(groups)])
    if len(unique_groups) < 2:
        return np.nan

    threshold = np.median(y_pred)
    y_binary = (y_pred >= threshold).astype(int)

    tprs, fprs = [], []
    for g in unique_groups:
        mask = groups == g
        yt, yp = y_true[mask], y_binary[mask]
        if yt.sum() == 0 or (1 - yt).sum() == 0:
            continue
        tpr = yp[yt == 1].mean() if yt.sum() > 0 else np.nan
        fpr = yp[yt == 0].mean() if (1 - yt).sum() > 0 else np.nan
        if not np.isnan(tpr):
            tprs.append(tpr)
        if not np.isnan(fpr):
            fprs.append(fpr)

    if len(tprs) < 2 or len(fprs) < 2:
        return np.nan

    eod = (max(tprs) - min(tprs)) + (max(fprs) - min(fprs))
    return eod


def _predictive_parity_gap(y_true: np.ndarray, y_pred: np.ndarray,
                            groups: np.ndarray) -> float:
    """Compute max |PPV_a - PPV_b| across groups."""
    unique_groups = np.unique(groups[~pd.isna(groups)])
    if len(unique_groups) < 2:
        return np.nan

    threshold = np.median(y_pred)
    y_binary = (y_pred >= threshold).astype(int)

    ppvs = []
    for g in unique_groups:
        mask = groups == g
        pred_pos = y_binary[mask] == 1
        if pred_pos.sum() == 0:
            continue
        ppv = y_true[mask][pred_pos].mean()
        ppvs.append(ppv)

    if len(ppvs) < 2:
        return np.nan
    return max(ppvs) - min(ppvs)


def _calibration_gap(y_true: np.ndarray, y_pred: np.ndarray,
                      groups: np.ndarray, n_bins: int = 5) -> float:
    """Compute max calibration gap across groups, averaged over score bins."""
    unique_groups = np.unique(groups[~pd.isna(groups)])
    if len(unique_groups) < 2:
        return np.nan

    bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 0.001
    bin_ids = np.digitize(y_pred, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    max_gaps = []
    for b in range(n_bins):
        in_bin = bin_ids == b
        if in_bin.sum() < 10:
            continue
        rates = []
        for g in unique_groups:
            mask = in_bin & (groups == g)
            if mask.sum() < 5:
                continue
            rates.append(y_true[mask].mean())
        if len(rates) >= 2:
            max_gaps.append(max(rates) - min(rates))

    if not max_gaps:
        return np.nan
    return np.mean(max_gaps)


def _auroc_gap(y_true: np.ndarray, y_pred: np.ndarray,
               groups: np.ndarray) -> float:
    """Max - min AUROC across groups."""
    unique_groups = np.unique(groups[~pd.isna(groups)])
    aurocs = []
    for g in unique_groups:
        mask = groups == g
        if mask.sum() < 20 or len(np.unique(y_true[mask])) < 2:
            continue
        try:
            aurocs.append(roc_auc_score(y_true[mask], y_pred[mask]))
        except ValueError:
            continue
    if len(aurocs) < 2:
        return np.nan
    return max(aurocs) - min(aurocs)


def _ece_gap(y_true: np.ndarray, y_pred: np.ndarray,
             groups: np.ndarray, n_bins: int = 10) -> float:
    """Max - min Expected Calibration Error across demographic groups."""
    unique_groups = np.unique(groups[~pd.isna(groups)])
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    eces = []
    for g in unique_groups:
        mask = groups == g
        if mask.sum() < 20:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        n = mask.sum()
        ece = 0.0
        for i in range(n_bins):
            bin_mask = (yp >= bins[i]) & (yp < bins[i + 1])
            if bin_mask.sum() == 0:
                continue
            acc = yt[bin_mask].mean()
            conf = yp[bin_mask].mean()
            ece += bin_mask.sum() / n * abs(acc - conf)
        eces.append(ece)
    if len(eces) < 2:
        return np.nan
    return max(eces) - min(eces)


FAIRNESS_METRICS = {
    "eod": _equalized_odds_diff,
    "ppg": _predictive_parity_gap,
    "cal_gap": _calibration_gap,
    "auroc_gap": _auroc_gap,
    "ece_gap": _ece_gap,
}


def compute_rsb(data: pd.DataFrame, ml_preds: np.ndarray,
                n_boot: int = BOOTSTRAP_ITERATIONS,
                axes: list = None) -> pd.DataFrame:
    """E4: Compute RSB_gap for each score, demographic axis, fairness metric.

    RSB_gap = |Fairness(ML vs ground_truth) - Fairness(ML vs classical_score)|
    """
    if axes is None:
        axes = [a for a in DEMO_AXES if a in data.columns and
                data[a].nunique() > 1]
    y_true = data["mortality"].values
    results = []

    for score_name in SCORE_NAMES:
        # Classical score as pseudo-ground-truth
        score_prob = _score_to_prob(data[score_name], data["mortality"]).values

        for axis in axes:
            groups = data[axis].values

            for metric_name, metric_fn in FAIRNESS_METRICS.items():
                # Fairness of ML vs true outcome
                fair_vs_gt = metric_fn(y_true, ml_preds, groups)

                # Fairness of ML vs classical score (treating score as reference)
                # Binarize score at median for "reference outcome"
                score_binary = (score_prob >= np.median(score_prob)).astype(int)
                fair_vs_score = metric_fn(score_binary, ml_preds, groups)

                rsb_gap = abs(fair_vs_gt - fair_vs_score) if not (
                    np.isnan(fair_vs_gt) or np.isnan(fair_vs_score)) else np.nan

                # Bootstrap CI on RSB gap
                rng = np.random.RandomState(RANDOM_SEED)
                boot_gaps = []
                for _ in range(min(n_boot, 500)):
                    idx = rng.choice(len(y_true), size=len(y_true), replace=True)
                    try:
                        fg = metric_fn(y_true[idx], ml_preds[idx], groups[idx])
                        fs = metric_fn(score_binary[idx], ml_preds[idx], groups[idx])
                        if not (np.isnan(fg) or np.isnan(fs)):
                            boot_gaps.append(abs(fg - fs))
                    except Exception:
                        continue

                ci_lo = np.percentile(boot_gaps, 2.5) if len(boot_gaps) > 10 else np.nan
                ci_hi = np.percentile(boot_gaps, 97.5) if len(boot_gaps) > 10 else np.nan

                results.append({
                    "score": score_name,
                    "axis": axis,
                    "metric": metric_name,
                    "fair_vs_gt": fair_vs_gt,
                    "fair_vs_score": fair_vs_score,
                    "rsb_gap": rsb_gap,
                    "rsb_gap_ci_lo": ci_lo,
                    "rsb_gap_ci_hi": ci_hi,
                })

    return pd.DataFrame(results)


def compute_ml_improvement(data: pd.DataFrame, ml_preds: np.ndarray,
                           axes: list = None) -> pd.DataFrame:
    """E5: Measure how much ML closes the fairness gap vs classical scores."""
    if axes is None:
        axes = [a for a in DEMO_AXES if a in data.columns and
                data[a].nunique() > 1]
    y_true = data["mortality"].values
    results = []

    for score_name in SCORE_NAMES:
        score_prob = _score_to_prob(data[score_name], data["mortality"]).values

        for axis in axes:
            groups = data[axis].values

            for metric_name, metric_fn in FAIRNESS_METRICS.items():
                # Fairness gap of classical score
                score_gap = metric_fn(y_true, score_prob, groups)
                # Fairness gap of ML
                ml_gap = metric_fn(y_true, ml_preds, groups)

                improvement = score_gap - ml_gap if not (
                    np.isnan(score_gap) or np.isnan(ml_gap)) else np.nan
                pct_improvement = (improvement / max(score_gap, 1e-6) * 100
                                   if not np.isnan(improvement) else np.nan)

                results.append({
                    "score": score_name,
                    "axis": axis,
                    "metric": metric_name,
                    "score_fairness_gap": score_gap,
                    "ml_fairness_gap": ml_gap,
                    "improvement": improvement,
                    "pct_improvement": pct_improvement,
                })

    return pd.DataFrame(results)
