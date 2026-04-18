# MIT License — Anonymous Authors, 2026
# Part of ATLAS
"""Pre-specified and intersectional subgroup auditing of clinical scores."""

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from sklearn.calibration import calibration_curve as sklearn_cal_curve
from scipy import stats
from itertools import product
from ..data.config import (BOOTSTRAP_ITERATIONS, MIN_SUBGROUP_SIZE, FDR_Q,
                           RANDOM_SEED)

SCORE_NAMES = ["sofa", "qsofa", "apache2", "news2"]
DEMO_AXES = ["race_cat", "sex", "age_group", "diag_type"]

# SOFA component physiologic columns (GOSSIS naming)
SOFA_COMPONENTS = {
    "respiratory": "pf_ratio_min",
    "coagulation": "platelets_min",
    "liver": "bilirubin_max",
    "cardiovascular": "map_min",
    "cns": "gcs_total",
    "renal": "creatinine_max",
}

# Standard clinical decision thresholds
CLINICAL_THRESHOLDS = {
    "sofa":     [2, 6, 11],
    "qsofa":    [1, 2],
    "apache2":  [15, 20, 25],
    "news2":    [5, 9, 13],
}


def _safe_auroc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def _safe_auprc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_score)


def _calibration_metrics(y_true, y_score, n_bins=10):
    """Compute calibration slope and intercept via logistic regression of
    observed vs predicted in bins."""
    try:
        frac_pos, mean_pred = sklearn_cal_curve(y_true, y_score, n_bins=n_bins,
                                                 strategy="quantile")
        if len(frac_pos) < 3:
            return np.nan, np.nan
        slope, intercept, _, _, _ = stats.linregress(mean_pred, frac_pos)
        return slope, intercept
    except Exception:
        return np.nan, np.nan


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray,
                                n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_score >= bins[i]) & (y_score < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_score[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return ece


def _score_to_prob(score: pd.Series, y_true: pd.Series) -> pd.Series:
    """Convert integer score to pseudo-probability using isotonic-like mapping.
    Bin by score value, compute empirical mortality rate per bin."""
    mapping = y_true.groupby(score).mean()
    return score.map(mapping).fillna(y_true.mean())


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute AUROC, AUPRC, Brier score, calibration slope/intercept."""
    auroc = _safe_auroc(y_true, y_score)
    auprc = _safe_auprc(y_true, y_score)
    brier = brier_score_loss(y_true, y_score) if not np.any(np.isnan(y_score)) else np.nan
    cal_slope, cal_int = _calibration_metrics(y_true, y_score)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "cal_slope": cal_slope,
        "cal_intercept": cal_int,
        "n": len(y_true),
        "prevalence": y_true.mean(),
    }


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray,
                 metric_fn, n_boot=BOOTSTRAP_ITERATIONS,
                 seed=RANDOM_SEED) -> tuple:
    """Compute 95% bootstrap CI for a metric."""
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            v = metric_fn(y_true[idx], y_score[idx])
            if not np.isnan(v):
                vals.append(v)
        except Exception:
            continue
    if len(vals) < 10:
        return np.nan, np.nan
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


# ═══════════════════════════════════════════════���══════════════════════════
# E1: Pre-specified subgroup audit
# ═══════════════════════════════════════════════════════════════���══════════

def prespecified_audit(data: pd.DataFrame, n_boot: int = BOOTSTRAP_ITERATIONS,
                       axes: list = None) -> pd.DataFrame:
    """E1: For each score x demographic axis, compute performance metrics
    per subgroup with bootstrap CIs."""
    if axes is None:
        axes = [a for a in DEMO_AXES if a in data.columns and
                data[a].nunique() > 1]
    results = []

    for score_name in SCORE_NAMES:
        # Convert score to probability
        prob = _score_to_prob(data[score_name], data["mortality"])
        y_true = data["mortality"].values

        for axis in axes:
            groups = data[axis].dropna().unique()
            for grp in groups:
                mask = data[axis] == grp
                if mask.sum() < 20:
                    continue

                yt = y_true[mask]
                yp = prob.values[mask]

                metrics = compute_metrics(yt, yp)

                # Bootstrap CIs for AUROC
                lo, hi = bootstrap_ci(yt, yp, _safe_auroc, n_boot=n_boot)

                row = {
                    "score": score_name,
                    "axis": axis,
                    "group": grp,
                    **metrics,
                    "auroc_ci_lo": lo,
                    "auroc_ci_hi": hi,
                }
                results.append(row)

    results_df = pd.DataFrame(results)

    # Compute gaps per score x axis
    gaps = []
    for score_name in SCORE_NAMES:
        for axis in axes:
            sub = results_df[(results_df["score"] == score_name) &
                             (results_df["axis"] == axis)]
            if len(sub) < 2:
                continue
            aurocs = sub["auroc"].dropna()
            if len(aurocs) < 2:
                continue
            gap = aurocs.max() - aurocs.min()
            gaps.append({
                "score": score_name,
                "axis": axis,
                "auroc_gap": gap,
                "worst_group": sub.loc[aurocs.idxmin(), "group"],
                "best_group": sub.loc[aurocs.idxmax(), "group"],
                "worst_auroc": aurocs.min(),
                "best_auroc": aurocs.max(),
            })

    gaps_df = pd.DataFrame(gaps)
    return results_df, gaps_df


# ═════════════��═══════════════════════════════════════════════════��════════
# E2: Intersectional analysis
# ═══════��══════════════════════════════════════════════════════════���═══════

def intersectional_audit(data: pd.DataFrame,
                         min_n: int = MIN_SUBGROUP_SIZE,
                         n_boot: int = BOOTSTRAP_ITERATIONS,
                         axes: list = None) -> pd.DataFrame:
    """E2: Cross demographic axes, identify worst subgroups per score."""
    if axes is None:
        axes = [a for a in DEMO_AXES if a in data.columns and
                data[a].nunique() > 1]
    results = []

    # All pairwise combinations of demographic axes
    axis_pairs = [(a, b) for i, a in enumerate(axes)
                  for b in axes[i+1:]]

    for score_name in SCORE_NAMES:
        prob = _score_to_prob(data[score_name], data["mortality"])
        y_true = data["mortality"].values

        for ax1, ax2 in axis_pairs:
            groups1 = data[ax1].dropna().unique()
            groups2 = data[ax2].dropna().unique()

            for g1, g2 in product(groups1, groups2):
                mask = (data[ax1] == g1) & (data[ax2] == g2)
                n_sub = mask.sum()
                if n_sub < min_n:
                    continue

                yt = y_true[mask]
                yp = prob.values[mask]

                metrics = compute_metrics(yt, yp)
                lo, hi = bootstrap_ci(yt, yp, _safe_auroc, n_boot=min(n_boot, 500))

                results.append({
                    "score": score_name,
                    "axis1": ax1, "group1": g1,
                    "axis2": ax2, "group2": g2,
                    "subgroup": f"{g1} x {g2}",
                    **metrics,
                    "auroc_ci_lo": lo,
                    "auroc_ci_hi": hi,
                })

    results_df = pd.DataFrame(results)

    # Find top-10 worst per score
    worst = {}
    for score_name in SCORE_NAMES:
        sub = results_df[results_df["score"] == score_name]
        sub_sorted = sub.dropna(subset=["auroc"]).sort_values("auroc")
        worst[score_name] = sub_sorted.head(10)

    return results_df, worst


# ═════════════════════════════════════════��════════════════════════════════
# BH-FDR correction for multiple comparisons
# ══════════��════════════════��══════════════════════════════════════════════

def apply_fdr_correction(results_df: pd.DataFrame, pval_col: str = "pval",
                         q: float = FDR_Q) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction."""
    pvals = results_df[pval_col].dropna()
    n = len(pvals)
    if n == 0:
        results_df["significant_fdr"] = False
        return results_df

    sorted_idx = pvals.sort_values().index
    ranks = np.arange(1, n + 1)
    thresholds = q * ranks / n

    significant = np.zeros(n, dtype=bool)
    for i in range(n - 1, -1, -1):
        if pvals.loc[sorted_idx[i]] <= thresholds[i]:
            significant[:i + 1] = True
            break

    results_df["significant_fdr"] = False
    for i, idx in enumerate(sorted_idx):
        results_df.loc[idx, "significant_fdr"] = significant[i]

    return results_df


# ═══════════════════════════════════════════════════════════════════════════
# E6: Score-conditional mortality analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_score_conditional_mortality(data: pd.DataFrame,
                                        demo_axis: str = "age_group",
                                        min_bin_n: int = 50) -> pd.DataFrame:
    """E6: For each score value (or bin), compute empirical mortality rate
    stratified by demographic group. Reveals that the same score maps to
    very different mortality rates across groups.

    Returns tidy DataFrame: score, score_value, group, mortality_rate, n.
    """
    results = []

    for score_name in SCORE_NAMES:
        score_vals = data[score_name].dropna()
        unique_vals = sorted(score_vals.unique())

        # Bin continuous scores into at most 15 quantile bins
        if len(unique_vals) > 15:
            quantiles = np.percentile(score_vals, np.linspace(0, 100, 16))
            quantiles = np.unique(quantiles)
            labels = [f"{quantiles[i]:.1f}-{quantiles[i+1]:.1f}"
                      for i in range(len(quantiles) - 1)]
            bin_col = pd.cut(data[score_name], bins=quantiles,
                             labels=labels, include_lowest=True)
        else:
            bin_col = data[score_name].astype(str)
            labels = [str(v) for v in sorted(unique_vals)]

        groups = sorted(data[demo_axis].dropna().unique())
        for bin_label in labels:
            in_bin = bin_col == bin_label
            if in_bin.sum() < min_bin_n:
                continue

            for grp in groups:
                mask = in_bin & (data[demo_axis] == grp)
                n = mask.sum()
                if n < 10:
                    continue
                mort_rate = data.loc[mask, "mortality"].mean()
                results.append({
                    "score": score_name,
                    "score_bin": bin_label,
                    "demo_axis": demo_axis,
                    "group": grp,
                    "mortality_rate": mort_rate,
                    "n": n,
                })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# E7: Clinical threshold analysis
# ═══════════════════════════════════════════════════════════════════════════

def clinical_threshold_audit(data: pd.DataFrame,
                              axes: list = None) -> pd.DataFrame:
    """E7: At each standard clinical decision threshold, compute sensitivity,
    specificity, PPV, NPV per demographic subgroup.

    Returns tidy DataFrame with one row per score x threshold x axis x group.
    """
    if axes is None:
        axes = [a for a in DEMO_AXES if a in data.columns and
                data[a].nunique() > 1]
    results = []
    y_true = data["mortality"].values

    for score_name, thresholds in CLINICAL_THRESHOLDS.items():
        score_vals = data[score_name].values

        for threshold in thresholds:
            y_pred = (score_vals >= threshold).astype(int)

            # Overall
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            ppv = tp / max(tp + fp, 1)
            npv = tn / max(tn + fn, 1)
            results.append({
                "score": score_name, "threshold": threshold,
                "axis": "overall", "group": "overall",
                "sensitivity": sens, "specificity": spec,
                "ppv": ppv, "npv": npv,
                "n": len(y_true), "prevalence": y_true.mean(),
                "positive_rate": y_pred.mean(),
            })

            for axis in axes:
                groups = data[axis].dropna().unique()
                for grp in groups:
                    mask = (data[axis] == grp).values
                    if mask.sum() < 20:
                        continue
                    yt = y_true[mask]
                    yp = y_pred[mask]
                    tp_ = ((yp == 1) & (yt == 1)).sum()
                    fp_ = ((yp == 1) & (yt == 0)).sum()
                    fn_ = ((yp == 0) & (yt == 1)).sum()
                    tn_ = ((yp == 0) & (yt == 0)).sum()
                    results.append({
                        "score": score_name, "threshold": threshold,
                        "axis": axis, "group": grp,
                        "sensitivity": tp_ / max(tp_ + fn_, 1),
                        "specificity": tn_ / max(tn_ + fp_, 1),
                        "ppv": tp_ / max(tp_ + fp_, 1),
                        "npv": tn_ / max(tn_ + fn_, 1),
                        "n": mask.sum(),
                        "prevalence": yt.mean(),
                        "positive_rate": yp.mean(),
                    })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# SOFA component attribution
# ═══════════════════════════════════════════════════════════════════════════

def decompose_sofa_components(data: pd.DataFrame,
                               demo_axis: str = "age_group") -> pd.DataFrame:
    """Audit each SOFA component individually for fairness gaps.

    For each physiologic variable, computes AUROC stratified by demo_axis.
    Reveals which SOFA components drive age-related disparities.
    """
    results = []

    available_components = {
        name: col for name, col in SOFA_COMPONENTS.items()
        if col in data.columns
    }

    if not available_components:
        warnings.warn("No SOFA component columns found in data.")
        return pd.DataFrame()

    groups = sorted(data[demo_axis].dropna().unique())
    y_true = data["mortality"].values

    for comp_name, col in available_components.items():
        comp_vals = data[col].copy()
        # Handle direction: some components, higher = worse (creatinine, bilirubin);
        # some lower = worse (platelets, PF ratio, MAP, GCS).
        # Normalize by rank to put all on 0-1 scale (higher = worse mortality).
        lower_worse = {"platelets_min", "pf_ratio_min", "map_min", "gcs_total"}
        if col in lower_worse:
            comp_vals = -comp_vals  # invert so higher = worse

        # Fill missing with median
        comp_vals = comp_vals.fillna(comp_vals.median())
        y_score = comp_vals.values

        # Overall AUROC
        if len(np.unique(y_true)) >= 2:
            overall_auroc = _safe_auroc(y_true, y_score)
        else:
            overall_auroc = np.nan

        results.append({
            "component": comp_name,
            "column": col,
            "demo_axis": demo_axis,
            "group": "overall",
            "auroc": overall_auroc,
            "n": len(y_true),
        })

        aurocs_by_group = []
        for grp in groups:
            mask = (data[demo_axis] == grp).values
            if mask.sum() < 20 or len(np.unique(y_true[mask])) < 2:
                continue
            auroc = _safe_auroc(y_true[mask], y_score[mask])
            aurocs_by_group.append(auroc)
            results.append({
                "component": comp_name,
                "column": col,
                "demo_axis": demo_axis,
                "group": grp,
                "auroc": auroc,
                "n": mask.sum(),
            })

        # Append gap row
        if len(aurocs_by_group) >= 2:
            results.append({
                "component": comp_name,
                "column": col,
                "demo_axis": demo_axis,
                "group": "_gap",
                "auroc": max(aurocs_by_group) - min(aurocs_by_group),
                "n": np.nan,
            })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# Hospital-stratified race analysis
# ═══════════════════════════════════════════════════════════════════════════

def hospital_stratified_audit(data: pd.DataFrame,
                               hospital_col: str = "hospitalid",
                               min_hospital_n: int = 100) -> pd.DataFrame:
    """Compute race AUROC gaps within hospitals (controls for hospital mix).

    Addresses Simpson's paradox: aggregate race gaps can differ substantially
    from within-hospital gaps due to differential hospital quality.
    """
    results = []
    if hospital_col not in data.columns:
        warnings.warn(f"Column '{hospital_col}' not found.")
        return pd.DataFrame()
    if "race_cat" not in data.columns:
        warnings.warn("Column 'race_cat' not found.")
        return pd.DataFrame()

    hospitals = data[hospital_col].unique()

    for score_name in SCORE_NAMES:
        prob = _score_to_prob(data[score_name], data["mortality"])
        y_true = data["mortality"].values

        within_hospital_gaps = []

        for hosp in hospitals:
            hosp_mask = data[hospital_col] == hosp
            if hosp_mask.sum() < min_hospital_n:
                continue

            hosp_data = data[hosp_mask]
            hosp_prob = prob[hosp_mask].values
            hosp_true = y_true[hosp_mask]
            races = hosp_data["race_cat"].dropna().unique()

            aurocs = {}
            for race in races:
                race_mask = (hosp_data["race_cat"] == race).values
                if race_mask.sum() < 10 or len(np.unique(hosp_true[race_mask])) < 2:
                    continue
                auroc = _safe_auroc(hosp_true[race_mask], hosp_prob[race_mask])
                if not np.isnan(auroc):
                    aurocs[race] = auroc

            if len(aurocs) >= 2:
                gap = max(aurocs.values()) - min(aurocs.values())
                within_hospital_gaps.append({
                    "score": score_name,
                    "hospitalid": hosp,
                    "n": hosp_mask.sum(),
                    "race_auroc_gap": gap,
                    "worst_race": min(aurocs, key=aurocs.get),
                    "best_race": max(aurocs, key=aurocs.get),
                    "n_races_with_data": len(aurocs),
                })

        results.extend(within_hospital_gaps)

    df = pd.DataFrame(results)

    # Summary: weighted mean within-hospital gap vs aggregate gap
    summary = []
    for score_name in SCORE_NAMES:
        sub = df[df["score"] == score_name]
        if sub.empty:
            continue
        # Weighted mean gap (by hospital size)
        weights = sub["n"]
        wmean_gap = np.average(sub["race_auroc_gap"], weights=weights)

        # Aggregate gap (treat full dataset)
        prob = _score_to_prob(data[score_name], data["mortality"])
        races = data["race_cat"].dropna().unique()
        agg_aurocs = []
        for race in races:
            mask = data["race_cat"] == race
            if mask.sum() < 20 or len(np.unique(data.loc[mask, "mortality"])) < 2:
                continue
            try:
                a = _safe_auroc(data.loc[mask, "mortality"].values, prob[mask].values)
                if not np.isnan(a):
                    agg_aurocs.append(a)
            except Exception:
                continue
        agg_gap = max(agg_aurocs) - min(agg_aurocs) if len(agg_aurocs) >= 2 else np.nan

        summary.append({
            "score": score_name,
            "n_hospitals": len(sub),
            "within_hospital_gap_wmean": wmean_gap,
            "aggregate_gap": agg_gap,
            "simpsons_paradox_ratio": wmean_gap / max(agg_gap, 1e-6) if not np.isnan(agg_gap) else np.nan,
        })

    return df, pd.DataFrame(summary)
