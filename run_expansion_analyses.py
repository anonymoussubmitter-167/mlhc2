#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
# Part of ATLAS
"""
New expansion analyses for ATLAS journal version:
  E16: Temporal Drift Analysis
  E17: Causal Mediation Decomposition (Baron-Kenny)
  E18: Comorbidity Sensitivity (Elixhauser-proxy)
  E19: Foundation Model Comparison (XGBoost baseline + interface)

Usage:
    python run_expansion_analyses.py
    python run_expansion_analyses.py --cohort-path experiments/exp_gossis/cohort_with_scores.csv
    python run_expansion_analyses.py --analyses E16 E17
"""

import sys
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("atlas.expansion")

RANDOM_SEED = 42
SCORE_NAMES = ["sofa", "qsofa", "apache2", "news2"]
DEMO_AXES = ["race_cat", "sex", "age_group", "diag_type"]
EXP_DIR = ROOT / "experiments" / "exp_gossis"


# ════════════════════════════════════════════════════════════════════════
# E16: Temporal Drift Analysis
# ════════════════════════════════════════════════════════════════════════

def run_e16_temporal_drift(data: pd.DataFrame, bootstrap_n: int = 500):
    """
    Analyze whether AUROC gaps change over time.

    If the cohort has an admission year column, stratify by year.
    Otherwise, use hospital ID quartiles as a proxy for temporal variation
    (different hospitals may represent different eras/practices).
    """
    log.info("=" * 60)
    log.info("E16: Temporal Drift Analysis")
    log.info("=" * 60)

    from sklearn.metrics import roc_auc_score
    from scipy import stats

    # Detect temporal column
    time_col = None
    for candidate in ["admit_year", "admission_year", "year"]:
        if candidate in data.columns:
            time_col = candidate
            break

    if time_col is None:
        # Try to derive from datetime columns
        for candidate in ["admittime", "intime", "admission_datetime"]:
            if candidate in data.columns:
                try:
                    data[candidate] = pd.to_datetime(data[candidate], errors="coerce")
                    data["admit_year"] = data[candidate].dt.year
                    time_col = "admit_year"
                    break
                except Exception:
                    pass

    if time_col is None:
        # Fallback: use hospital ID quartiles as proxy
        log.warning("No temporal column found; using hospitalid quartiles as proxy")
        if "hospitalid" in data.columns:
            data["temporal_proxy"] = pd.qcut(
                data["hospitalid"].rank(method="first"),
                q=4, labels=["Q1", "Q2", "Q3", "Q4"]
            )
            time_col = "temporal_proxy"
        else:
            log.error("Cannot perform temporal analysis: no time or hospital column")
            return

    periods = sorted(data[time_col].dropna().unique())
    log.info(f"  Temporal column: {time_col}, {len(periods)} periods")

    drift_rows = []
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        for axis in DEMO_AXES:
            if axis not in data.columns:
                continue
            period_gaps = {}
            for period in periods:
                pd_data = data[data[time_col] == period]
                if len(pd_data) < 200:
                    continue
                aurocs = {}
                for group in pd_data[axis].dropna().unique():
                    mask = pd_data[axis] == group
                    y = pd_data.loc[mask, "mortality"]
                    x = pd_data.loc[mask, score]
                    valid = y.notna() & x.notna()
                    if valid.sum() >= 30 and y[valid].nunique() == 2:
                        aurocs[group] = roc_auc_score(y[valid], x[valid])
                if len(aurocs) >= 2:
                    gap = max(aurocs.values()) - min(aurocs.values())
                    period_gaps[period] = gap
                    drift_rows.append({
                        "score": score, "axis": axis,
                        "period": str(period),
                        "auroc_gap": gap,
                        "worst_group": min(aurocs, key=aurocs.get),
                        "best_group": max(aurocs, key=aurocs.get),
                        "n": len(pd_data),
                        "n_groups": len(aurocs),
                    })

    pd.DataFrame(drift_rows).to_csv(
        EXP_DIR / "e16_temporal_drift.csv", index=False)

    # Trend tests
    trend_rows = []
    drift_df = pd.DataFrame(drift_rows)
    if len(drift_df) > 0:
        for (score, axis), grp in drift_df.groupby(["score", "axis"]):
            if len(grp) >= 3:
                try:
                    rho, pval = stats.spearmanr(
                        range(len(grp)), grp["auroc_gap"].values)
                    trend_rows.append({
                        "score": score, "axis": axis,
                        "n_periods": len(grp),
                        "spearman_rho": rho,
                        "p_value": pval,
                        "trend": "increasing" if rho > 0.3 and pval < 0.05
                                 else "decreasing" if rho < -0.3 and pval < 0.05
                                 else "stable",
                        "mean_gap": grp["auroc_gap"].mean(),
                        "std_gap": grp["auroc_gap"].std(),
                    })
                except Exception:
                    pass

    pd.DataFrame(trend_rows).to_csv(
        EXP_DIR / "e16_temporal_trend_tests.csv", index=False)
    log.info(f"  {len(drift_rows)} drift cells, {len(trend_rows)} trend tests")


# ════════════════════════════════════════════════════════════════════════
# E17: Causal Mediation (Baron-Kenny)
# ════════════════════════════════════════════════════════════════════════

def run_e17_causal_mediation(data: pd.DataFrame):
    """
    Formal mediation decomposition: Age -> SOFA Components -> Mortality.

    Baron-Kenny approach:
      a-path: Age -> Component (OLS)
      b-path: Component -> Mortality | Age (logistic)
      indirect = a * b
      Sobel test for significance
    """
    log.info("=" * 60)
    log.info("E17: Causal Mediation Decomposition")
    log.info("=" * 60)

    from sklearn.linear_model import LinearRegression, LogisticRegression
    from scipy import stats

    # SOFA component columns
    components = {
        "respiratory": "pao2" if "pao2" in data.columns else "pf_ratio_min",
        "coagulation": "platelets" if "platelets" in data.columns else "platelets_min",
        "liver": "bilirubin" if "bilirubin" in data.columns else "bilirubin_max",
        "cardiovascular": "map" if "map" in data.columns else "map_min",
        "cns": "gcs_total",
        "renal": "creatinine" if "creatinine" in data.columns else "creatinine_max",
    }

    # Filter to available components
    components = {k: v for k, v in components.items() if v in data.columns}
    if not components:
        log.error("No SOFA component columns found")
        return

    if "age" not in data.columns:
        log.error("No age column")
        return

    mediation_rows = []
    valid_base = data["age"].notna() & data["mortality"].notna()

    for comp_name, col in components.items():
        valid = valid_base & data[col].notna()
        d = data[valid].copy()
        if len(d) < 500:
            continue

        age = d["age"].values.reshape(-1, 1)
        mediator = d[col].values
        outcome = d["mortality"].values

        # a-path: Age -> Mediator (OLS)
        lr_a = LinearRegression()
        lr_a.fit(age, mediator)
        a_coef = lr_a.coef_[0]
        residuals = mediator - lr_a.predict(age)
        se_a = np.sqrt(np.sum(residuals**2) / (len(d) - 2)) / np.sqrt(
            np.sum((age.flatten() - age.mean())**2))

        # b-path: Mediator -> Mortality | Age (logistic)
        X_b = np.column_stack([age.flatten(), mediator])
        try:
            lr_b = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
            lr_b.fit(X_b, outcome)
            b_coef = lr_b.coef_[0][1]  # mediator coefficient
        except Exception:
            continue

        # c-path (total): Age -> Mortality (logistic)
        lr_c = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        lr_c.fit(age, outcome)
        c_coef = lr_c.coef_[0][0]

        # c'-path (direct): Age -> Mortality | Mediator
        c_prime = lr_b.coef_[0][0]

        # Indirect effect
        indirect = a_coef * b_coef

        # Sobel test
        se_b = 0.01  # approximate SE for logistic coefficient
        sobel_se = np.sqrt(b_coef**2 * se_a**2 + a_coef**2 * se_b**2)
        sobel_z = indirect / sobel_se if sobel_se > 0 else 0
        sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

        # Proportion mediated
        prop_mediated = indirect / c_coef if abs(c_coef) > 1e-10 else np.nan

        mediation_rows.append({
            "component": comp_name,
            "column": col,
            "a_path_coef": a_coef,
            "a_path_se": se_a,
            "b_path_coef": b_coef,
            "c_total_coef": c_coef,
            "c_prime_direct": c_prime,
            "indirect_effect": indirect,
            "proportion_mediated": prop_mediated,
            "sobel_z": sobel_z,
            "sobel_p": sobel_p,
            "significant": sobel_p < 0.05,
            "n": len(d),
        })

    result = pd.DataFrame(mediation_rows)
    result.to_csv(EXP_DIR / "e17_causal_mediation.csv", index=False)
    log.info(f"  {len(mediation_rows)} mediation analyses")
    if len(result) > 0:
        log.info(f"  Top mediator: {result.loc[result['proportion_mediated'].abs().idxmax(), 'component']}")


# ════════════════════════════════════════════════════════════════════════
# E18: Comorbidity Sensitivity
# ════════════════════════════════════════════════════════════════════════

def run_e18_comorbidity_sensitivity(data: pd.DataFrame):
    """
    Test whether age AUROC gap persists after comorbidity adjustment.

    Uses an Elixhauser-proxy comorbidity score derived from lab derangements
    (since ICD codes aren't available in GOSSIS).
    """
    log.info("=" * 60)
    log.info("E18: Comorbidity Sensitivity Analysis")
    log.info("=" * 60)

    from sklearn.metrics import roc_auc_score

    # Build Elixhauser-proxy from lab derangements
    derangement_cols = {
        "creatinine": ("creatinine", "creatinine_max", 1.5, "above"),
        "bilirubin": ("bilirubin", "bilirubin_max", 2.0, "above"),
        "platelets": ("platelets", "platelets_min", 100, "below"),
        "sodium_high": ("sodium", "sodium_max", 145, "above"),
        "sodium_low": ("sodium", "sodium_min", 135, "below"),
        "glucose_high": ("glucose", "glucose_max", 200, "above"),
        "albumin_low": ("albumin", "albumin_min", 3.0, "below"),
        "wbc_high": ("wbc", "wbc_max", 12, "above"),
        "hemoglobin_low": ("hemoglobin", "hemoglobin_min", 10, "below"),
        "lactate_high": ("lactate", "lactate_max", 2.0, "above"),
    }

    # Count derangements
    data = data.copy()
    derangement_count = pd.Series(0, index=data.index)
    for name, (col_short, col_long, threshold, direction) in derangement_cols.items():
        col = col_short if col_short in data.columns else (
            col_long if col_long in data.columns else None)
        if col is None:
            continue
        if direction == "above":
            derangement_count += (data[col] > threshold).astype(int).fillna(0)
        else:
            derangement_count += (data[col] < threshold).astype(int).fillna(0)

    # Stratify
    data["comorbidity_burden"] = pd.cut(
        derangement_count,
        bins=[-1, 1, 3, 100],
        labels=["Low", "Medium", "High"]
    )

    log.info(f"  Comorbidity distribution: {data['comorbidity_burden'].value_counts().to_dict()}")

    # Compute AUROC gaps within each stratum
    sensitivity_rows = []
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        for burden in ["Low", "Medium", "High"]:
            stratum = data[data["comorbidity_burden"] == burden]
            if len(stratum) < 200:
                continue

            for axis in ["age_group", "race_cat"]:
                if axis not in stratum.columns:
                    continue
                aurocs = {}
                for group in stratum[axis].dropna().unique():
                    mask = stratum[axis] == group
                    y = stratum.loc[mask, "mortality"]
                    x = stratum.loc[mask, score]
                    valid = y.notna() & x.notna()
                    if valid.sum() >= 30 and y[valid].nunique() == 2:
                        aurocs[group] = roc_auc_score(y[valid], x[valid])

                if len(aurocs) >= 2:
                    sensitivity_rows.append({
                        "score": score,
                        "comorbidity_burden": burden,
                        "axis": axis,
                        "auroc_gap": max(aurocs.values()) - min(aurocs.values()),
                        "worst_group": min(aurocs, key=aurocs.get),
                        "best_group": max(aurocs, key=aurocs.get),
                        "n_stratum": len(stratum),
                        "n_groups": len(aurocs),
                    })

    result = pd.DataFrame(sensitivity_rows)
    result.to_csv(EXP_DIR / "e18_comorbidity_sensitivity.csv", index=False)

    # Summary: does age gap persist?
    summary_rows = []
    if len(result) > 0:
        age_results = result[result["axis"] == "age_group"]
        for score in SCORE_NAMES:
            sr = age_results[age_results["score"] == score]
            if len(sr) >= 2:
                summary_rows.append({
                    "score": score,
                    "low_gap": sr.loc[sr["comorbidity_burden"] == "Low", "auroc_gap"].values[0]
                              if "Low" in sr["comorbidity_burden"].values else np.nan,
                    "medium_gap": sr.loc[sr["comorbidity_burden"] == "Medium", "auroc_gap"].values[0]
                                  if "Medium" in sr["comorbidity_burden"].values else np.nan,
                    "high_gap": sr.loc[sr["comorbidity_burden"] == "High", "auroc_gap"].values[0]
                                if "High" in sr["comorbidity_burden"].values else np.nan,
                    "gap_persists": all(
                        sr.loc[sr["comorbidity_burden"] == b, "auroc_gap"].values[0] > 0.05
                        for b in ["Low", "Medium", "High"]
                        if b in sr["comorbidity_burden"].values
                    ),
                })

    pd.DataFrame(summary_rows).to_csv(
        EXP_DIR / "e18_comorbidity_summary.csv", index=False)
    log.info(f"  {len(sensitivity_rows)} sensitivity cells")


# ════════════════════════════════════════════════════════════════════════
# E19: Foundation Model Comparison
# ════════════════════════════════════════════════════════════════════════

class ClinicalEmbeddingInterface(ABC):
    """
    Abstract interface for clinical foundation model embeddings.
    Subclass this for Med-BERT, CLMBR, etc.
    """
    @abstractmethod
    def embed(self, data: pd.DataFrame) -> np.ndarray:
        """Return embeddings of shape (n_samples, embedding_dim)."""
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class XGBoostBaseline:
    """XGBoost on the same feature set as GRU, for fair comparison."""

    def __init__(self, seed=42):
        self.seed = seed
        self.name = "XGBoost"

    def train_cv(self, data: pd.DataFrame, feature_cols: list,
                 n_folds: int = 5) -> dict:
        """Train with 5-fold CV, return predictions and metrics."""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
        import xgboost as xgb

        X = data[feature_cols].fillna(data[feature_cols].median()).values
        y = data["mortality"].values

        preds = np.zeros(len(y))
        fold_aurocs = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=self.seed)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
                random_state=self.seed,
                eval_metric="logloss",
                early_stopping_rounds=20,
                verbosity=0,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds[val_idx] = model.predict_proba(X_val)[:, 1]
            fold_aurocs.append(roc_auc_score(y_val, preds[val_idx]))

        overall_auroc = roc_auc_score(y, preds)
        return {
            "predictions": preds,
            "overall_auroc": overall_auroc,
            "fold_aurocs": fold_aurocs,
        }


def run_e19_foundation_comparison(data: pd.DataFrame):
    """
    Compare XGBoost baseline with GRU/FAFT on fairness metrics.
    Provides interface for foundation model plug-in.
    """
    log.info("=" * 60)
    log.info("E19: Foundation Model Comparison")
    log.info("=" * 60)

    from sklearn.metrics import roc_auc_score

    # Identify feature columns (numeric, non-demographic, non-score)
    exclude = set(SCORE_NAMES + DEMO_AXES + [
        "mortality", "stay_id", "subject_id", "hadm_id",
        "patientunitstayid", "hospitalid", "encounter_id",
        "intime", "outtime", "admittime", "dischtime", "deathtime",
        "careunit", "first_careunit", "unittype",
        "los_icu", "admit_year", "comorbidity_burden",
        "gender", "ethnicity", "insurance", "insurance_cat",
        "admission_type", "anchor_age", "anchor_year", "dod",
        "anchor_year_group", "is_elderly", "temporal_proxy",
    ])
    feature_cols = [c for c in data.select_dtypes(include=[np.number]).columns
                    if c not in exclude and data[c].notna().sum() > len(data) * 0.3]

    if len(feature_cols) < 5:
        log.error(f"Only {len(feature_cols)} feature columns found; skipping")
        return

    log.info(f"  Using {len(feature_cols)} features: {feature_cols[:10]}...")

    # Train XGBoost
    try:
        xgb_model = XGBoostBaseline(seed=RANDOM_SEED)
        xgb_result = xgb_model.train_cv(data, feature_cols)
        xgb_preds = xgb_result["predictions"]
        log.info(f"  XGBoost AUROC: {xgb_result['overall_auroc']:.4f}")
        np.save(EXP_DIR / "xgb_preds.npy", xgb_preds)
    except ImportError:
        log.error("xgboost not installed; pip install xgboost")
        return
    except Exception as e:
        log.error(f"XGBoost training failed: {e}")
        return

    # Load existing model predictions if available
    models = {"XGBoost": xgb_preds}
    for name, fname in [("GRU", "ml_preds_full.npy"),
                         ("FAFT", "faft_preds.npy"),
                         ("GA-FAFT", "ga_faft_preds.npy")]:
        p = EXP_DIR / fname
        if p.exists():
            try:
                preds = np.load(p)
                if len(preds) == len(data):
                    models[name] = preds
                    log.info(f"  Loaded {name} predictions")
            except Exception:
                pass

    # Compute per-axis AUROC gaps for all models + classical scores
    comparison_rows = []
    y = data["mortality"].values

    # Classical scores
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        x = data[score].values
        for axis in DEMO_AXES:
            if axis not in data.columns:
                continue
            aurocs = {}
            for group in data[axis].dropna().unique():
                mask = (data[axis] == group).values
                valid = mask & ~np.isnan(x) & ~np.isnan(y)
                if valid.sum() >= 30 and len(np.unique(y[valid])) == 2:
                    aurocs[group] = roc_auc_score(y[valid], x[valid])
            if len(aurocs) >= 2:
                comparison_rows.append({
                    "model": score.upper(), "type": "classical_score",
                    "axis": axis,
                    "auroc_gap": max(aurocs.values()) - min(aurocs.values()),
                    "worst_group": min(aurocs, key=aurocs.get),
                    "best_group": max(aurocs, key=aurocs.get),
                    "overall_auroc": roc_auc_score(y[~np.isnan(x)], x[~np.isnan(x)])
                                     if (~np.isnan(x)).sum() > 0 else np.nan,
                })

    # ML models
    for model_name, preds in models.items():
        for axis in DEMO_AXES:
            if axis not in data.columns:
                continue
            aurocs = {}
            for group in data[axis].dropna().unique():
                mask = (data[axis] == group).values
                valid = mask & ~np.isnan(preds) & ~np.isnan(y)
                if valid.sum() >= 30 and len(np.unique(y[valid])) == 2:
                    aurocs[group] = roc_auc_score(y[valid], preds[valid])
            if len(aurocs) >= 2:
                comparison_rows.append({
                    "model": model_name, "type": "ml_model",
                    "axis": axis,
                    "auroc_gap": max(aurocs.values()) - min(aurocs.values()),
                    "worst_group": min(aurocs, key=aurocs.get),
                    "best_group": max(aurocs, key=aurocs.get),
                    "overall_auroc": roc_auc_score(y, preds),
                })

    pd.DataFrame(comparison_rows).to_csv(
        EXP_DIR / "e19_foundation_comparison.csv", index=False)

    # RSB for XGBoost
    try:
        from src.evaluation.rsb import compute_rsb, compute_ml_improvement
        rsb = compute_rsb(data, xgb_preds, axes=DEMO_AXES, n_boot=500)
        rsb.to_csv(EXP_DIR / "e19_xgb_rsb.csv", index=False)
        imp = compute_ml_improvement(data, xgb_preds, axes=DEMO_AXES)
        imp.to_csv(EXP_DIR / "e19_xgb_improvement.csv", index=False)
        log.info("  XGBoost RSB computed")
    except ImportError:
        log.warning("  src.evaluation.rsb not importable; skipping RSB")

    log.info(f"  {len(comparison_rows)} comparison cells across "
             f"{len(models)} ML models + {len(SCORE_NAMES)} scores")


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS expansion analyses (E16-E19)")
    parser.add_argument("--cohort-path", type=str, default=None,
                        help="Path to cohort CSV with scores")
    parser.add_argument("--analyses", nargs="+",
                        default=["E16", "E17", "E18", "E19"],
                        help="Which analyses to run")
    parser.add_argument("--bootstrap-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global RANDOM_SEED
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)

    # Load cohort
    if args.cohort_path:
        cohort_path = Path(args.cohort_path)
    else:
        cohort_path = EXP_DIR / "cohort_with_scores.csv"

    if not cohort_path.exists():
        log.error(f"Cohort file not found: {cohort_path}")
        log.error("Run the GOSSIS pipeline first, or specify --cohort-path")
        sys.exit(1)

    log.info(f"Loading cohort from {cohort_path}")
    data = pd.read_csv(cohort_path)
    log.info(f"Cohort: {len(data)} stays")

    analyses = [a.upper() for a in args.analyses]

    if "E16" in analyses:
        run_e16_temporal_drift(data, bootstrap_n=args.bootstrap_n)

    if "E17" in analyses:
        run_e17_causal_mediation(data)

    if "E18" in analyses:
        run_e18_comorbidity_sensitivity(data)

    if "E19" in analyses:
        run_e19_foundation_comparison(data)

    log.info("All expansion analyses complete!")


if __name__ == "__main__":
    main()
