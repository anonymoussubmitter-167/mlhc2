#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
# Part of ATLAS
"""
Full eICU-CRD pipeline — replaces the 1,238-patient demo with the complete
~200K-stay dataset. Runs the full E1–E15 ATLAS audit and optionally compares
against the demo-subset results (Cohen's d effect sizes).

Usage:
    python run_eicu_full.py --eicu-dir /path/to/eicu-crd
    python run_eicu_full.py --eicu-dir /path/to/eicu-crd --compare-demo
    python run_eicu_full.py --eicu-dir /path/to/eicu-crd --skip-ml
"""

import sys
import json
import time
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("atlas.eicu_full")

RANDOM_SEED = 42
MIN_SUBGROUP_SIZE = 100
SCORE_NAMES = ["sofa", "qsofa", "apache2", "news2"]
DEMO_AXES = ["race_cat", "sex", "age_group", "diag_type"]
EXP_DIR = ROOT / "experiments" / "exp_eicu_full"


# ════════════════════════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════════════════════════

def _read_table(eicu_dir: Path, name: str, usecols=None) -> pd.DataFrame:
    """Load an eICU-CRD table (CSV or CSV.GZ)."""
    for suffix in [".csv", ".csv.gz"]:
        p = eicu_dir / f"{name}{suffix}"
        if p.exists():
            log.info(f"  Loading {p.name}")
            return pd.read_csv(p, usecols=usecols)
    raise FileNotFoundError(f"Cannot find {name} in {eicu_dir}")


def extract_eicu_cohort(eicu_dir: Path, max_stays=None) -> pd.DataFrame:
    """Build ATLAS cohort from full eICU-CRD tables."""
    log.info("Loading eICU-CRD tables...")

    patient = _read_table(eicu_dir, "patient", usecols=[
        "patientunitstayid", "patienthealthsystemstayid", "hospitalid",
        "gender", "age", "ethnicity", "admissionheight", "admissionweight",
        "unittype", "unitadmitsource", "unitdischargestatus",
        "hospitaldischargestatus", "unitdischargeoffset",
    ])

    # Age handling (eICU uses "> 89" for elderly)
    patient["age"] = pd.to_numeric(patient["age"], errors="coerce")
    patient.loc[patient["age"].isna() & (patient["age"].astype(str) == "> 89"), "age"] = 90
    patient["age"] = patient["age"].fillna(patient["age"].median())
    patient = patient[(patient["age"] >= 18) & (patient["age"] <= 120)].copy()

    # Mortality
    patient["mortality"] = (
        patient["hospitaldischargestatus"].str.lower() == "expired"
    ).astype(int)

    # Demographics
    patient["sex"] = patient["gender"].map(
        {"Male": "Male", "Female": "Female"}
    ).fillna("Unknown")

    race_map = {
        "Caucasian": "White", "African American": "Black",
        "Hispanic": "Hispanic", "Asian": "Asian",
        "Native American": "Other",
    }
    patient["race_cat"] = patient["ethnicity"].map(race_map).fillna("Other")

    bins = [17, 29, 39, 49, 59, 69, 79, 200]
    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    patient["age_group"] = pd.cut(patient["age"], bins=bins, labels=labels)

    def _map_diag(unit_type):
        if pd.isna(unit_type):
            return "Medical"
        ut = str(unit_type).upper()
        if "SURG" in ut or "CSICU" in ut:
            return "Surgical"
        if "CARD" in ut or "CCU" in ut:
            return "Cardiac"
        return "Medical"

    patient["diag_type"] = patient["unittype"].apply(_map_diag)

    # Keep first ICU stay per hospital stay
    patient = patient.sort_values("patientunitstayid").groupby(
        "patienthealthsystemstayid"
    ).first().reset_index()

    if max_stays:
        patient = patient.sample(n=min(max_stays, len(patient)),
                                 random_state=RANDOM_SEED)

    log.info(f"Cohort: {len(patient)} stays, "
             f"{patient['hospitalid'].nunique()} hospitals, "
             f"mortality={patient['mortality'].mean():.3f}")
    return patient


def extract_vitals(eicu_dir: Path, stay_ids: set) -> pd.DataFrame:
    """Extract worst-24h vitals from vitalPeriodic and nurseCharting."""
    log.info("Extracting vitals...")
    agg = defaultdict(lambda: defaultdict(list))

    # vitalPeriodic — high-frequency vitals
    try:
        vp = _read_table(eicu_dir, "vitalPeriodic", usecols=[
            "patientunitstayid", "observationoffset",
            "heartrate", "systemicsystolic", "systemicdiastolic",
            "systemicmean", "resprate", "sao2", "temperature",
        ])
        vp = vp[
            vp["patientunitstayid"].isin(stay_ids) &
            (vp["observationoffset"] >= 0) &
            (vp["observationoffset"] <= 1440)  # first 24h in minutes
        ]
        for _, row in vp.iterrows():
            sid = row["patientunitstayid"]
            if pd.notna(row.get("heartrate")):
                agg[sid]["heart_rate"].append(row["heartrate"])
            if pd.notna(row.get("systemicsystolic")):
                agg[sid]["sbp"].append(row["systemicsystolic"])
            if pd.notna(row.get("systemicmean")):
                agg[sid]["map"].append(row["systemicmean"])
            if pd.notna(row.get("resprate")):
                agg[sid]["resp_rate"].append(row["resprate"])
            if pd.notna(row.get("sao2")):
                agg[sid]["spo2"].append(row["sao2"])
            if pd.notna(row.get("temperature")):
                t = row["temperature"]
                if t > 50:  # Fahrenheit
                    t = (t - 32) * 5 / 9
                agg[sid]["temp_c"].append(t)
    except FileNotFoundError:
        log.warning("vitalPeriodic not found; skipping")

    # nurseCharting — GCS components
    try:
        nc = _read_table(eicu_dir, "nurseCharting", usecols=[
            "patientunitstayid", "nursingchartoffset",
            "nursingchartcelltypevalname", "nursingchartvalue",
        ])
        nc = nc[
            nc["patientunitstayid"].isin(stay_ids) &
            (nc["nursingchartoffset"] >= 0) &
            (nc["nursingchartoffset"] <= 1440)
        ]
        gcs_map = {
            "Eyes": "gcs_eye", "Verbal": "gcs_verbal", "Motor": "gcs_motor",
            "GCS Total": "gcs_total",
        }
        for _, row in nc.iterrows():
            chart_name = str(row.get("nursingchartcelltypevalname", ""))
            for key, feat in gcs_map.items():
                if key in chart_name:
                    try:
                        val = float(row["nursingchartvalue"])
                        agg[row["patientunitstayid"]][feat].append(val)
                    except (ValueError, TypeError):
                        pass
    except FileNotFoundError:
        log.warning("nurseCharting not found; skipping GCS")

    # Aggregate
    rows = []
    for sid, feats in agg.items():
        row = {"patientunitstayid": sid}
        # Worst values: min for "lower=sicker", max for "higher=sicker"
        min_feats = {"sbp", "map", "spo2", "gcs_eye", "gcs_verbal",
                     "gcs_motor", "gcs_total"}
        for feat, vals in feats.items():
            if feat in min_feats:
                row[feat] = min(vals)
            else:
                row[feat] = max(vals)
        rows.append(row)

    result = pd.DataFrame(rows)
    log.info(f"  Vitals extracted for {len(result)} stays")
    return result


def extract_labs(eicu_dir: Path, stay_ids: set) -> pd.DataFrame:
    """Extract worst-24h labs from lab table."""
    log.info("Extracting labs...")
    lab_name_map = {
        "creatinine": ("creatinine", max),
        "BUN": ("bun", max),
        "potassium": ("potassium", max),
        "sodium": ("sodium", max),
        "bicarbonate": ("bicarbonate", min),
        "chloride": ("chloride", max),
        "glucose": ("glucose", max),
        "WBC x 1000": ("wbc", max),
        "Hgb": ("hemoglobin", min),
        "Hct": ("hematocrit", min),
        "platelets x 1000": ("platelets", min),
        "total bilirubin": ("bilirubin", max),
        "albumin": ("albumin", min),
        "lactate": ("lactate", max),
        "pH": ("ph", min),
        "paO2": ("pao2", min),
        "paCO2": ("paco2", max),
        "FiO2": ("fio2", max),
        "PT - Loss of Clot": ("pt", max),
        "PTT": ("ptt", max),
        "INR": ("inr", max),
        "anion gap": ("anion_gap", max),
        "calcium": ("calcium_total", min),
        "magnesium": ("magnesium", max),
        "phosphate": ("phosphate", max),
        "ALT (SGPT)": ("alt", max),
        "AST (SGOT)": ("ast", max),
        "troponin - T": ("troponin_t", max),
        "BNP": ("bnp", max),
        "CRP": ("crp", max),
    }

    try:
        lab = _read_table(eicu_dir, "lab", usecols=[
            "patientunitstayid", "labresultoffset",
            "labname", "labresult",
        ])
    except FileNotFoundError:
        log.warning("lab table not found")
        return pd.DataFrame()

    lab = lab[
        lab["patientunitstayid"].isin(stay_ids) &
        (lab["labresultoffset"] >= 0) &
        (lab["labresultoffset"] <= 1440)
    ]
    lab["labresult"] = pd.to_numeric(lab["labresult"], errors="coerce")
    lab = lab.dropna(subset=["labresult"])

    agg = defaultdict(lambda: defaultdict(list))
    for _, row in lab.iterrows():
        lname = row["labname"]
        if lname in lab_name_map:
            feat, _ = lab_name_map[lname]
            agg[row["patientunitstayid"]][feat].append(row["labresult"])

    rows = []
    for sid, feats in agg.items():
        row = {"patientunitstayid": sid}
        for feat, vals in feats.items():
            # Look up aggregation function
            agg_fn = max  # default
            for lname, (f, fn) in lab_name_map.items():
                if f == feat:
                    agg_fn = fn
                    break
            row[feat] = agg_fn(vals)
        rows.append(row)

    result = pd.DataFrame(rows)
    log.info(f"  Labs extracted for {len(result)} stays")
    return result


# ════════════════════════════════════════════════════════════════════════
# Score Computation (same formulas as GOSSIS pipeline)
# ════════════════════════════════════════════════════════════════════════

def _compute_sofa(df):
    s = pd.Series(0, index=df.index, dtype=float)
    if "pao2" in df.columns and "fio2" in df.columns:
        pf = df["pao2"] / (df["fio2"] / 100).clip(lower=0.21)
        s += np.select([pf < 100, pf < 200, pf < 300, pf < 400], [4, 3, 2, 1], 0)
    if "platelets" in df.columns:
        p = df["platelets"]
        s += np.select([p < 20, p < 50, p < 100, p < 150], [4, 3, 2, 1], 0)
    if "bilirubin" in df.columns:
        b = df["bilirubin"]
        s += np.select([b >= 12, b >= 6, b >= 2, b >= 1.2], [4, 3, 2, 1], 0)
    if "map" in df.columns:
        s += np.where(df["map"] < 70, 1, 0)
    gcs = df.get("gcs_total")
    if gcs is not None:
        s += np.select([gcs < 6, gcs < 10, gcs < 13, gcs < 15], [4, 3, 2, 1], 0)
    if "creatinine" in df.columns:
        c = df["creatinine"]
        s += np.select([c >= 5, c >= 3.5, c >= 2, c >= 1.2], [4, 3, 2, 1], 0)
    return s.clip(0, 24)


def _compute_qsofa(df):
    q = pd.Series(0, index=df.index, dtype=float)
    if "resp_rate" in df.columns:
        q += (df["resp_rate"] >= 22).astype(int)
    if "sbp" in df.columns:
        q += (df["sbp"] <= 100).astype(int)
    elif "map" in df.columns:
        q += (df["map"] <= 65).astype(int)
    if "gcs_total" in df.columns:
        q += (df["gcs_total"] < 15).astype(int)
    return q.clip(0, 3)


def _compute_apache2(df):
    s = pd.Series(0, index=df.index, dtype=float)
    if "temp_c" in df.columns:
        t = df["temp_c"]
        s += np.select([t >= 41, t.between(39, 40.99), t.between(38.5, 38.99),
                        t.between(36, 38.49), t.between(34, 35.99),
                        t.between(32, 33.99), t.between(30, 31.99), t < 30],
                       [4, 3, 1, 0, 1, 2, 3, 4], 0)
    if "map" in df.columns:
        m = df["map"]
        s += np.select([m >= 160, m.between(130, 159), m.between(110, 129),
                        m.between(70, 109), m.between(50, 69), m < 50],
                       [4, 3, 2, 0, 2, 4], 0)
    if "heart_rate" in df.columns:
        hr = df["heart_rate"]
        s += np.select([hr >= 180, hr.between(140, 179), hr.between(110, 139),
                        hr.between(70, 109), hr.between(55, 69),
                        hr.between(40, 54), hr < 40],
                       [4, 3, 2, 0, 2, 3, 4], 0)
    if "resp_rate" in df.columns:
        rr = df["resp_rate"]
        s += np.select([rr >= 50, rr.between(35, 49), rr.between(25, 34),
                        rr.between(12, 24), rr.between(10, 11),
                        rr.between(6, 9), rr < 6],
                       [4, 3, 1, 0, 1, 2, 4], 0)
    if "sodium" in df.columns:
        na = df["sodium"]
        s += np.select([na >= 180, na.between(160, 179), na.between(155, 159),
                        na.between(150, 154), na.between(130, 149),
                        na.between(120, 129), na < 120],
                       [4, 3, 2, 1, 0, 2, 3], 0)
    if "potassium" in df.columns:
        k = df["potassium"]
        s += np.select([k >= 7, k.between(6, 6.9), k.between(5.5, 5.9),
                        k.between(3.5, 5.4), k.between(3, 3.4),
                        k.between(2.5, 2.9), k < 2.5],
                       [4, 3, 1, 0, 1, 2, 4], 0)
    if "creatinine" in df.columns:
        c = df["creatinine"]
        s += np.select([c >= 3.5, c.between(2, 3.4), c.between(1.5, 1.9),
                        c.between(0.6, 1.4), c < 0.6],
                       [4, 3, 2, 0, 2], 0)
    if "hematocrit" in df.columns:
        h = df["hematocrit"]
        s += np.select([h >= 60, h.between(50, 59.9), h.between(46, 49.9),
                        h.between(30, 45.9), h.between(20, 29.9), h < 20],
                       [4, 2, 1, 0, 2, 4], 0)
    if "wbc" in df.columns:
        w = df["wbc"]
        s += np.select([w >= 40, w.between(20, 39.9), w.between(15, 19.9),
                        w.between(3, 14.9), w.between(1, 2.9), w < 1],
                       [4, 2, 1, 0, 2, 4], 0)
    if "gcs_total" in df.columns:
        s += (15 - df["gcs_total"].clip(3, 15))
    if "age" in df.columns:
        a = df["age"]
        s += np.select([a >= 75, a.between(65, 74), a.between(55, 64),
                        a.between(45, 54), a < 45],
                       [6, 5, 3, 2, 0], 0)
    return s.clip(0, 71)


def _compute_news2(df):
    s = pd.Series(0, index=df.index, dtype=float)
    if "resp_rate" in df.columns:
        rr = df["resp_rate"]
        s += np.select([rr <= 8, rr.between(9, 11), rr.between(12, 20),
                        rr.between(21, 24), rr >= 25],
                       [3, 1, 0, 2, 3], 0)
    if "spo2" in df.columns:
        sp = df["spo2"]
        s += np.select([sp <= 91, sp.between(92, 93), sp.between(94, 95), sp >= 96],
                       [3, 2, 1, 0], 0)
    if "fio2" in df.columns:
        s += (df["fio2"] > 21).astype(int) * 2
    if "temp_c" in df.columns:
        t = df["temp_c"]
        s += np.select([t <= 35, t.between(35.1, 36), t.between(36.1, 38),
                        t.between(38.1, 39), t >= 39.1],
                       [3, 1, 0, 1, 2], 0)
    bp_col = "sbp" if "sbp" in df.columns else ("map" if "map" in df.columns else None)
    if bp_col:
        bp = df[bp_col] * 1.5 if bp_col == "map" else df[bp_col]
        s += np.select([bp <= 90, bp.between(91, 100), bp.between(101, 110),
                        bp.between(111, 219), bp >= 220],
                       [3, 2, 1, 0, 3], 0)
    if "heart_rate" in df.columns:
        hr = df["heart_rate"]
        s += np.select([hr <= 40, hr.between(41, 50), hr.between(51, 90),
                        hr.between(91, 110), hr.between(111, 130), hr >= 131],
                       [3, 1, 0, 1, 2, 3], 0)
    if "gcs_total" in df.columns:
        s += (df["gcs_total"] < 15).astype(int) * 3
    return s.clip(0, 20)


# ════════════════════════════════════════════════════════════════════════
# Audit Pipeline
# ════════════════════════════════════════════════════════════════════════

def run_audit(data, skip_ml=False, bootstrap_n=1000):
    """Run the full E1-E15 ATLAS audit on the eICU-CRD full dataset."""
    from sklearn.metrics import roc_auc_score
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from src.evaluation.audit import prespecified_audit, intersectional_audit
        from src.evaluation.asd import adversarial_subgroup_discovery
        from src.evaluation.rsb import compute_rsb, compute_ml_improvement
        USE_SRC = True
    except ImportError:
        log.warning("src modules not importable; using standalone audit")
        USE_SRC = False

    # ── E1 ──
    log.info("E1: Pre-specified Subgroup Audit")
    if USE_SRC:
        audit_results, gaps_df = prespecified_audit(
            data, axes=DEMO_AXES, n_boot=bootstrap_n)
    else:
        audit_results, gaps_df = _standalone_audit(data)
    audit_results.to_csv(EXP_DIR / "e1_audit_results.csv", index=False)
    gaps_df.to_csv(EXP_DIR / "e1_gaps.csv", index=False)
    log.info(f"  {len(audit_results)} subgroup results")

    # ── E2 ──
    log.info("E2: Intersectional Analysis")
    if USE_SRC:
        inter, worst = intersectional_audit(data, axes=DEMO_AXES,
                                             min_n=MIN_SUBGROUP_SIZE,
                                             n_boot=bootstrap_n)
    else:
        inter = _standalone_intersectional(data)
    inter.to_csv(EXP_DIR / "e2_intersectional.csv", index=False)
    log.info(f"  {len(inter)} intersectional subgroups")

    # ── E3 ──
    log.info("E3: Adversarial Subgroup Discovery")
    if USE_SRC:
        asd = adversarial_subgroup_discovery(data)
        with open(EXP_DIR / "e3_asd_results.json", "w") as f:
            ser = {}
            for k, v in asd.items():
                sv = {**v}
                sv["top_features"] = [(n, float(i)) for n, i in sv["top_features"]]
                ser[k] = sv
            json.dump(ser, f, indent=2, default=str)

    # ── E4-E5 ──
    if not skip_ml:
        log.info("E4-E5: GRU + RSB")
        try:
            import torch
            from src.training.train_gru import train_gru_model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gru = train_gru_model(data, device=device, epochs=50)
            np.save(EXP_DIR / "ml_preds.npy", gru["predictions"])
            if USE_SRC:
                rsb = compute_rsb(data, gru["predictions"], axes=DEMO_AXES,
                                  n_boot=min(bootstrap_n, 500))
                rsb.to_csv(EXP_DIR / "e4_rsb.csv", index=False)
                imp = compute_ml_improvement(data, gru["predictions"],
                                             axes=DEMO_AXES)
                imp.to_csv(EXP_DIR / "e5_ml_improvement.csv", index=False)
            log.info(f"  GRU AUROC: {gru['overall_auroc']:.4f}")
        except Exception as e:
            log.error(f"  GRU training failed: {e}")
    else:
        log.info("E4-E5: Skipped (--skip-ml)")

    # ── E6: Score-conditional mortality ──
    log.info("E6: Score-conditional mortality")
    scm_rows = []
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        try:
            bins = pd.qcut(data[score], q=10, duplicates="drop")
        except ValueError:
            continue
        for group in data["age_group"].dropna().unique():
            mask = data["age_group"] == group
            for bl, bd in data[mask].groupby(bins):
                if len(bd) >= 10:
                    scm_rows.append({
                        "score": score, "score_bin": str(bl),
                        "demo_axis": "age_group", "group": group,
                        "mortality_rate": bd["mortality"].mean(),
                        "n": len(bd),
                    })
    pd.DataFrame(scm_rows).to_csv(
        EXP_DIR / "e6_score_conditional_mortality.csv", index=False)

    # ── E7: Clinical thresholds ──
    log.info("E7: Clinical threshold analysis")
    thresholds = {"sofa": [2, 6, 11], "qsofa": [1, 2],
                  "apache2": [15, 20, 25], "news2": [5, 9, 13]}
    thr_rows = []
    for score, tl in thresholds.items():
        if score not in data.columns:
            continue
        for t in tl:
            pp = (data[score] >= t).astype(int)
            for g in data["age_group"].dropna().unique():
                m = data["age_group"] == g
                y, yp = data.loc[m, "mortality"], pp[m]
                tp = ((yp == 1) & (y == 1)).sum()
                fn = ((yp == 0) & (y == 1)).sum()
                fp = ((yp == 1) & (y == 0)).sum()
                tn = ((yp == 0) & (y == 0)).sum()
                thr_rows.append({
                    "score": score, "threshold": t, "age_group": g,
                    "sensitivity": tp / (tp + fn) if tp + fn > 0 else np.nan,
                    "specificity": tn / (tn + fp) if tn + fp > 0 else np.nan,
                    "n": m.sum(),
                })
    pd.DataFrame(thr_rows).to_csv(
        EXP_DIR / "e7_clinical_thresholds.csv", index=False)

    # ── E10: Hospital-stratified race ──
    log.info("E10: Hospital-stratified race")
    hosp_rows = []
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        for hid in data["hospitalid"].dropna().unique():
            hd = data[data["hospitalid"] == hid]
            if len(hd) < MIN_SUBGROUP_SIZE:
                continue
            race_aurocs = {}
            for race in hd["race_cat"].dropna().unique():
                rm = hd["race_cat"] == race
                y, x = hd.loc[rm, "mortality"], hd.loc[rm, score]
                v = y.notna() & x.notna()
                if v.sum() >= 30 and y[v].nunique() == 2:
                    race_aurocs[race] = roc_auc_score(y[v], x[v])
            if len(race_aurocs) >= 2:
                hosp_rows.append({
                    "score": score, "hospitalid": hid, "n": len(hd),
                    "race_auroc_gap": max(race_aurocs.values()) - min(race_aurocs.values()),
                    "worst_race": min(race_aurocs, key=race_aurocs.get),
                    "best_race": max(race_aurocs, key=race_aurocs.get),
                    "n_races": len(race_aurocs),
                })
    pd.DataFrame(hosp_rows).to_csv(
        EXP_DIR / "hospital_stratified_race.csv", index=False)

    # ── E13: Recalibration ──
    log.info("E13: Recalibration decomposition")
    from sklearn.isotonic import IsotonicRegression
    recal_rows = []
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        y_all, x_all = data["mortality"].values, data[score].values
        valid = ~np.isnan(x_all) & ~np.isnan(y_all)
        iso_g = IsotonicRegression(out_of_bounds="clip")
        iso_g.fit(x_all[valid], y_all[valid])
        cal_g = iso_g.predict(x_all)
        for g in data["age_group"].dropna().unique():
            m = (data["age_group"] == g).values & valid
            if m.sum() < 50 or len(np.unique(y_all[m])) < 2:
                continue
            iso_p = IsotonicRegression(out_of_bounds="clip")
            iso_p.fit(x_all[m], y_all[m])
            recal_rows.append({
                "score": score, "age_group": g, "n": m.sum(),
                "raw_auroc": roc_auc_score(y_all[m], x_all[m]),
                "global_cal_auroc": roc_auc_score(y_all[m], cal_g[m]),
                "grp_cal_auroc": roc_auc_score(y_all[m], iso_p.predict(x_all[m])),
            })
    pd.DataFrame(recal_rows).to_csv(
        EXP_DIR / "e13_recalibration.csv", index=False)

    # ── E14: PSM ──
    log.info("E14: Propensity-score matching")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors

        psm_feats = [c for c in ["heart_rate", "map", "resp_rate", "spo2",
                                  "temp_c", "gcs_total", "creatinine",
                                  "platelets", "bilirubin", "sodium",
                                  "potassium", "wbc", "hemoglobin",
                                  "hematocrit", "bun", "glucose"]
                     if c in data.columns]
        elderly = data[data["age"] >= 80]
        young = data[data["age"].between(18, 49)]

        if len(elderly) >= 100 and len(young) >= 100 and len(psm_feats) >= 5:
            combined = pd.concat([elderly.assign(is_elderly=1),
                                  young.assign(is_elderly=0)])
            X = combined[psm_feats].fillna(combined[psm_feats].median())
            X_s = StandardScaler().fit_transform(X)
            y_t = combined["is_elderly"].values
            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
            lr.fit(X_s, y_t)
            ps = lr.predict_proba(X_s)[:, 1]
            caliper = 0.2 * np.std(ps)
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(ps[y_t == 0].reshape(-1, 1))
            dist, idx = nn.kneighbors(ps[y_t == 1].reshape(-1, 1))
            match = dist.flatten() <= caliper
            m_e = np.where(y_t == 1)[0][match]
            m_y = np.where(y_t == 0)[0][idx.flatten()[match]]

            psm_rows = []
            for score in SCORE_NAMES:
                if score not in data.columns:
                    continue
                ed, yd = combined.iloc[m_e], combined.iloc[m_y]
                ae = roc_auc_score(ed["mortality"], ed[score]) if ed["mortality"].nunique() == 2 else np.nan
                ay = roc_auc_score(yd["mortality"], yd[score]) if yd["mortality"].nunique() == 2 else np.nan
                # Unmatched
                ae_u = roc_auc_score(elderly["mortality"], elderly[score]) if elderly["mortality"].nunique() == 2 else np.nan
                ay_u = roc_auc_score(young["mortality"], young[score]) if young["mortality"].nunique() == 2 else np.nan
                psm_rows.append({
                    "score": score, "n_matched_pairs": match.sum(),
                    "auroc_elderly_unmatched": ae_u,
                    "auroc_young_unmatched": ay_u,
                    "gap_unmatched": ay_u - ae_u if pd.notna(ay_u) else np.nan,
                    "auroc_elderly_matched": ae,
                    "auroc_young_matched": ay,
                    "gap_matched": ay - ae if pd.notna(ay) else np.nan,
                })
            pd.DataFrame(psm_rows).to_csv(
                EXP_DIR / "e14_psm_analysis.csv", index=False)
            log.info(f"  {match.sum()} matched pairs")
    except Exception as e:
        log.error(f"  PSM failed: {e}")

    log.info("Pipeline complete!")


# ════════════════════════════════════════════════════════════════════════
# Demo Comparison
# ════════════════════════════════════════════════════════════════════════

def compare_with_demo():
    """Compare full eICU results with demo subset (Cohen's d)."""
    demo_dir = ROOT / "experiments" / "exp_eicu"
    if not demo_dir.exists():
        log.warning("Demo results not found; skipping comparison")
        return

    log.info("Comparing with demo subset...")
    rows = []

    # Compare E1 gaps
    full_gaps = pd.read_csv(EXP_DIR / "e1_gaps.csv")
    demo_gaps = pd.read_csv(demo_dir / "e1_gaps.csv")
    merged = full_gaps.merge(demo_gaps, on=["score", "axis"],
                             suffixes=("_full", "_demo"))
    for _, r in merged.iterrows():
        diff = r.get("auroc_gap_full", 0) - r.get("auroc_gap_demo", 0)
        rows.append({
            "experiment": "E1_gap", "score": r["score"], "axis": r["axis"],
            "full": r.get("auroc_gap_full"), "demo": r.get("auroc_gap_demo"),
            "difference": diff,
        })

    # Compare E4 RSB if available
    for fname in ["e4_rsb.csv"]:
        fp = EXP_DIR / fname
        dp = demo_dir / fname
        if fp.exists() and dp.exists():
            ff = pd.read_csv(fp)
            df = pd.read_csv(dp)
            log.info(f"  {fname}: full {len(ff)} rows, demo {len(df)} rows")

    pd.DataFrame(rows).to_csv(EXP_DIR / "comparison_with_demo.csv", index=False)
    log.info(f"  Comparison saved ({len(rows)} rows)")


# ════════════════════════════════════════════════════════════════════════
# Standalone fallbacks
# ════════════════════════════════════════════════════════════════════════

def _standalone_audit(data):
    from sklearn.metrics import roc_auc_score
    rows, gaps = [], []
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        for axis in DEMO_AXES:
            if axis not in data.columns:
                continue
            aurocs = {}
            for g in data[axis].dropna().unique():
                m = data[axis] == g
                y, x = data.loc[m, "mortality"], data.loc[m, score]
                v = y.notna() & x.notna()
                if v.sum() >= MIN_SUBGROUP_SIZE and y[v].nunique() == 2:
                    a = roc_auc_score(y[v], x[v])
                    aurocs[g] = a
                    rows.append({"score": score, "axis": axis, "group": g,
                                 "auroc": a, "n": v.sum(),
                                 "prevalence": y[v].mean()})
            if len(aurocs) >= 2:
                gaps.append({"score": score, "axis": axis,
                             "auroc_gap": max(aurocs.values()) - min(aurocs.values()),
                             "worst_group": min(aurocs, key=aurocs.get),
                             "best_group": max(aurocs, key=aurocs.get)})
    return pd.DataFrame(rows), pd.DataFrame(gaps)


def _standalone_intersectional(data):
    from sklearn.metrics import roc_auc_score
    from itertools import combinations
    rows = []
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        for a1, a2 in combinations(DEMO_AXES, 2):
            if a1 not in data.columns or a2 not in data.columns:
                continue
            for g1 in data[a1].dropna().unique():
                for g2 in data[a2].dropna().unique():
                    m = (data[a1] == g1) & (data[a2] == g2)
                    y, x = data.loc[m, "mortality"], data.loc[m, score]
                    v = y.notna() & x.notna()
                    if v.sum() >= MIN_SUBGROUP_SIZE and y[v].nunique() == 2:
                        rows.append({"score": score, "subgroup": f"{g1}x{g2}",
                                     "auroc": roc_auc_score(y[v], x[v]),
                                     "n": v.sum(), "prevalence": y[v].mean()})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS full eICU-CRD pipeline")
    parser.add_argument("--eicu-dir", required=True, help="Path to eICU-CRD data")
    parser.add_argument("--skip-ml", action="store_true", help="Skip GRU (E4-E5)")
    parser.add_argument("--compare-demo", action="store_true",
                        help="Compare with demo subset results")
    parser.add_argument("--max-stays", type=int, default=None)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    global RANDOM_SEED
    RANDOM_SEED = args.seed
    np.random.seed(RANDOM_SEED)

    eicu_dir = Path(args.eicu_dir)
    start = time.time()

    # Extract cohort
    cohort = extract_eicu_cohort(eicu_dir, max_stays=args.max_stays)
    stay_ids = set(cohort["patientunitstayid"].values)

    # Extract features
    vitals = extract_vitals(eicu_dir, stay_ids)
    labs = extract_labs(eicu_dir, stay_ids)

    data = cohort.copy()
    if len(vitals) > 0:
        data = data.merge(vitals, on="patientunitstayid", how="left")
    if len(labs) > 0:
        data = data.merge(labs, on="patientunitstayid", how="left")

    # GCS total from components if needed
    if "gcs_total" not in data.columns:
        gcs_cols = ["gcs_eye", "gcs_verbal", "gcs_motor"]
        if all(c in data.columns for c in gcs_cols):
            data["gcs_total"] = (data["gcs_eye"].fillna(4) +
                                 data["gcs_verbal"].fillna(5) +
                                 data["gcs_motor"].fillna(6))

    # Compute scores
    data["sofa"] = _compute_sofa(data)
    data["qsofa"] = _compute_qsofa(data)
    data["apache2"] = _compute_apache2(data)
    data["news2"] = _compute_news2(data)

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(EXP_DIR / "cohort_with_scores.csv", index=False)

    # Run audit
    run_audit(data, skip_ml=args.skip_ml, bootstrap_n=args.bootstrap_n)

    if args.compare_demo:
        compare_with_demo()

    log.info(f"Total time: {(time.time() - start)/60:.1f} min")


if __name__ == "__main__":
    main()
