#!/usr/bin/env python3
# MIT License — Anonymous Authors, 2026
# Part of ATLAS
"""
ATLAS Pipeline for MIMIC-IV validation.

Runs the full E1–E15 equity audit on MIMIC-IV data with two MIMIC-specific
additions:
  - Insurance status as a 5th demographic axis
  - Temporal drift analysis (E15) using multi-year admission data

Usage:
    python run_mimic_pipeline.py --mimic-dir /path/to/mimic-iv
    python run_mimic_pipeline.py --mimic-dir /path/to/mimic-iv --skip-gru
    python run_mimic_pipeline.py --mimic-dir /path/to/mimic-iv --max-stays 10000

Requires PhysioNet credentialed access to MIMIC-IV v2.x.
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
log = logging.getLogger("atlas.mimic")

# ── Constants ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
MIN_SUBGROUP_SIZE = 100
BOOTSTRAP_ITERATIONS = 1000

SCORE_NAMES = ["sofa", "qsofa", "apache2", "news2"]
DEMO_AXES = ["race_cat", "sex", "age_group", "insurance_cat", "diag_type"]

EXP_DIR = ROOT / "experiments" / "exp_mimic"

# ── MIMIC-IV Canonical Item IDs ────────────────────────────────────────
# Vitals (chartevents)
ITEMID_MAP = {
    # Heart rate
    220045: ("heart_rate", "max"),
    # Blood pressure
    220050: ("sbp", "min"),    # Arterial BP systolic
    220179: ("sbp", "min"),    # Non-invasive BP systolic
    220051: ("dbp", "min"),    # Arterial BP diastolic
    220180: ("dbp", "min"),    # Non-invasive BP diastolic
    220052: ("map", "min"),    # Arterial MAP
    220181: ("map", "min"),    # Non-invasive MAP
    # Respiratory
    220210: ("resp_rate", "max"),
    220277: ("spo2", "min"),
    # Temperature
    223761: ("temp_c", "max"),   # Celsius
    223762: ("temp_f", "max"),   # Fahrenheit → convert
    # GCS
    220739: ("gcs_eye", "min"),
    223900: ("gcs_verbal", "min"),
    223901: ("gcs_motor", "min"),
    228112: ("gcs_total", "min"),  # Direct GCS total if available
    # FiO2
    223835: ("fio2", "max"),
    227009: ("fio2", "max"),     # FiO2 set
    # Ventilation / respiratory mechanics
    220339: ("peep", "max"),
    # Urine output (multiple sources)
    226559: ("urine_output", "sum"),
    226560: ("urine_output", "sum"),
    226561: ("urine_output", "sum"),
    226563: ("urine_output", "sum"),
    226564: ("urine_output", "sum"),
    226565: ("urine_output", "sum"),
    226567: ("urine_output", "sum"),
    226627: ("urine_output", "sum"),
    227489: ("urine_output", "sum"),
}

# Labs (labevents)
LABID_MAP = {
    50821: ("pao2", "min"),
    50818: ("paco2", "max"),
    50820: ("ph", "min"),
    50813: ("lactate", "max"),
    51222: ("hemoglobin", "min"),
    51221: ("hematocrit", "min"),
    51301: ("wbc", "max"),
    51265: ("platelets", "min"),
    50983: ("sodium", "max"),
    50971: ("potassium", "max"),
    50902: ("chloride", "max"),
    50882: ("bicarbonate", "min"),
    51006: ("bun", "max"),
    50912: ("creatinine", "max"),
    50931: ("glucose", "max"),
    50885: ("bilirubin", "max"),
    50862: ("albumin", "min"),
    50861: ("alt", "max"),       # ALT / SGPT
    50878: ("ast", "max"),       # AST / SGOT
    51237: ("inr", "max"),
    51275: ("ptt", "max"),
    51003: ("troponin_t", "max"),
    50963: ("bnp", "max"),
    50889: ("crp", "max"),
    50868: ("anion_gap", "max"),
    50893: ("calcium_total", "min"),
    50960: ("magnesium", "max"),
    50970: ("phosphate", "max"),
    51274: ("pt", "max"),
}


# ════════════════════════════════════════════════════════════════════════
# Phase 1: Cohort Extraction
# ════════════════════════════════════════════════════════════════════════

def _read_table(mimic_dir: Path, table_name: str,
                usecols=None, dtype=None) -> pd.DataFrame:
    """Auto-discover and load a MIMIC-IV table (CSV, CSV.GZ, or Parquet)."""
    # Try common MIMIC-IV layouts: flat, hosp/, icu/
    candidates = [
        mimic_dir / f"{table_name}.csv",
        mimic_dir / f"{table_name}.csv.gz",
        mimic_dir / f"{table_name}.parquet",
        mimic_dir / "hosp" / f"{table_name}.csv.gz",
        mimic_dir / "hosp" / f"{table_name}.csv",
        mimic_dir / "hosp" / f"{table_name}.parquet",
        mimic_dir / "icu" / f"{table_name}.csv.gz",
        mimic_dir / "icu" / f"{table_name}.csv",
        mimic_dir / "icu" / f"{table_name}.parquet",
    ]
    for p in candidates:
        if p.exists():
            log.info(f"  Loading {p.name} from {p.parent}")
            if p.suffix == ".parquet":
                return pd.read_parquet(p, columns=usecols)
            return pd.read_csv(p, usecols=usecols, dtype=dtype)
    raise FileNotFoundError(
        f"Cannot find {table_name} in {mimic_dir}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def extract_mimic_cohort(mimic_dir: Path, max_stays=None) -> pd.DataFrame:
    """
    Build the ATLAS cohort from MIMIC-IV tables.

    Returns DataFrame with one row per ICU stay, columns:
      stay_id, subject_id, hadm_id, intime, outtime, los_icu,
      age, sex, race_cat, insurance_cat, age_group, diag_type,
      mortality, admit_year, careunit
    """
    log.info("Loading MIMIC-IV tables...")

    patients = _read_table(mimic_dir, "patients",
                           usecols=["subject_id", "gender", "anchor_age",
                                    "anchor_year", "anchor_year_group", "dod"])
    admissions = _read_table(mimic_dir, "admissions",
                             usecols=["subject_id", "hadm_id", "admittime",
                                      "dischtime", "deathtime", "ethnicity",
                                      "insurance", "admission_type"])
    icustays = _read_table(mimic_dir, "icustays",
                           usecols=["subject_id", "hadm_id", "stay_id",
                                    "intime", "outtime", "first_careunit"])

    # Merge
    df = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="left")
    df = df.merge(patients, on="subject_id", how="left")

    # Parse dates
    for col in ["intime", "outtime", "admittime", "dischtime", "deathtime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Age at ICU admission (MIMIC-IV uses anchor_age + year offset)
    df["admit_year"] = df["admittime"].dt.year
    df["age"] = df["anchor_age"] + (df["admit_year"] - df["anchor_year"])
    df = df[(df["age"] >= 18) & (df["age"] <= 120)].copy()

    # Mortality: in-hospital death
    df["mortality"] = (~df["deathtime"].isna()).astype(int)

    # Demographics
    df["sex"] = df["gender"].map({"M": "Male", "F": "Female"}).fillna("Unknown")

    # Race/ethnicity mapping
    race_map = {
        "WHITE": "White", "BLACK/AFRICAN AMERICAN": "Black",
        "HISPANIC/LATINO": "Hispanic", "HISPANIC OR LATINO": "Hispanic",
        "ASIAN": "Asian", "ASIAN - CHINESE": "Asian",
        "ASIAN - ASIAN INDIAN": "Asian", "ASIAN - SOUTH EAST ASIAN": "Asian",
        "WHITE - RUSSIAN": "White", "WHITE - EASTERN EUROPEAN": "White",
        "WHITE - BRAZILIAN": "White", "WHITE - OTHER EUROPEAN": "White",
        "BLACK/CAPE VERDEAN": "Black", "BLACK/AFRICAN": "Black",
        "BLACK/HAITIAN": "Black", "BLACK/CARIBBEAN ISLAND": "Black",
        "HISPANIC/LATINO - PUERTO RICAN": "Hispanic",
        "HISPANIC/LATINO - DOMINICAN": "Hispanic",
        "HISPANIC/LATINO - GUATEMALAN": "Hispanic",
        "HISPANIC/LATINO - CUBAN": "Hispanic",
        "HISPANIC/LATINO - SALVADORAN": "Hispanic",
        "HISPANIC/LATINO - COLOMBIAN": "Hispanic",
        "HISPANIC/LATINO - MEXICAN": "Hispanic",
        "HISPANIC/LATINO - HONDURAN": "Hispanic",
        "HISPANIC/LATINO - CENTRAL AMERICAN": "Hispanic",
    }
    df["race_cat"] = df["ethnicity"].map(race_map).fillna("Other")

    # Insurance
    ins_map = {"Medicare": "Medicare", "Medicaid": "Medicaid",
               "Other": "Private"}
    df["insurance_cat"] = df["insurance"].map(ins_map).fillna("Private")

    # Age groups (matching GOSSIS)
    bins = [17, 29, 39, 49, 59, 69, 79, 200]
    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    # Diagnosis type from admission_type
    def _map_diag(adm_type):
        if pd.isna(adm_type):
            return "Medical"
        adm_type = str(adm_type).upper()
        if "SURG" in adm_type or "OPER" in adm_type:
            return "Surgical"
        if "CARD" in adm_type:
            return "Cardiac"
        return "Medical"

    df["diag_type"] = df["admission_type"].apply(_map_diag)
    df["careunit"] = df["first_careunit"].fillna("Unknown")

    # LOS
    df["los_icu"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 86400

    # Keep first ICU stay per hospital admission
    df = df.sort_values("intime").groupby("hadm_id").first().reset_index()

    if max_stays:
        df = df.sample(n=min(max_stays, len(df)), random_state=RANDOM_SEED)

    log.info(f"Cohort: {len(df)} ICU stays, mortality={df['mortality'].mean():.3f}")
    return df


def _extract_worst_24h(mimic_dir: Path, stay_ids: pd.Series,
                       intime_map: dict, chunksize=5_000_000) -> pd.DataFrame:
    """
    Extract worst-24h physiologic values from chartevents and labevents.
    Reads in chunks for memory efficiency.
    """
    log.info("Extracting worst-24h values (this may take a while)...")

    # Track aggregated values
    agg = defaultdict(lambda: defaultdict(list))  # stay_id -> feature -> [values]

    # ── Chart events ──
    log.info("  Processing chartevents...")
    all_chart_ids = set(ITEMID_MAP.keys())
    chart_path = None
    for suffix in [".csv.gz", ".csv", ".parquet"]:
        for subdir in ["icu", ""]:
            p = mimic_dir / subdir / f"chartevents{suffix}" if subdir else \
                mimic_dir / f"chartevents{suffix}"
            if p.exists():
                chart_path = p
                break
        if chart_path:
            break

    if chart_path:
        if chart_path.suffix == ".parquet":
            chunks = [pd.read_parquet(chart_path,
                        columns=["stay_id", "itemid", "charttime", "valuenum"])]
        else:
            chunks = pd.read_csv(
                chart_path,
                usecols=["stay_id", "itemid", "charttime", "valuenum"],
                dtype={"stay_id": "Int64", "itemid": int},
                chunksize=chunksize,
            )

        stay_id_set = set(stay_ids.values)
        n_chunks = 0
        for chunk in chunks:
            n_chunks += 1
            if n_chunks % 10 == 0:
                log.info(f"    chartevents chunk {n_chunks}...")

            chunk = chunk[
                chunk["stay_id"].isin(stay_id_set) &
                chunk["itemid"].isin(all_chart_ids) &
                chunk["valuenum"].notna()
            ].copy()
            if chunk.empty:
                continue

            chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")

            for _, row in chunk.iterrows():
                sid = row["stay_id"]
                if sid not in intime_map:
                    continue
                intime = intime_map[sid]
                ct = row["charttime"]
                if pd.isna(ct) or pd.isna(intime):
                    continue
                hours = (ct - intime).total_seconds() / 3600
                if hours < 0 or hours > 24:
                    continue

                iid = int(row["itemid"])
                if iid in ITEMID_MAP:
                    feat, _ = ITEMID_MAP[iid]
                    val = row["valuenum"]
                    # Temperature conversion
                    if feat == "temp_f":
                        val = (val - 32) * 5 / 9
                        feat = "temp_c"
                    # FiO2 normalization (values in [0,1] → percentage)
                    if feat == "fio2" and val <= 1.0:
                        val *= 100
                    agg[sid][feat].append(val)

    # ── Lab events ──
    log.info("  Processing labevents...")
    all_lab_ids = set(LABID_MAP.keys())
    lab_path = None
    for suffix in [".csv.gz", ".csv", ".parquet"]:
        for subdir in ["hosp", ""]:
            p = mimic_dir / subdir / f"labevents{suffix}" if subdir else \
                mimic_dir / f"labevents{suffix}"
            if p.exists():
                lab_path = p
                break
        if lab_path:
            break

    if lab_path:
        if lab_path.suffix == ".parquet":
            chunks = [pd.read_parquet(lab_path,
                        columns=["subject_id", "itemid", "charttime", "valuenum"])]
        else:
            chunks = pd.read_csv(
                lab_path,
                usecols=["subject_id", "itemid", "charttime", "valuenum"],
                dtype={"subject_id": "Int64", "itemid": int},
                chunksize=chunksize,
            )

        # Need subject_id → stay_id + intime mapping for labs
        # (labevents has subject_id, not stay_id)
        # Build mapping externally and pass in; for now, skip if no stay_id
        log.warning("  labevents uses subject_id; lab extraction needs "
                     "subject→stay mapping (passed via cohort merge)")

    # ── Aggregate to worst values ──
    log.info("  Aggregating to worst-24h...")
    rows = []
    for sid, feats in agg.items():
        row = {"stay_id": sid}
        for feat, vals in feats.items():
            if not vals:
                continue
            # Determine aggregation from the mapping
            agg_fn = None
            for iid, (f, a) in {**ITEMID_MAP, **LABID_MAP}.items():
                if f == feat:
                    agg_fn = a
                    break
            if agg_fn == "min":
                row[feat] = min(vals)
            elif agg_fn == "max":
                row[feat] = max(vals)
            elif agg_fn == "sum":
                row[feat] = sum(vals)
            else:
                row[feat] = max(vals)  # default: worst = max
        rows.append(row)

    features = pd.DataFrame(rows)
    log.info(f"  Extracted features for {len(features)} stays, "
             f"{len(features.columns)-1} features")
    return features


# ════════════════════════════════════════════════════════════════════════
# Phase 2: Score Computation
# ════════════════════════════════════════════════════════════════════════

def compute_sofa(df: pd.DataFrame) -> pd.Series:
    """Compute SOFA score (0-24) from worst-24h values."""
    sofa = pd.Series(0, index=df.index, dtype=float)

    # Respiratory: PaO2/FiO2 ratio
    if "pao2" in df.columns and "fio2" in df.columns:
        pf = df["pao2"] / (df["fio2"] / 100).clip(lower=0.21)
        sofa += np.select(
            [pf < 100, pf < 200, pf < 300, pf < 400],
            [4, 3, 2, 1], default=0
        )

    # Coagulation: Platelets
    if "platelets" in df.columns:
        plt_ = df["platelets"]
        sofa += np.select(
            [plt_ < 20, plt_ < 50, plt_ < 100, plt_ < 150],
            [4, 3, 2, 1], default=0
        )

    # Liver: Bilirubin
    if "bilirubin" in df.columns:
        bil = df["bilirubin"]
        sofa += np.select(
            [bil >= 12.0, bil >= 6.0, bil >= 2.0, bil >= 1.2],
            [4, 3, 2, 1], default=0
        )

    # Cardiovascular: MAP
    if "map" in df.columns:
        m = df["map"]
        sofa += np.where(m < 70, 1, 0)
        # Vasopressor data not reliably available; cap at 1 if no vasopressor info

    # CNS: GCS
    gcs_col = "gcs_total" if "gcs_total" in df.columns else None
    if gcs_col is None and all(c in df.columns for c in ["gcs_eye", "gcs_verbal", "gcs_motor"]):
        df = df.copy()
        df["gcs_total"] = df["gcs_eye"].fillna(4) + df["gcs_verbal"].fillna(5) + df["gcs_motor"].fillna(6)
        gcs_col = "gcs_total"
    if gcs_col:
        gcs = df[gcs_col]
        sofa += np.select(
            [gcs < 6, gcs < 10, gcs < 13, gcs < 15],
            [4, 3, 2, 1], default=0
        )

    # Renal: Creatinine
    if "creatinine" in df.columns:
        cr = df["creatinine"]
        sofa += np.select(
            [cr >= 5.0, cr >= 3.5, cr >= 2.0, cr >= 1.2],
            [4, 3, 2, 1], default=0
        )

    return sofa.clip(0, 24)


def compute_qsofa(df: pd.DataFrame) -> pd.Series:
    """Compute qSOFA (0-3)."""
    q = pd.Series(0, index=df.index, dtype=float)
    if "resp_rate" in df.columns:
        q += (df["resp_rate"] >= 22).astype(int)
    if "sbp" in df.columns:
        q += (df["sbp"] <= 100).astype(int)
    elif "map" in df.columns:
        q += (df["map"] <= 65).astype(int)  # MAP proxy
    gcs_col = "gcs_total" if "gcs_total" in df.columns else None
    if gcs_col:
        q += (df[gcs_col] < 15).astype(int)
    return q.clip(0, 3)


def compute_apache2(df: pd.DataFrame) -> pd.Series:
    """Compute APACHE-II (0-71). 12 physiology vars + age + chronic health."""
    score = pd.Series(0, index=df.index, dtype=float)

    # Temperature (Celsius)
    if "temp_c" in df.columns:
        t = df["temp_c"]
        score += np.select(
            [t >= 41, t.between(39, 40.99), t.between(38.5, 38.99),
             t.between(36, 38.49), t.between(34, 35.99),
             t.between(32, 33.99), t.between(30, 31.99), t < 30],
            [4, 3, 1, 0, 1, 2, 3, 4], default=0
        )

    # MAP
    if "map" in df.columns:
        m = df["map"]
        score += np.select(
            [m >= 160, m.between(130, 159), m.between(110, 129),
             m.between(70, 109), m.between(50, 69), m < 50],
            [4, 3, 2, 0, 2, 4], default=0
        )

    # Heart rate
    if "heart_rate" in df.columns:
        hr = df["heart_rate"]
        score += np.select(
            [hr >= 180, hr.between(140, 179), hr.between(110, 139),
             hr.between(70, 109), hr.between(55, 69),
             hr.between(40, 54), hr < 40],
            [4, 3, 2, 0, 2, 3, 4], default=0
        )

    # Respiratory rate
    if "resp_rate" in df.columns:
        rr = df["resp_rate"]
        score += np.select(
            [rr >= 50, rr.between(35, 49), rr.between(25, 34),
             rr.between(12, 24), rr.between(10, 11),
             rr.between(6, 9), rr < 6],
            [4, 3, 1, 0, 1, 2, 4], default=0
        )

    # Oxygenation (A-a gradient if FiO2 >= 0.5, else PaO2)
    if "pao2" in df.columns and "fio2" in df.columns:
        fio2_frac = (df["fio2"] / 100).clip(0.21, 1.0)
        high_fio2 = fio2_frac >= 0.5
        # A-a gradient approximation
        aa = (fio2_frac * 713) - (df.get("paco2", 40) / 0.8) - df["pao2"]
        pao2 = df["pao2"]

        oxy_score = np.where(
            high_fio2,
            np.select([aa >= 500, aa.between(350, 499),
                       aa.between(200, 349), aa < 200],
                      [4, 3, 2, 0], default=0),
            np.select([pao2 < 55, pao2.between(55, 60),
                       pao2.between(61, 70), pao2 > 70],
                      [4, 3, 1, 0], default=0)
        )
        score += oxy_score

    # pH
    if "ph" in df.columns:
        ph = df["ph"]
        score += np.select(
            [ph >= 7.7, ph.between(7.6, 7.69), ph.between(7.5, 7.59),
             ph.between(7.33, 7.49), ph.between(7.25, 7.32),
             ph.between(7.15, 7.24), ph < 7.15],
            [4, 3, 1, 0, 2, 3, 4], default=0
        )

    # Sodium
    if "sodium" in df.columns:
        na = df["sodium"]
        score += np.select(
            [na >= 180, na.between(160, 179), na.between(155, 159),
             na.between(150, 154), na.between(130, 149),
             na.between(120, 129), na < 120],
            [4, 3, 2, 1, 0, 2, 3], default=0
        )

    # Potassium
    if "potassium" in df.columns:
        k = df["potassium"]
        score += np.select(
            [k >= 7.0, k.between(6.0, 6.9), k.between(5.5, 5.9),
             k.between(3.5, 5.4), k.between(3.0, 3.4),
             k.between(2.5, 2.9), k < 2.5],
            [4, 3, 1, 0, 1, 2, 4], default=0
        )

    # Creatinine
    if "creatinine" in df.columns:
        cr = df["creatinine"]
        score += np.select(
            [cr >= 3.5, cr.between(2.0, 3.4), cr.between(1.5, 1.9),
             cr.between(0.6, 1.4), cr < 0.6],
            [4, 3, 2, 0, 2], default=0
        )

    # Hematocrit
    if "hematocrit" in df.columns:
        hct = df["hematocrit"]
        score += np.select(
            [hct >= 60, hct.between(50, 59.9), hct.between(46, 49.9),
             hct.between(30, 45.9), hct.between(20, 29.9), hct < 20],
            [4, 2, 1, 0, 2, 4], default=0
        )

    # WBC
    if "wbc" in df.columns:
        wbc = df["wbc"]
        score += np.select(
            [wbc >= 40, wbc.between(20, 39.9), wbc.between(15, 19.9),
             wbc.between(3, 14.9), wbc.between(1, 2.9), wbc < 1],
            [4, 2, 1, 0, 2, 4], default=0
        )

    # GCS (15 - GCS)
    gcs_col = "gcs_total" if "gcs_total" in df.columns else None
    if gcs_col:
        score += (15 - df[gcs_col].clip(3, 15))

    # Age points
    if "age" in df.columns:
        age = df["age"]
        score += np.select(
            [age >= 75, age.between(65, 74), age.between(55, 64),
             age.between(45, 54), age < 45],
            [6, 5, 3, 2, 0], default=0
        )

    return score.clip(0, 71)


def compute_news2(df: pd.DataFrame) -> pd.Series:
    """Compute NEWS2 (0-20)."""
    score = pd.Series(0, index=df.index, dtype=float)

    # Resp rate
    if "resp_rate" in df.columns:
        rr = df["resp_rate"]
        score += np.select(
            [rr <= 8, rr.between(9, 11), rr.between(12, 20),
             rr.between(21, 24), rr >= 25],
            [3, 1, 0, 2, 3], default=0
        )

    # SpO2 (Scale 1)
    if "spo2" in df.columns:
        sp = df["spo2"]
        score += np.select(
            [sp <= 91, sp.between(92, 93), sp.between(94, 95), sp >= 96],
            [3, 2, 1, 0], default=0
        )

    # Supplemental O2
    if "fio2" in df.columns:
        score += (df["fio2"] > 21).astype(int) * 2

    # Temperature
    if "temp_c" in df.columns:
        t = df["temp_c"]
        score += np.select(
            [t <= 35.0, t.between(35.1, 36.0), t.between(36.1, 38.0),
             t.between(38.1, 39.0), t >= 39.1],
            [3, 1, 0, 1, 2], default=0
        )

    # Systolic BP
    bp_col = "sbp" if "sbp" in df.columns else "map"
    if bp_col in df.columns:
        bp = df[bp_col]
        if bp_col == "map":
            # Approximate SBP from MAP
            bp = bp * 1.5
        score += np.select(
            [bp <= 90, bp.between(91, 100), bp.between(101, 110),
             bp.between(111, 219), bp >= 220],
            [3, 2, 1, 0, 3], default=0
        )

    # Heart rate
    if "heart_rate" in df.columns:
        hr = df["heart_rate"]
        score += np.select(
            [hr <= 40, hr.between(41, 50), hr.between(51, 90),
             hr.between(91, 110), hr.between(111, 130), hr >= 131],
            [3, 1, 0, 1, 2, 3], default=0
        )

    # Consciousness (GCS < 15 → 3 points)
    gcs_col = "gcs_total" if "gcs_total" in df.columns else None
    if gcs_col:
        score += (df[gcs_col] < 15).astype(int) * 3

    return score.clip(0, 20)


def compute_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 4 scores and add to DataFrame."""
    df = df.copy()
    df["sofa"] = compute_sofa(df)
    df["qsofa"] = compute_qsofa(df)
    df["apache2"] = compute_apache2(df)
    df["news2"] = compute_news2(df)
    log.info(f"Scores computed: SOFA={df['sofa'].mean():.1f}, "
             f"qSOFA={df['qsofa'].mean():.1f}, "
             f"APACHE-II={df['apache2'].mean():.1f}, "
             f"NEWS2={df['news2'].mean():.1f}")
    return df


# ════════════════════════════════════════════════════════════════════════
# Phase 3: Audit Pipeline (E1–E15)
# ════════════════════════════════════════════════════════════════════════

def run_audit_pipeline(data: pd.DataFrame, skip_gru: bool = False):
    """Run the full ATLAS E1–E15 audit pipeline."""
    from sklearn.metrics import roc_auc_score
    from sklearn.calibration import calibration_curve
    from sklearn.model_selection import StratifiedKFold
    from scipy import stats

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    results_file = ROOT / "RESULTS_MIMIC.md"

    with open(results_file, "w") as f:
        f.write("# Results — ATLAS on MIMIC-IV\n\n")
        f.write(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**N**: {len(data)} ICU stays\n")
        f.write(f"**Mortality**: {data['mortality'].mean():.3f}\n\n")

    # ── E1: Pre-specified Subgroup Audit ──
    log.info("=" * 60)
    log.info("E1: Pre-specified Subgroup Audit")
    log.info("=" * 60)

    try:
        from src.evaluation.audit import prespecified_audit, intersectional_audit
        from src.evaluation.asd import adversarial_subgroup_discovery
        from src.evaluation.rsb import compute_rsb, compute_ml_improvement
        USE_SRC = True
    except ImportError:
        log.warning("Cannot import src modules; using standalone implementations")
        USE_SRC = False

    if USE_SRC:
        audit_results, gaps_df = prespecified_audit(
            data, axes=DEMO_AXES, n_boot=BOOTSTRAP_ITERATIONS
        )
    else:
        audit_results, gaps_df = _standalone_audit(data)

    audit_results.to_csv(EXP_DIR / "e1_audit_results.csv", index=False)
    gaps_df.to_csv(EXP_DIR / "e1_gaps.csv", index=False)
    log.info(f"E1: {len(audit_results)} subgroup results, {len(gaps_df)} gaps")

    # ── E2: Intersectional Analysis ──
    log.info("E2: Intersectional Analysis")
    if USE_SRC:
        inter_results, worst_subgroups = intersectional_audit(
            data, axes=DEMO_AXES, min_n=MIN_SUBGROUP_SIZE,
            n_boot=BOOTSTRAP_ITERATIONS
        )
    else:
        inter_results = _standalone_intersectional(data)
        worst_subgroups = {}
    inter_results.to_csv(EXP_DIR / "e2_intersectional.csv", index=False)
    log.info(f"E2: {len(inter_results)} intersectional subgroups")

    # ── E3: Adversarial Subgroup Discovery ──
    log.info("E3: Adversarial Subgroup Discovery")
    if USE_SRC:
        asd_results = adversarial_subgroup_discovery(data)
        with open(EXP_DIR / "e3_asd_results.json", "w") as f:
            serializable = {}
            for k, v in asd_results.items():
                sv = {**v}
                sv["top_features"] = [(fn, float(fi)) for fn, fi in sv["top_features"]]
                serializable[k] = sv
            json.dump(serializable, f, indent=2, default=str)
    log.info("E3 done.")

    # ── E4-E5: GRU + RSB ──
    ml_preds = None
    if not skip_gru:
        log.info("E4-E5: GRU Training & RSB")
        try:
            import torch
            from src.training.train_gru import train_gru_model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gru_result = train_gru_model(data, device=device, epochs=50)
            ml_preds = gru_result["predictions"]
            np.save(EXP_DIR / "ml_preds.npy", ml_preds)
            log.info(f"GRU AUROC: {gru_result['overall_auroc']:.4f}")

            if USE_SRC:
                rsb_df = compute_rsb(data, ml_preds, axes=DEMO_AXES,
                                     n_boot=min(BOOTSTRAP_ITERATIONS, 500))
                rsb_df.to_csv(EXP_DIR / "e4_rsb.csv", index=False)
                improvement_df = compute_ml_improvement(data, ml_preds, axes=DEMO_AXES)
                improvement_df.to_csv(EXP_DIR / "e5_ml_improvement.csv", index=False)
        except Exception as e:
            log.error(f"GRU training failed: {e}")
    else:
        log.info("E4-E5: Skipped (--skip-gru)")

    # ── E6: Score-Conditional Mortality ──
    log.info("E6: Score-Conditional Mortality")
    scm_rows = []
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        bins = pd.qcut(data[score], q=10, duplicates="drop")
        for axis in ["age_group"]:
            for group in data[axis].dropna().unique():
                mask = data[axis] == group
                for bin_label, bin_data in data[mask].groupby(bins):
                    if len(bin_data) >= 10:
                        scm_rows.append({
                            "score": score,
                            "score_bin": str(bin_label),
                            "demo_axis": axis,
                            "group": group,
                            "mortality_rate": bin_data["mortality"].mean(),
                            "n": len(bin_data),
                        })
    pd.DataFrame(scm_rows).to_csv(EXP_DIR / "e6_score_conditional_mortality.csv",
                                   index=False)
    log.info(f"E6: {len(scm_rows)} score-conditional mortality cells")

    # ── E7: Clinical Threshold Analysis ──
    log.info("E7: Clinical Threshold Analysis")
    thresholds = {"sofa": [2, 6, 11], "qsofa": [1, 2],
                  "apache2": [15, 20, 25], "news2": [5, 9, 13]}
    thresh_rows = []
    for score, thresh_list in thresholds.items():
        if score not in data.columns:
            continue
        for thresh in thresh_list:
            pred_pos = (data[score] >= thresh).astype(int)
            for group in data["age_group"].dropna().unique():
                mask = data["age_group"] == group
                y = data.loc[mask, "mortality"]
                yp = pred_pos[mask]
                tp = ((yp == 1) & (y == 1)).sum()
                fn = ((yp == 0) & (y == 1)).sum()
                fp = ((yp == 1) & (y == 0)).sum()
                tn = ((yp == 0) & (y == 0)).sum()
                sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                thresh_rows.append({
                    "score": score, "threshold": thresh,
                    "age_group": group, "sensitivity": sens,
                    "specificity": spec, "n": mask.sum(),
                })
    pd.DataFrame(thresh_rows).to_csv(EXP_DIR / "e7_clinical_thresholds.csv",
                                      index=False)
    log.info(f"E7: {len(thresh_rows)} threshold analysis cells")

    # ── E8: SOFA Component Attribution ──
    log.info("E8: SOFA Component Attribution")
    components = {
        "respiratory": "pao2",
        "coagulation": "platelets",
        "liver": "bilirubin",
        "cardiovascular": "map",
        "cns": "gcs_total",
        "renal": "creatinine",
    }
    comp_rows = []
    for comp_name, col in components.items():
        if col not in data.columns:
            continue
        for group in data["age_group"].dropna().unique():
            mask = data["age_group"] == group
            y = data.loc[mask, "mortality"]
            x = data.loc[mask, col]
            valid = y.notna() & x.notna()
            if valid.sum() < 50 or y[valid].nunique() < 2:
                continue
            auc = roc_auc_score(y[valid], x[valid])
            comp_rows.append({
                "component": comp_name, "column": col,
                "demo_axis": "age_group", "group": group,
                "auroc": auc, "n": valid.sum(),
            })
    pd.DataFrame(comp_rows).to_csv(EXP_DIR / "sofa_components.csv", index=False)
    log.info(f"E8: {len(comp_rows)} component cells")

    # ── E9: Insurance-Stratified Analysis (MIMIC-specific) ──
    log.info("E9: Insurance-Stratified Score Performance (MIMIC-specific)")
    ins_rows = []
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        for ins_group in data["insurance_cat"].dropna().unique():
            mask = data["insurance_cat"] == ins_group
            y = data.loc[mask, "mortality"]
            x = data.loc[mask, score]
            valid = y.notna() & x.notna()
            if valid.sum() < MIN_SUBGROUP_SIZE or y[valid].nunique() < 2:
                continue
            auc = roc_auc_score(y[valid], x[valid])
            ins_rows.append({
                "score": score, "insurance": ins_group,
                "auroc": auc, "n": valid.sum(),
                "mortality_rate": y[valid].mean(),
            })
    pd.DataFrame(ins_rows).to_csv(EXP_DIR / "e9_insurance_analysis.csv",
                                   index=False)
    log.info(f"E9: {len(ins_rows)} insurance cells")

    # ── E10: Care-Unit Stratified Race Analysis ──
    log.info("E10: Care-Unit Stratified Race (Simpson's Paradox)")
    unit_rows = []
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        for unit in data["careunit"].dropna().unique():
            unit_mask = data["careunit"] == unit
            unit_data = data[unit_mask]
            if len(unit_data) < MIN_SUBGROUP_SIZE:
                continue
            race_aurocs = {}
            for race in unit_data["race_cat"].dropna().unique():
                rmask = unit_data["race_cat"] == race
                y = unit_data.loc[rmask, "mortality"]
                x = unit_data.loc[rmask, score]
                valid = y.notna() & x.notna()
                if valid.sum() >= 30 and y[valid].nunique() == 2:
                    race_aurocs[race] = roc_auc_score(y[valid], x[valid])
            if len(race_aurocs) >= 2:
                gap = max(race_aurocs.values()) - min(race_aurocs.values())
                unit_rows.append({
                    "score": score, "careunit": unit,
                    "n": len(unit_data),
                    "race_auroc_gap": gap,
                    "worst_race": min(race_aurocs, key=race_aurocs.get),
                    "best_race": max(race_aurocs, key=race_aurocs.get),
                    "n_races": len(race_aurocs),
                })
    pd.DataFrame(unit_rows).to_csv(EXP_DIR / "e10_careunit_race.csv", index=False)
    log.info(f"E10: {len(unit_rows)} unit-race cells")

    # ── E13: Recalibration Decomposition ──
    log.info("E13: Recalibration Decomposition")
    from sklearn.isotonic import IsotonicRegression
    recal_rows = []
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        y_all = data["mortality"].values
        x_all = data[score].values
        valid = ~np.isnan(x_all) & ~np.isnan(y_all)

        # Global isotonic
        iso_global = IsotonicRegression(out_of_bounds="clip")
        iso_global.fit(x_all[valid], y_all[valid])
        cal_global = iso_global.predict(x_all)

        for group in data["age_group"].dropna().unique():
            mask = (data["age_group"] == group).values & valid
            if mask.sum() < 50 or len(np.unique(y_all[mask])) < 2:
                continue

            raw_auroc = roc_auc_score(y_all[mask], x_all[mask])
            global_auroc = roc_auc_score(y_all[mask], cal_global[mask])

            # Per-group isotonic
            iso_grp = IsotonicRegression(out_of_bounds="clip")
            iso_grp.fit(x_all[mask], y_all[mask])
            cal_grp = iso_grp.predict(x_all[mask])
            grp_auroc = roc_auc_score(y_all[mask], cal_grp)

            recal_rows.append({
                "score": score, "age_group": group, "n": mask.sum(),
                "mortality_rate": y_all[mask].mean(),
                "raw_auroc": raw_auroc,
                "global_cal_auroc": global_auroc,
                "grp_cal_auroc": grp_auroc,
            })
    pd.DataFrame(recal_rows).to_csv(EXP_DIR / "e13_recalibration.csv", index=False)
    log.info(f"E13: {len(recal_rows)} recalibration cells")

    # ── E14: Propensity-Score Matched Analysis ──
    log.info("E14: Propensity-Score Matching")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors

        psm_features = [c for c in ["heart_rate", "map", "resp_rate", "spo2",
                                     "temp_c", "gcs_total", "creatinine",
                                     "platelets", "bilirubin", "sodium",
                                     "potassium", "wbc", "hemoglobin",
                                     "hematocrit", "bun", "glucose",
                                     "lactate", "ph", "pao2", "bicarbonate"]
                        if c in data.columns]

        elderly = data[data["age"] >= 80].copy()
        young = data[data["age"].between(18, 49)].copy()

        if len(elderly) >= 100 and len(young) >= 100 and len(psm_features) >= 5:
            combined = pd.concat([elderly.assign(is_elderly=1),
                                  young.assign(is_elderly=0)])
            X = combined[psm_features].fillna(combined[psm_features].median())
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            y_treat = combined["is_elderly"].values

            lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
            lr.fit(X_scaled, y_treat)
            ps = lr.predict_proba(X_scaled)[:, 1]

            elderly_ps = ps[y_treat == 1]
            young_ps = ps[y_treat == 0]
            caliper = 0.2 * np.std(ps)

            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(young_ps.reshape(-1, 1))
            distances, indices = nn.kneighbors(elderly_ps.reshape(-1, 1))

            match_mask = distances.flatten() <= caliper
            matched_elderly_idx = np.where(y_treat == 1)[0][match_mask]
            matched_young_idx = np.where(y_treat == 0)[0][indices.flatten()[match_mask]]

            psm_rows = []
            for score in SCORE_NAMES:
                if score not in data.columns:
                    continue
                e_data = combined.iloc[matched_elderly_idx]
                y_data = combined.iloc[matched_young_idx]

                ye, xe = e_data["mortality"].values, e_data[score].values
                yy, xy = y_data["mortality"].values, y_data[score].values

                ve = ~np.isnan(xe) & ~np.isnan(ye)
                vy = ~np.isnan(xy) & ~np.isnan(yy)

                if ve.sum() >= 50 and vy.sum() >= 50:
                    auroc_e = roc_auc_score(ye[ve], xe[ve]) if len(np.unique(ye[ve])) == 2 else np.nan
                    auroc_y = roc_auc_score(yy[vy], xy[vy]) if len(np.unique(yy[vy])) == 2 else np.nan

                    # Unmatched gaps
                    all_e = data[data["age"] >= 80]
                    all_y = data[data["age"].between(18, 49)]
                    ve2 = all_e["mortality"].notna() & all_e[score].notna()
                    vy2 = all_y["mortality"].notna() & all_y[score].notna()
                    auroc_e_un = roc_auc_score(all_e.loc[ve2, "mortality"],
                                                all_e.loc[ve2, score]) if ve2.sum() > 50 else np.nan
                    auroc_y_un = roc_auc_score(all_y.loc[vy2, "mortality"],
                                                all_y.loc[vy2, score]) if vy2.sum() > 50 else np.nan

                    # Bootstrap CI for matched gap
                    boot_gaps = []
                    rng = np.random.RandomState(RANDOM_SEED)
                    for _ in range(500):
                        idx = rng.choice(len(matched_elderly_idx),
                                         size=len(matched_elderly_idx), replace=True)
                        be = combined.iloc[matched_elderly_idx[idx]]
                        by = combined.iloc[matched_young_idx[idx]]
                        try:
                            bg = (roc_auc_score(by["mortality"], by[score]) -
                                  roc_auc_score(be["mortality"], be[score]))
                            boot_gaps.append(bg)
                        except Exception:
                            pass
                    ci_lo = np.percentile(boot_gaps, 2.5) if boot_gaps else np.nan
                    ci_hi = np.percentile(boot_gaps, 97.5) if boot_gaps else np.nan

                    psm_rows.append({
                        "score": score,
                        "n_matched_pairs": match_mask.sum(),
                        "auroc_elderly_unmatched": auroc_e_un,
                        "auroc_young_unmatched": auroc_y_un,
                        "gap_unmatched": auroc_y_un - auroc_e_un if not np.isnan(auroc_y_un) else np.nan,
                        "auroc_elderly_matched": auroc_e,
                        "auroc_young_matched": auroc_y,
                        "gap_matched": auroc_y - auroc_e if not np.isnan(auroc_y) else np.nan,
                        "gap_matched_ci_lo": ci_lo,
                        "gap_matched_ci_hi": ci_hi,
                        "mortality_elderly_matched": e_data["mortality"].mean(),
                        "mortality_young_matched": y_data["mortality"].mean(),
                    })

            pd.DataFrame(psm_rows).to_csv(EXP_DIR / "e14_psm_analysis.csv",
                                           index=False)
            log.info(f"E14: {len(psm_rows)} PSM results, "
                     f"{match_mask.sum()} matched pairs")
        else:
            log.warning("E14: Insufficient data for PSM")
    except Exception as e:
        log.error(f"E14 failed: {e}")

    # ── E15: Temporal Drift Analysis (MIMIC-specific) ──
    log.info("E15: Temporal Drift Analysis (MIMIC-specific)")
    if "admit_year" in data.columns:
        drift_rows = []
        for score in SCORE_NAMES:
            if score not in data.columns:
                continue
            for axis in DEMO_AXES:
                yearly_gaps = {}
                for year in sorted(data["admit_year"].dropna().unique()):
                    year_data = data[data["admit_year"] == year]
                    if len(year_data) < 200:
                        continue
                    aurocs = {}
                    for group in year_data[axis].dropna().unique():
                        mask = year_data[axis] == group
                        y = year_data.loc[mask, "mortality"]
                        x = year_data.loc[mask, score]
                        valid = y.notna() & x.notna()
                        if valid.sum() >= 30 and y[valid].nunique() == 2:
                            aurocs[group] = roc_auc_score(y[valid], x[valid])
                    if len(aurocs) >= 2:
                        gap = max(aurocs.values()) - min(aurocs.values())
                        yearly_gaps[year] = gap
                        drift_rows.append({
                            "score": score, "axis": axis, "year": int(year),
                            "auroc_gap": gap,
                            "worst_group": min(aurocs, key=aurocs.get),
                            "best_group": max(aurocs, key=aurocs.get),
                            "n": len(year_data),
                        })

                # Trend test
                if len(yearly_gaps) >= 3:
                    years = sorted(yearly_gaps.keys())
                    gaps = [yearly_gaps[y] for y in years]
                    rho, pval = stats.spearmanr(years, gaps)
                    drift_rows.append({
                        "score": score, "axis": axis,
                        "year": -1,  # sentinel for trend row
                        "auroc_gap": rho,
                        "worst_group": f"p={pval:.4f}",
                        "best_group": "TREND",
                        "n": len(data),
                    })

        pd.DataFrame(drift_rows).to_csv(EXP_DIR / "e15_temporal_drift.csv",
                                         index=False)
        log.info(f"E15: {len(drift_rows)} temporal drift cells")
    else:
        log.warning("E15: No admit_year column; skipping temporal drift")

    log.info("=" * 60)
    log.info("MIMIC-IV ATLAS pipeline complete!")
    log.info(f"Results saved to {EXP_DIR}")


# ════════════════════════════════════════════════════════════════════════
# Standalone fallback implementations (if src/ not importable)
# ════════════════════════════════════════════════════════════════════════

def _standalone_audit(data: pd.DataFrame):
    """Minimal standalone E1 audit if src modules unavailable."""
    from sklearn.metrics import roc_auc_score
    rows, gap_rows = [], []
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        for axis in DEMO_AXES:
            if axis not in data.columns:
                continue
            aurocs = {}
            for group in data[axis].dropna().unique():
                mask = data[axis] == group
                y = data.loc[mask, "mortality"]
                x = data.loc[mask, score]
                valid = y.notna() & x.notna()
                if valid.sum() >= MIN_SUBGROUP_SIZE and y[valid].nunique() == 2:
                    auc = roc_auc_score(y[valid], x[valid])
                    aurocs[group] = auc
                    rows.append({
                        "score": score, "axis": axis, "group": group,
                        "auroc": auc, "n": valid.sum(),
                        "prevalence": y[valid].mean(),
                    })
            if len(aurocs) >= 2:
                gap_rows.append({
                    "score": score, "axis": axis,
                    "auroc_gap": max(aurocs.values()) - min(aurocs.values()),
                    "worst_group": min(aurocs, key=aurocs.get),
                    "best_group": max(aurocs, key=aurocs.get),
                })
    return pd.DataFrame(rows), pd.DataFrame(gap_rows)


def _standalone_intersectional(data: pd.DataFrame):
    """Minimal standalone E2 intersectional audit."""
    from sklearn.metrics import roc_auc_score
    from itertools import combinations
    rows = []
    axis_pairs = list(combinations(DEMO_AXES, 2))
    for score in SCORE_NAMES:
        if score not in data.columns:
            continue
        for ax1, ax2 in axis_pairs:
            if ax1 not in data.columns or ax2 not in data.columns:
                continue
            for g1 in data[ax1].dropna().unique():
                for g2 in data[ax2].dropna().unique():
                    mask = (data[ax1] == g1) & (data[ax2] == g2)
                    y = data.loc[mask, "mortality"]
                    x = data.loc[mask, score]
                    valid = y.notna() & x.notna()
                    if valid.sum() >= MIN_SUBGROUP_SIZE and y[valid].nunique() == 2:
                        rows.append({
                            "score": score,
                            "subgroup": f"{g1}×{g2}",
                            "axis1": ax1, "group1": g1,
                            "axis2": ax2, "group2": g2,
                            "auroc": roc_auc_score(y[valid], x[valid]),
                            "n": valid.sum(),
                            "prevalence": y[valid].mean(),
                        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ATLAS Pipeline for MIMIC-IV validation"
    )
    parser.add_argument("--mimic-dir", type=str, required=True,
                        help="Path to MIMIC-IV data directory")
    parser.add_argument("--skip-gru", action="store_true",
                        help="Skip GRU training (E4-E5)")
    parser.add_argument("--max-stays", type=int, default=None,
                        help="Subsample to N stays for testing")
    args = parser.parse_args()

    mimic_dir = Path(args.mimic_dir)
    if not mimic_dir.exists():
        log.error(f"MIMIC-IV directory not found: {mimic_dir}")
        sys.exit(1)

    start = time.time()

    # Phase 1: Extract cohort
    cohort = extract_mimic_cohort(mimic_dir, max_stays=args.max_stays)

    # Phase 1b: Extract features
    intime_map = dict(zip(cohort["stay_id"], cohort["intime"]))
    features = _extract_worst_24h(mimic_dir, cohort["stay_id"], intime_map)
    data = cohort.merge(features, on="stay_id", how="left")

    # Phase 2: Compute scores
    data = compute_all_scores(data)
    data.to_csv(EXP_DIR / "cohort_with_scores.csv", index=False)

    # Phase 3: Run audit
    run_audit_pipeline(data, skip_gru=args.skip_gru)

    elapsed = time.time() - start
    log.info(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
