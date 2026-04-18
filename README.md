# Structural Age Bias in ICU Acuity Scores

**Mechanistic Decompositions and Fairness-Aware Predictive Models for Equitable Critical Care**

Anonymous Authors

---

## Overview

Clinical acuity scores---SOFA, qSOFA, APACHE-II, and NEWS2---guide millions of ICU triage decisions daily. This repository contains the full research codebase for a systematic equity audit of these four scores across 87,315 ICU stays from 147 hospitals in six countries (WiDS/GOSSIS), with independent external validation on eICU-CRD.

The audit demonstrates that age is the dominant disparity axis by a wide margin, that the gap is intrinsic to score design rather than physiologic population differences, and that using biased scores as fairness benchmarks for machine learning models distorts assessments by up to 12 percentage points. We propose two novel fairness-aware predictive models---a Fairness-Aware Feature Transformer and a group-adaptive extension with per-group pairwise discrimination ranking---that reduce the age discrimination gap 26--27% versus SOFA.

---

## Key Findings

**Structural age bias.** SOFA's discrimination gap between patients aged 18--29 and 80+ is 0.129; APACHE-II's is 0.145. The same SOFA score of 2--3 corresponds to 0.7% mortality in 18--29 year-olds versus 11.1% in 80+ (a 15-fold difference). SOFA ≥ 6 misses more than 1 in 4 dying elderly patients (43% sensitivity vs. 67% in young adults).

**Mechanistic proof of design bias.** Two independent analyses confirm the gap is intrinsic, not physiologic. Isotonic recalibration eliminates calibration gaps but cannot change rank ordering---the discrimination gap is invariant. Propensity-score matching on 20 vital-sign features causes the gap to widen (SOFA: 0.129 → 0.119 post-match), ruling out physiologic population differences.

**Simpson's paradox in race.** Within-hospital race discrimination gaps (0.186--0.206) are 6--10× larger than the aggregate gaps (0.019--0.033), driven by concentration of racial minority patients in lower-performing hospitals.

**Reference standard bias.** Benchmarking machine learning models against biased classical scores distorts fairness assessments by up to 12 percentage points for equalized odds (age-axis: 0.373). A model that "matches SOFA fairness" is in fact 12 points worse against true outcomes.

**Proposed models.** The Fairness-Aware Feature Transformer (0.884 discrimination, 113K parameters) combines transformer self-attention over clinical feature tokens with gradient-reversal adversarial debiasing. Its group-adaptive extension introduces the first per-group pairwise discrimination ranking loss for tabular ICU prediction, reducing the age discrimination gap to 0.094±0.003 (27% reduction vs. SOFA's 0.129).

---

## Datasets

| Dataset | N stays | Hospitals | Countries | Mortality | Role |
|---------|---------|-----------|-----------|-----------|------|
| WiDS/GOSSIS | 87,315 | 147 | 6 | 8.3% | Primary |
| eICU-CRD | 1,238 | 186 | 1 | 9.5% | External validation |

---

## Repository Structure

```
├── paper/                         # LaTeX manuscript (MLHC 2026)
│   ├── main.tex                   # Main document
│   ├── references.bib             # Verified bibliography
│   ├── sections/                  # Introduction, methods, results, etc.
│   └── figures/                   # Publication figures (PDF + PNG)
├── src/
│   ├── evaluation/
│   │   ├── audit.py               # Core equity audit (E1-E2, E6-E8)
│   │   ├── asd.py                 # Adversarial subgroup discovery (E3)
│   │   ├── rsb.py                 # Reference standard bias (E4)
│   │   └── figures.py             # Figure generation
│   ├── models/
│   │   └── gru.py                 # GRU mortality prediction model
│   └── training/
│       └── train_gru.py           # GRU training with 5-fold CV
├── experiments/
│   ├── exp_gossis/                # Primary GOSSIS results (CSV)
│   └── exp_eicu/                  # eICU-CRD validation results (CSV)
├── improve_figures.py             # Publication figure generation
├── generate_missing_figures.py    # Supplementary figure generation
├── run_pipeline.py                # Main audit pipeline (E1-E3)
├── run_gossis_pipeline.py         # GOSSIS E1-E5 pipeline
├── run_gossis_e4e5.py             # GRU training on GOSSIS (E4-E5)
├── run_expansion.py               # Mechanistic analyses (E6-E8)
├── run_full_gru.py                # Full-cohort GRU and FAFT training
├── LICENSE
└── README.md
```

---

## Analyses

| ID | Analysis | Key Result |
|----|----------|------------|
| E1 | Pre-specified subgroup audit | Age discrimination gaps: 0.083--0.145 across all four scores |
| E2 | Intersectional analysis (388 subgroups) | Hispanic×18--29 qSOFA: 0.619; non-additive compounding |
| E3 | Adversarial subgroup discovery | XGBoost error prediction: 0.624--0.630; age extremes dominate |
| E4 | Reference standard bias | Equalized odds RSB up to 0.126 (SOFA); age-axis RSB 0.373 |
| E5 | ML fairness comparison | GRU reduces calibration/sex/race gaps; worsens age threshold fairness |
| E6 | Score-conditional mortality | 15-fold mortality difference at SOFA 2--3 by age group |
| E7 | Clinical threshold analysis | 24-pp sensitivity gap at SOFA ≥ 6 (elderly vs. young adults) |
| E8 | SOFA component attribution | All six components show age gaps exceeding the composite |
| E9 | Hospital-stratified race | 6--10× Simpson's paradox in within-hospital race gaps |
| E13 | Isotonic recalibration | Discrimination gap invariant; calibration gaps eliminated---proves design bias |
| E14 | Propensity-score matching | Gaps widen post-match---rules out physiologic confounding |
| E15 | Fairness-Aware Feature Transformer | 27% age gap reduction vs. SOFA (0.094±0.003 vs. 0.129) |

---

## Requirements

```
Python 3.10+
torch >= 2.0
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
xgboost >= 1.7
scipy >= 1.10
```

---

## License

MIT License. Copyright (c) 2026 Anonymous, Anonymous, Anonymous, Anonymous. See [LICENSE](LICENSE) for full terms.
