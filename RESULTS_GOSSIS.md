# Results — ATLAS on WiDS/GOSSIS

**Last Updated**: 2026-04-06 00:41
**Dataset**: WiDS Datathon 2020 / GOSSIS


## Cohort Summary
**N**: 87315 ICU stays from 147 hospitals
**Mortality rate**: 0.083
| Attribute | Distribution |
|-----------|-------------|
| Race | {'White': 67054, 'Black': 9256, 'Other': 6347, 'Hispanic': 3565, 'Asian': 1093} |
| Sex | {'Male': 47645, 'Female': 39657} |
| Age (mean) | 62.4 +/- 16.7 |
| Diag type | {'Medical': 62593, 'Cardiac': 19766, 'Surgical': 4956} |

### Score Distributions
| Score | Mean | Median | Range |
|-------|------|--------|-------|
| SOFA | 3.1 | 2 | [0, 18] |
| qSOFA | 1.6 | 2 | [0, 3] |
| APACHE-II | 15.7 | 14 | [0, 51] |
| NEWS2 | 9.4 | 9 | [0, 20] |


## E1: Pre-specified Subgroup Audit
### AUROC Gaps
| score   | axis      |   auroc_gap | worst_group   | best_group   |   worst_auroc |   best_auroc |
|:--------|:----------|------------:|:--------------|:-------------|--------------:|-------------:|
| sofa    | race_cat  |  0.0252712  | White         | Black        |      0.775641 |     0.800912 |
| sofa    | sex       |  0.00943052 | Male          | Female       |      0.776809 |     0.786239 |
| sofa    | age_group |  0.129371   | 80+           | 18-29        |      0.735287 |     0.864657 |
| sofa    | diag_type |  0.0236643  | Surgical      | Medical      |      0.76216  |     0.785824 |
| qsofa   | race_cat  |  0.0328111  | Asian         | Other        |      0.707649 |     0.74046  |
| qsofa   | sex       |  0.0193934  | Female        | Male         |      0.708568 |     0.727961 |
| qsofa   | age_group |  0.0825676  | 80+           | 18-29        |      0.681419 |     0.763987 |
| qsofa   | diag_type |  0.0175856  | Surgical      | Medical      |      0.702969 |     0.720555 |
| apache2 | race_cat  |  0.0194296  | White         | Other        |      0.822139 |     0.841569 |
| apache2 | sex       |  0.00274048 | Male          | Female       |      0.823259 |     0.825999 |
| apache2 | age_group |  0.145225   | 80+           | 18-29        |      0.766568 |     0.911794 |
| apache2 | diag_type |  0.0267934  | Surgical      | Medical      |      0.803416 |     0.83021  |
| news2   | race_cat  |  0.0271796  | White         | Other        |      0.789857 |     0.817037 |
| news2   | sex       |  0.0077473  | Female        | Male         |      0.789922 |     0.797669 |
| news2   | age_group |  0.11383    | 80+           | 18-29        |      0.761787 |     0.875617 |
| news2   | diag_type |  0.0583423  | Surgical      | Medical      |      0.739    |     0.797343 |


## E2: Intersectional Analysis
**Subgroups evaluated**: 388

### SOFA — Worst Subgroups
| subgroup         |    auroc |     n |   prevalence |
|:-----------------|---------:|------:|-------------:|
| Asian x 80+      | 0.66683  |   167 |    0.125749  |
| Asian x Cardiac  | 0.700917 |   215 |    0.0418605 |
| Other x 80+      | 0.706144 |   726 |    0.136364  |
| 40-49 x Surgical | 0.715559 |   419 |    0.0572792 |
| 70-79 x Surgical | 0.722618 |  1172 |    0.0656997 |
| Male x 80+       | 0.723135 |  6771 |    0.13676   |
| White x 80+      | 0.73236  | 11704 |    0.13081   |
| 80+ x Medical    | 0.732488 | 10116 |    0.141558  |
| Hispanic x 50-59 | 0.735584 |   601 |    0.0665557 |
| 70-79 x Cardiac  | 0.735807 |  4964 |    0.0876309 |
### QSOFA — Worst Subgroups
| subgroup         |    auroc |     n |   prevalence |
|:-----------------|---------:|------:|-------------:|
| Hispanic x 18-29 | 0.618697 |   246 |    0.0325203 |
| Asian x 80+      | 0.654925 |   167 |    0.125749  |
| 70-79 x Surgical | 0.663221 |  1172 |    0.0656997 |
| 80+ x Medical    | 0.669668 | 10116 |    0.141558  |
| 60-69 x Surgical | 0.670591 |  1173 |    0.0673487 |
| Asian x 70-79    | 0.674258 |   234 |    0.106838  |
| Black x Surgical | 0.674692 |   709 |    0.0620592 |
| 50-59 x Surgical | 0.674757 |   942 |    0.0467091 |
| Asian x Cardiac  | 0.675836 |   215 |    0.0418605 |
| Female x 80+     | 0.676558 |  7267 |    0.130178  |
### APACHE2 — Worst Subgroups
| subgroup         |    auroc |     n |   prevalence |
|:-----------------|---------:|------:|-------------:|
| Asian x 80+      | 0.71983  |   167 |    0.125749  |
| 40-49 x Surgical | 0.727954 |   419 |    0.0572792 |
| Other x 80+      | 0.748377 |   726 |    0.136364  |
| Male x 80+       | 0.748605 |  6771 |    0.13676   |
| Hispanic x 50-59 | 0.76152  |   601 |    0.0665557 |
| White x 80+      | 0.76303  | 11704 |    0.13081   |
| 80+ x Medical    | 0.764118 | 10116 |    0.141558  |
| Asian x Cardiac  | 0.76753  |   215 |    0.0418605 |
| 80+ x Cardiac    | 0.76898  |  3170 |    0.10694   |
| Asian x 70-79    | 0.772536 |   234 |    0.106838  |
### NEWS2 — Worst Subgroups
| subgroup          |    auroc |    n |   prevalence |
|:------------------|---------:|-----:|-------------:|
| 70-79 x Surgical  | 0.685323 | 1172 |    0.0656997 |
| 40-49 x Surgical  | 0.705327 |  419 |    0.0572792 |
| Other x Surgical  | 0.717167 |  253 |    0.0790514 |
| Black x Surgical  | 0.718985 |  709 |    0.0620592 |
| 50-59 x Surgical  | 0.720414 |  942 |    0.0467091 |
| Female x Surgical | 0.731604 | 2121 |    0.0740217 |
| Other x 80+       | 0.733652 |  726 |    0.136364  |
| 60-69 x Surgical  | 0.735577 | 1173 |    0.0673487 |
| Asian x 70-79     | 0.738469 |  234 |    0.106838  |
| Male x Surgical   | 0.74668  | 2835 |    0.0666667 |


## E3: Adversarial Subgroup Discovery

### SOFA
- Error prediction AUROC: 0.627
- Top features: ['age_group_80+', 'age_group_70-79', 'age_group_18-29', 'age', 'unit_MICU']

### QSOFA
- Error prediction AUROC: 0.630
- Top features: ['age_group_80+', 'age_group_70-79', 'age_group_18-29', 'age_group_40-49', 'age']

### APACHE2
- Error prediction AUROC: 0.624
- Top features: ['age_group_80+', 'age_group_18-29', 'age_group_70-79', 'age_group_50-59', 'age']

### NEWS2
- Error prediction AUROC: 0.630
- Top features: ['age_group_80+', 'age_group_70-79', 'age_group_18-29', 'age_group_40-49', 'age']


## E4: Reference Standard Bias
### RSB Gap Summary
| score   |   auroc_gap |   cal_gap |       eod |       ppg |
|:--------|------------:|----------:|----------:|----------:|
| apache2 |  0.00997176 | 0.094844  | 0.0137986 | 0.103129  |
| news2   |  0.0136231  | 0.0595725 | 0.0326794 | 0.0289505 |
| qsofa   |  0.0170683  | 0.0780992 | 0.0375107 | 0.02536   |
| sofa    |  0.0146118  | 0.0471276 | 0.0410761 | 0.0348978 |

## E5: ML Fairness Improvement
| score   |   pct_improvement |
|:--------|------------------:|
| apache2 |           3.00589 |
| news2   |          19.6702  |
| qsofa   |          28.3776  |
| sofa    |          36.6906  |

### GRU Performance
- Overall CV AUROC: 0.8707
- Training samples: 20000


---
## Pipeline Complete
**Date**: 2026-04-06 02:18


---
## Full-Cohort GRU Results — 2026-04-11 03:28

**GRU AUROC (full 87315 stays)**: 0.8801
**Fold metrics**: [{'fold': 0, 'auroc': 0.884024158163364, 'val_loss': 0.7846152576522061}, {'fold': 1, 'auroc': 0.8744301243243964, 'val_loss': 0.814842912206083}, {'fold': 2, 'auroc': 0.8823168202254871, 'val_loss': 0.7925653251211497}, {'fold': 3, 'auroc': 0.8791368472840629, 'val_loss': 0.8001782529997687}, {'fold': 4, 'auroc': 0.8809673599178179, 'val_loss': 0.796127457457747}]

### E4 RSB (full cohort)
| score   |   auroc_gap |   cal_gap |   ece_gap |       eod |       ppg |
|:--------|------------:|----------:|----------:|----------:|----------:|
| apache2 |   0.0103209 | 0.0497229 | 0.0322568 | 0.0496839 | 0.022091  |
| news2   |   0.0181379 | 0.090541  | 0.0222393 | 0.110285  | 0.047951  |
| qsofa   |   0.0127177 | 0.0693763 | 0.0173266 | 0.116904  | 0.0257667 |
| sofa    |   0.0146053 | 0.0730495 | 0.0406318 | 0.125586  | 0.0453516 |

### E5 ML Improvement (full cohort)
| score   |   auroc_gap |   cal_gap |   ece_gap |       eod |       ppg |
|:--------|------------:|----------:|----------:|----------:|----------:|
| apache2 |    -36.8261 | 33.0641   |  -559.519 |   6.39116 | -13.8133  |
| news2   |     22.9965 | 29.9074   |  -875.68  | -61.9056  |  -3.23727 |
| qsofa   |     -1.3202 | -0.396812 | -1204.11  | -76.0574  |  21.5432  |
| sofa    |      7.5904 | 36.7539   |  -227.333 | -97.5181  |  30.5311  |

---
## FAFT Results — 2026-04-11 03:39

**FAFT AUROC (full 87315 stays)**: 0.8839
**Architecture**: d_model=64, n_heads=4, n_layers=2, d_ff=256, adv_lambda=0.3
**Fold metrics**: [{'fold': 0, 'auroc': 0.8882351575688457, 'val_loss': 0.7714496228741745}, {'fold': 1, 'auroc': 0.8802495257394616, 'val_loss': 0.8062489459889797}, {'fold': 2, 'auroc': 0.8863553534374808, 'val_loss': 0.7784164765255494}, {'fold': 3, 'auroc': 0.8828082905222758, 'val_loss': 0.7971581277285887}, {'fold': 4, 'auroc': 0.8857874005363043, 'val_loss': 0.7804826502673563}]

### E4 RSB (FAFT)
| score   |   auroc_gap |   cal_gap |   ece_gap |       eod |        ppg |
|:--------|------------:|----------:|----------:|----------:|-----------:|
| apache2 |   0.0093894 | 0.0731993 | 0.0303831 | 0.0200942 | 0.0184342  |
| news2   |   0.0245169 | 0.0599279 | 0.0244784 | 0.0974684 | 0.025461   |
| qsofa   |   0.02083   | 0.0489808 | 0.0193512 | 0.100788  | 0.00571033 |
| sofa    |   0.0158629 | 0.0438133 | 0.0402198 | 0.103247  | 0.0243749  |

### E5 ML Improvement (FAFT)
| score   |   auroc_gap |   cal_gap |   ece_gap |       eod |       ppg |
|:--------|------------:|----------:|----------:|----------:|----------:|
| apache2 |   -16.8306  |   48.7128 |  -765.129 |  -10.4159 | -22.2104  |
| news2   |    29.0182  |   47.6805 | -1371.97  |  -59.8557 |  -6.73736 |
| qsofa   |    -1.29829 |   23.0727 | -1933.01  |  -68.211  |  19.3103  |
| sofa    |    11.3694  |   49.7543 |  -280.591 | -113.661  |  28.7391  |