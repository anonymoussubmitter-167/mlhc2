# Training Log — ATLAS


## Run 001 — 2026-04-05 17:16
- **Experiment**: E4-E5 GRU mortality prediction
- **Config**:
  - Model: MortalityGRU (hidden=128, layers=2, dropout=0.3)
  - LR: 1e-3, Schedule: ReduceLROnPlateau
  - Batch size: 256
  - Epochs: 50 (early stopping patience=10)
  - 5-fold CV
- **Data**: 86 ICU stays, 0.081 mortality rate
- **Hardware**: cpu
- **Final Metrics**:
  - Overall CV AUROC: 0.7233
  - Model params: 215,809
- **Fold metrics**: [{'fold': 0, 'auroc': 0.625, 'val_loss': 1.5981677770614624}, {'fold': 1, 'auroc': 1.0, 'val_loss': 0.854205310344696}, {'fold': 2, 'auroc': 0.9375, 'val_loss': 0.43155819177627563}, {'fold': 3, 'auroc': 0.5625, 'val_loss': 1.0490046739578247}, {'fold': 4, 'auroc': 0.6333333333333333, 'val_loss': 1.6329270601272583}]
- **Status**: Completed


## Run eICU-001 — 2026-04-05 23:31
- **Dataset**: eICU-CRD Demo (1238 stays)
- **Hardware**: cpu
- **Overall CV AUROC**: 0.7963
- **Model params**: 213,377


## Run GOSSIS-001 — 2026-04-06 02:47
- **Dataset**: WiDS/GOSSIS (20,000 sampled from 87,315)
- **Hardware**: cpu (using ~4 cores)
- **Status**: In Progress (Fold 1/5 running, estimated 2.5h remaining)
- **Config**: Same as eICU-001 (hidden=128, layers=2, epochs=50, patience=10, batch=256, 5-fold CV)
- **Note**: Full GOSSIS E4-E5 run; will update when complete


## Run GOSSIS-001 — 2026-04-06 02:18
- **Dataset**: GOSSIS (20000 stays)
- **Device**: cpu
- **Overall CV AUROC**: 0.8707
- **Fold metrics**: [{'fold': 0, 'auroc': 0.8650417839982976, 'val_loss': 0.8463907299041749}, {'fold': 1, 'auroc': 0.8866412918471317, 'val_loss': 0.7851730995178222}, {'fold': 2, 'auroc': 0.8869763025348856, 'val_loss': 0.7779648356437683}, {'fold': 3, 'auroc': 0.8613756089505409, 'val_loss': 0.8617448420524597}, {'fold': 4, 'auroc': 0.8577099331186525, 'val_loss': 0.86410520362854}]


## Run GOSSIS-FULL — 2026-04-11 03:28
- **Dataset**: GOSSIS (FULL, 87315 stays)
- **Device**: cuda:0
- **Overall CV AUROC**: 0.8801
- **Fold metrics**: [{'fold': 0, 'auroc': 0.884024158163364, 'val_loss': 0.7846152576522061}, {'fold': 1, 'auroc': 0.8744301243243964, 'val_loss': 0.814842912206083}, {'fold': 2, 'auroc': 0.8823168202254871, 'val_loss': 0.7925653251211497}, {'fold': 3, 'auroc': 0.8791368472840629, 'val_loss': 0.8001782529997687}, {'fold': 4, 'auroc': 0.8809673599178179, 'val_loss': 0.796127457457747}]


## FAFT Run — 2026-04-11 03:39
- **Dataset**: GOSSIS (FULL, 87315 stays)
- **Device**: cuda:0
- **Architecture**: FAFT (d=64, heads=4, layers=2, ff=256, adv_lambda=0.3)
- **Overall CV AUROC**: 0.8839
- **Fold metrics**: [{'fold': 0, 'auroc': 0.8882351575688457, 'val_loss': 0.7714496228741745}, {'fold': 1, 'auroc': 0.8802495257394616, 'val_loss': 0.8062489459889797}, {'fold': 2, 'auroc': 0.8863553534374808, 'val_loss': 0.7784164765255494}, {'fold': 3, 'auroc': 0.8828082905222758, 'val_loss': 0.7971581277285887}, {'fold': 4, 'auroc': 0.8857874005363043, 'val_loss': 0.7804826502673563}]
