# Hyperparameter search — deep model and XGBoost

Random search from each model's previously-established winning config, selected by mean **unweighted validation AUROC** across the 5 conditions (same method the 4-way ablation used), then confirmed on the untouched test split with survey-weighted AUROC/AUPRC.

## Deep model

Swept: learning rate, hidden-layer width, dropout. Width != 64 means the pretrained backbone's weights don't fit, so those trials fall back to random init (flagged below).

| lr | width | dropout | pretrain used | mean val AUROC |
|---:|---:|---:|:---:|---:|
| 0.001 | 64 | 0.2 | yes | 0.8151 |
| 0.003 | 64 | 0.1 | yes | 0.8140 |
| 0.003 | 64 | 0.3 | yes | 0.8140 |
| 0.001 | 64 | 0.4 | yes | 0.8139 |
| 0.003 | 64 | 0.2 | yes | 0.8138 |
| 0.0005 | 64 | 0.4 | yes | 0.8118 |
| 0.001 | 32 | 0.4 | no | 0.8095 |
| 0.002 | 128 | 0.3 | no | 0.8082 |
| 0.0005 | 128 | 0.1 | no | 0.8070 |
| 0.0005 | 128 | 0.3 | no | 0.8063 |

**Best config:** lr=0.001, width=64, dropout=0.2, pretrain_used=True (mean val AUROC 0.8151)

| condition | val AUROC (best config) | test AUROC (wt.) | test AUPRC (wt.) |
|---|---:|---:|---:|
| diabetes | 0.854 | 0.857 | 0.457 |
| prediabetes | 0.747 | 0.746 | 0.560 |
| obesity | 0.776 | 0.792 | 0.671 |
| hypertension | 0.830 | 0.815 | 0.786 |
| metabolic_syndrome | 0.869 | 0.870 | 0.759 |

## XGBoost

Swept: `max_depth`, `learning_rate`, `n_estimators` (early stopping, 30 rounds, stays on throughout).

| max_depth | learning_rate | n_estimators | mean val AUROC |
|---:|---:|---:|---:|
| 4 | 0.05 | 1000 | 0.8176 |
| 4 | 0.03 | 1000 | 0.8176 |
| 6 | 0.03 | 600 | 0.8176 |
| 3 | 0.05 | 1000 | 0.8172 |
| 3 | 0.05 | 300 | 0.8172 |
| 5 | 0.1 | 1000 | 0.8161 |
| 3 | 0.1 | 600 | 0.8159 |
| 6 | 0.05 | 300 | 0.8157 |
| 6 | 0.05 | 600 | 0.8157 |
| 4 | 0.01 | 1000 | 0.8127 |

**Best config:** max_depth=4, learning_rate=0.05, n_estimators=1000 (mean val AUROC 0.8176)

| condition | val AUROC (best config) | test AUROC (wt.) | test AUPRC (wt.) |
|---|---:|---:|---:|
| diabetes | 0.856 | 0.853 | 0.457 |
| prediabetes | 0.749 | 0.749 | 0.556 |
| obesity | 0.779 | 0.787 | 0.654 |
| hypertension | 0.832 | 0.815 | 0.782 |
| metabolic_syndrome | 0.872 | 0.882 | 0.792 |

## Tuned deep model vs. tuned XGBoost (test, survey-weighted AUROC)

| condition | deep (tuned) | XGBoost (tuned) |
|---|---:|---:|
| diabetes | 0.857 | 0.853 |
| prediabetes | 0.746 | 0.749 |
| obesity | 0.792 | 0.787 |
| hypertension | 0.815 | 0.815 |
| metabolic_syndrome | 0.870 | 0.882 |
| **mean** | **0.816** | **0.817** |
