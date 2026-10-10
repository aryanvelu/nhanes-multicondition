# XGBoost baseline vs. the deep per-head model — test-set comparison

One independent `XGBClassifier` per condition, trained on the same per-head leakage-masked features and the same PSU-cluster splits as the neural model, scored with the identical survey-weighted AUROC/AUPRC functions used in `test_metrics.md`. Both models trained **without** survey-weighted loss (the deep model's best-AUROC config), so this isolates architecture, not weighting scheme.

## Survey-weighted AUROC (primary comparison)

| condition | n (test) | prevalence | XGBoost | Deep model | delta (deep − xgb) |
|---|---:|---:|---:|---:|---:|
| diabetes | 2,559 | 17.1% | 0.870 | 0.857 | -0.013 |
| prediabetes | 2,559 | 34.2% | 0.769 | 0.746 | -0.023 |
| obesity | 2,515 | 35.9% | 0.859 | 0.791 | -0.068 |
| hypertension | 2,405 | 50.9% | 0.818 | 0.815 | -0.003 |
| metabolic_syndrome | 996 | 34.7% | 0.884 | 0.870 | -0.014 |
| **mean** | | | **0.840** | **0.816** | **-0.024** |

## Survey-weighted AUPRC

| condition | XGBoost | Deep model | delta (deep − xgb) |
|---|---:|---:|---:|
| diabetes | 0.480 | 0.457 | -0.023 |
| prediabetes | 0.585 | 0.566 | -0.019 |
| obesity | 0.777 | 0.670 | -0.107 |
| hypertension | 0.787 | 0.785 | -0.002 |
| metabolic_syndrome | 0.794 | 0.759 | -0.035 |

## Unweighted AUROC/AUPRC (full detail)

| condition | XGBoost AUROC(u) | Deep AUROC(u) | XGBoost AUPRC(u) | Deep AUPRC(u) |
|---|---:|---:|---:|---:|
| diabetes | 0.861 | 0.845 | 0.555 | 0.527 |
| prediabetes | 0.747 | 0.728 | 0.562 | 0.550 |
| obesity | 0.852 | 0.778 | 0.767 | 0.668 |
| hypertension | 0.826 | 0.823 | 0.814 | 0.811 |
| metabolic_syndrome | 0.856 | 0.842 | 0.771 | 0.738 |

## Verdict

The deep model beats the XGBoost baseline on survey-weighted AUROC in **0 of 5** conditions (mean AUROC 0.816 vs 0.840).
