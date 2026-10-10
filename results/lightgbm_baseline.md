# LightGBM baseline — three-way comparison (deep / XGBoost / LightGBM)

Same per-head leakage-masked features, splits, and survey-weighted AUROC/AUPRC as the deep model and the XGBoost baseline. All three trained without survey-weighted loss (the deep model's best-AUROC config), so this is architecture vs. architecture, not weighting scheme vs. weighting scheme.

## Survey-weighted test AUROC

| condition | n (test) | prevalence | deep | XGBoost | LightGBM | best |
|---|---:|---:|---:|---:|---:|:---:|
| diabetes | 2,559 | 17.1% | 0.857 | 0.870 | 0.863 | xgb |
| prediabetes | 2,559 | 34.2% | 0.746 | 0.769 | 0.765 | xgb |
| obesity | 2,515 | 35.9% | 0.791 | 0.859 | 0.855 | xgb |
| hypertension | 2,405 | 50.9% | 0.815 | 0.818 | 0.820 | lgbm |
| metabolic_syndrome | 996 | 34.7% | 0.870 | 0.884 | 0.883 | xgb |
| **mean** | | | **0.816** | **0.840** | **0.837** | |

## Survey-weighted test AUPRC

| condition | deep | XGBoost | LightGBM | best |
|---|---:|---:|---:|:---:|
| diabetes | 0.457 | 0.480 | 0.467 | xgb |
| prediabetes | 0.566 | 0.585 | 0.570 | xgb |
| obesity | 0.670 | 0.777 | 0.773 | xgb |
| hypertension | 0.785 | 0.787 | 0.791 | lgbm |
| metabolic_syndrome | 0.759 | 0.794 | 0.792 | xgb |

## Verdict

AUROC wins by condition: deep 0/5, XGBoost 4/5, LightGBM 1/5 (mean: deep 0.816, XGBoost 0.840, LightGBM 0.837).
