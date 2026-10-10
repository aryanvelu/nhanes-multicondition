# Stacking ensemble — logistic-regression blend of FT-Transformer + XGBoost + LightGBM

Per-condition logistic regression on `[ft_prob, xgb_prob, lgbm_prob]`, fit on **validation-split** predictions from all three base models (never on data any base model was trained on), scored on the held-out **test** split with the same survey-weighted AUROC/AUPRC used throughout this project. All three base models trained on the current, fully-enriched feature set. Two 4th-base-learner candidates (grouped-MLP, pretrained FT-Transformer) were tried and rejected — see module docstring.

Caveat: all three base models used the validation split for early-stopping model selection, just not for fitting weights — standard practice for blending, not a fully nested cross-validation.

## Survey-weighted test AUROC

| condition | n | prevalence | FT-Transformer | XGBoost | LightGBM | **stacked** | best |
|---|---:|---:|---:|---:|---:|---:|:---:|
| diabetes | 2,559 | 17.1% | 0.862 | 0.870 | 0.863 | **0.869** | xgb |
| prediabetes | 2,559 | 34.2% | 0.770 | 0.769 | 0.765 | **0.775** | stack |
| obesity | 2,515 | 35.9% | 0.852 | 0.859 | 0.855 | **0.859** | stack |
| hypertension | 2,405 | 50.9% | 0.819 | 0.818 | 0.820 | **0.821** | stack |
| metabolic_syndrome | 996 | 34.7% | 0.879 | 0.884 | 0.883 | **0.885** | stack |
| **mean** | | | **0.836** | **0.840** | **0.837** | **0.842** | |

## Survey-weighted test AUPRC

| condition | FT-Transformer | XGBoost | LightGBM | **stacked** | best |
|---|---:|---:|---:|---:|:---:|
| diabetes | 0.475 | 0.480 | 0.467 | **0.482** | stack |
| prediabetes | 0.585 | 0.585 | 0.570 | **0.594** | stack |
| obesity | 0.767 | 0.777 | 0.773 | **0.778** | stack |
| hypertension | 0.791 | 0.787 | 0.791 | **0.793** | stack |
| metabolic_syndrome | 0.792 | 0.794 | 0.792 | **0.799** | stack |

## Meta-model coefficients (what does the blend lean on?)

`P = sigmoid(coef_ft*ft_prob + coef_xgb*xgb_prob + coef_lgbm*lgbm_prob + intercept)` — all coefficients positive and none dominating means the blend is genuinely using more than one model, not just reproducing the best single one.

| condition | coef (FT) | coef (xgb) | coef (lgbm) | intercept |
|---|---:|---:|---:|---:|
| diabetes | 3.10 | 1.90 | 1.88 | -3.25 |
| prediabetes | 2.92 | 1.34 | 1.48 | -2.63 |
| obesity | 2.47 | 1.65 | 1.76 | -2.84 |
| hypertension | 1.15 | 2.02 | 1.91 | -2.66 |
| metabolic_syndrome | 2.28 | 1.75 | 1.69 | -3.01 |

## Verdict

Stacking wins on AUROC in **4/5** conditions (mean AUROC 0.842 vs FT-Transformer 0.836, XGBoost 0.840, LightGBM 0.837) and on AUPRC in **5/5**.
