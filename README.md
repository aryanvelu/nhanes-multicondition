# Leakage-Safe Multi-Condition Cardiometabolic Risk Prediction (NHANES)

A single multi-task model that predicts five comorbid chronic conditions — **diabetes, prediabetes, obesity, hypertension, and metabolic syndrome** — from one NHANES participant profile (demographics, body measurements, labs, questionnaire data), trained on 29,400 participants from three NHANES cycles (2013–2018).

## Why this is different

Many published NHANES models quietly include the measurement that *defines* the label (fasting glucose or HbA1c for diabetes, blood pressure for hypertension, waist/triglycerides/HDL for metabolic syndrome). They report AUROCs of 0.89–0.97, but that is recovering a diagnostic rule from its own inputs, not prediction.

This project uses **per-condition leakage masking**: each prediction head is blind only to the measurements that define *its own* label, while other conditions' measurements remain legitimate cross-condition inputs.

Other design choices:

- **Physiology-structured input layer** — first-layer wiring grouped by physiological system (glycemic, lipid, adiposity, …) instead of a fully-connected layer.
- **Feature Tokenizer Transformer** backbone compared against the grouped MLP and tuned XGBoost / LightGBM.
- **Survey-weighted evaluation** — NHANES uses complex multistage sampling; metrics use survey weights.
- **Cluster-aware split** — no NHANES primary sampling unit is divided across train and test.
- **Self-supervised masked-feature pretraining**, SMOTE, hyperparameter search, and ensembling were all ablated.

## Results (held-out test split, survey-weighted AUROC)

| Condition | FT-Transformer | XGBoost | LightGBM | **Stacked ensemble** |
|---|---:|---:|---:|---:|
| Diabetes | 0.862 | 0.870 | 0.863 | **0.869** |
| Prediabetes | 0.770 | 0.769 | 0.765 | **0.775** |
| Obesity | 0.852 | 0.859 | 0.855 | **0.859** |
| Hypertension | 0.819 | 0.818 | 0.820 | **0.821** |
| Metabolic syndrome | 0.879 | 0.884 | 0.883 | **0.885** |
| **Mean** | 0.836 | 0.840 | 0.837 | **0.842** |

Among comparable studies that also avoid label-defining inputs, the ensemble matches or exceeds published results (e.g. obesity 0.859 vs 0.75–0.81 without anthropometric inputs; hypertension 0.821 vs 0.751 without blood-pressure inputs). Pretraining, SMOTE, wider hyperparameter search, and a fourth ensemble member all failed to beat this, which suggests ~0.84 is close to the ceiling for this feature set under a non-circular labelling constraint. The intended use is low-cost risk stratification and referral triage, not replacing lab diagnosis.

## Repository layout

```
src/data/         download, merge, labels, leakage masking, preprocessing, splits
src/models/       structured layer, grouped-MLP backbone, FT-Transformer
src/training/     pretraining, fine-tuning, sweeps, final training scripts
src/baselines/    XGBoost and LightGBM baselines
src/evaluation/   survey-weighted metrics, calibration, stacking
experiments/      per-run configs and metrics (JSON)
results/          human-readable result tables for every sweep/ablation
```

Raw data, processed parquet files, and model checkpoints are **not** committed (see `.gitignore`).

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Training ran on Apple Silicon through PyTorch's MPS backend (`torch.device("mps")`); no CUDA GPU is required.

## Pipeline

```bash
python -m src.data.download        # NHANES cycles from cdc.gov (public, no login)
python -m src.data.merge           # join modules by SEQN within each cycle
python -m src.data.labels          # five diagnostic labels from clinical thresholds
python -m src.data.preprocess      # normalization, categorical encoding, missingness flags
python -m src.data.splits          # cluster-aware train/val/test split
python -m src.training.pretrain    # optional masked-reconstruction pretraining
python -m src.training.finetune    # multi-head supervised fine-tuning
python -m src.baselines.xgboost_baseline
python -m src.evaluation.stacking  # logistic-regression stacking ensemble
```
The leakage-masking step lives in `src/data/perhead_leakage.py`. The exact run order and flags used for each experiment are recorded in the module docstrings and in `results/`.

## Data

All data come from the CDC's [NHANES](https://wwwn.cdc.gov/nchs/nhanes/) continuous survey, public and free to download.

## Author

Aryan Velu
