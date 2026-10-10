# FT-Transformer: pretrained vs. non-pretrained (10 seeds x 2 variants = 20 runs)

Masked-feature-**token** pretraining (`pretrain_ft.py`) vs. random init, same per-head leakage masking / weight-off training regime, same 10 fixed seeds used by every other sweep in this project. Both val (noise-floor check) and test (held-out, survey-weighted) results below — train/val/test.

## Validation AUROC (unweighted, mean +/- std across seeds)

| variant | diabetes | prediabetes | obesity | hypertension | metab. syndrome | mean |
|---|---:|---:|---:|---:|---:|---:|
| pretrain-on | 0.869 ± 0.003 | 0.766 ± 0.002 | 0.853 ± 0.002 | 0.835 ± 0.002 | 0.885 ± 0.003 | 0.842 |
| pretrain-off | 0.872 ± 0.002 | 0.771 ± 0.002 | 0.852 ± 0.002 | 0.837 ± 0.002 | 0.886 ± 0.004 | 0.844 |

### Does the val AUROC gap survive multiple seeds?

| condition | pretrain-on − pretrain-off | noise floor | gap real? |
|---|---:|---:|:---:|
| diabetes | -0.003 | 0.003 | no |
| prediabetes | -0.005 | 0.002 | **yes** |
| obesity | +0.001 | 0.002 | no |
| hypertension | -0.002 | 0.002 | **yes** |
| metabolic_syndrome | -0.001 | 0.004 | no |

## Test AUROC (survey-weighted, mean +/- std across the same seeds)

| variant | diabetes | prediabetes | obesity | hypertension | metab. syndrome | mean |
|---|---:|---:|---:|---:|---:|---:|
| pretrain-on | 0.867 ± 0.004 | 0.769 ± 0.004 | 0.853 ± 0.002 | 0.816 ± 0.002 | 0.872 ± 0.005 | 0.835 |
| pretrain-off | 0.862 ± 0.002 | 0.772 ± 0.004 | 0.854 ± 0.003 | 0.820 ± 0.002 | 0.878 ± 0.004 | 0.837 |

### Does the test AUROC gap survive multiple seeds?

| condition | pretrain-on − pretrain-off | noise floor | gap real? |
|---|---:|---:|:---:|
| diabetes | +0.005 | 0.004 | **yes** |
| prediabetes | -0.003 | 0.004 | no |
| obesity | -0.001 | 0.003 | no |
| hypertension | -0.004 | 0.002 | **yes** |
| metabolic_syndrome | -0.005 | 0.005 | **yes** |

## Test AUPRC (survey-weighted, mean +/- std across the same seeds)

| variant | diabetes | prediabetes | obesity | hypertension | metab. syndrome |
|---|---:|---:|---:|---:|---:|
| pretrain-on | 0.473 ± 0.012 | 0.579 ± 0.008 | 0.766 ± 0.005 | 0.783 ± 0.005 | 0.773 ± 0.009 |
| pretrain-off | 0.462 ± 0.008 | 0.588 ± 0.003 | 0.771 ± 0.007 | 0.789 ± 0.004 | 0.779 ± 0.010 |
