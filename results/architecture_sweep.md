# Architecture comparison — grouped-MLP vs. FT-Transformer (10 seeds x 2 architectures = 20 runs)

Validation-split AUROC, mean +/- std across seeds 0..9. Both per-head leakage masking, weight-off, **no pretraining for either** (FT-Transformer has no pretrained checkpoint, so grouped-MLP runs without one too, for a fair architecture-only comparison).

| architecture | diabetes | prediabetes | obesity | hypertension | metab. syndrome | mean |
|---|---:|---:|---:|---:|---:|---:|
| grouped-mlp | 0.845 ± 0.004 | 0.741 ± 0.004 | 0.769 ± 0.007 | 0.827 ± 0.002 | 0.859 ± 0.006 | 0.808 |
| ft-transformer | 0.855 ± 0.002 | 0.750 ± 0.002 | 0.775 ± 0.003 | 0.830 ± 0.001 | 0.869 ± 0.002 | 0.816 |

## Does either architecture win, or is the gap noise?

Same rule as every other effect in this project: a gap only counts if it's bigger than the noise floor (the larger of the two architectures' per-condition standard deviation across seeds).

| condition | ft-transformer − grouped-mlp | noise floor | gap real? | winner |
|---|---:|---:|:---:|:---:|
| diabetes | +0.010 | 0.004 | **yes** | ft-transformer |
| prediabetes | +0.010 | 0.004 | **yes** | ft-transformer |
| obesity | +0.006 | 0.007 | no | tie (noise) |
| hypertension | +0.004 | 0.002 | **yes** | ft-transformer |
| metabolic_syndrome | +0.010 | 0.006 | **yes** | ft-transformer |
