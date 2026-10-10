# Multi-seed ablation — per-head leakage setup (10 seeds x 4 variants = 40 runs)

Validation-split AUROC, mean +/- std across seeds 0..9. Same per-head leakage masking, features, and splits as the single-seed ablation in `results/ablation.md` — this only adds repeated seeds.

| variant | diabetes | prediabetes | obesity | hypertension | metab. syndrome | mean |
|---|---:|---:|---:|---:|---:|---:|
| pretrain-on_weight-on | 0.801 ± 0.006 | 0.701 ± 0.012 | 0.732 ± 0.006 | 0.824 ± 0.003 | 0.838 ± 0.005 | 0.779 |
| pretrain-on_weight-off | 0.802 ± 0.014 | 0.704 ± 0.026 | 0.727 ± 0.027 | 0.826 ± 0.006 | 0.843 ± 0.012 | 0.781 |
| pretrain-off_weight-on | 0.798 ± 0.005 | 0.716 ± 0.007 | 0.713 ± 0.010 | 0.826 ± 0.004 | 0.840 ± 0.005 | 0.779 |
| pretrain-off_weight-off | 0.805 ± 0.004 | 0.724 ± 0.007 | 0.731 ± 0.007 | 0.831 ± 0.002 | 0.849 ± 0.004 | 0.788 |

## Is the weighting / pretraining effect real, or noise?

Marginal effect of survey weighting (mean over pretrain on/off) and of pretraining (mean over weight on/off), per condition, compared against the noise floor — the average per-variant standard deviation across seeds for that condition. If a gap is smaller than the noise floor, it isn't distinguishable from seed-to-seed randomness.

| condition | weight-on − weight-off | pretrain-on − pretrain-off | noise floor (avg std) | weight gap real? | pretrain gap real? |
|---|---:|---:|---:|:---:|:---:|
| diabetes | -0.004 | +0.000 | 0.007 | no | no |
| prediabetes | -0.006 | -0.017 | 0.013 | no | **yes** |
| obesity | -0.006 | +0.007 | 0.012 | no | no |
| hypertension | -0.003 | -0.003 | 0.004 | no | no |
| metabolic_syndrome | -0.006 | -0.004 | 0.006 | no | no |
