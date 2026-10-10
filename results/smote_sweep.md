# SMOTE ablation — prediabetes & metabolic_syndrome (10 seeds x 2 variants = 20 runs)

Validation-split AUROC, mean +/- std across seeds 0..9, at the established best config (per-head, pretrain-on, weight-off — same seeds used in `results/seed_sweep.md`). SMOTE (`SMOTENC`, training split only, never val/test) balances prediabetes and metabolic_syndrome to parity; every other head trains on real data only and ignores the synthetic rows via `_valid=0`.

| variant | diabetes | prediabetes | obesity | hypertension | metab. syndrome | mean |
|---|---:|---:|---:|---:|---:|---:|
| smote-on | 0.847 ± 0.003 | 0.736 ± 0.006 | 0.764 ± 0.009 | 0.822 ± 0.003 | 0.856 ± 0.005 | 0.805 |
| smote-off | 0.851 ± 0.003 | 0.745 ± 0.003 | 0.773 ± 0.004 | 0.826 ± 0.001 | 0.867 ± 0.003 | 0.813 |

## Does the SMOTE gap survive multiple seeds?

Same rule as the pretrain/weight ablation: a gap only counts if it's bigger than the noise floor (the larger of the two variants' per-condition standard deviation across seeds).

| condition | smote-on − smote-off | noise floor (max of the two std) | gap real? | SMOTE'd this condition? |
|---|---:|---:|:---:|:---:|
| diabetes | -0.004 | 0.003 | **yes** | no (control) |
| prediabetes | -0.010 | 0.006 | **yes** | yes |
| obesity | -0.009 | 0.009 | no | no (control) |
| hypertension | -0.004 | 0.003 | **yes** | no (control) |
| metabolic_syndrome | -0.011 | 0.005 | **yes** | yes |
