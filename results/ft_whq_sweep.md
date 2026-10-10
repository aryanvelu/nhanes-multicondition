# FT-Transformer: before vs. after WHQ (weight-history) features (10 seeds)

Same FT-Transformer, no pretraining (settled negative — see `results/ft_pretrain_sweep.md`), per-head leakage masking, weight-off, same 10 fixed seeds. Only the feature set changed: `weight_history` (`WHD020`/`WHD050`/`WHD120`) added, with `WHD020` **and** `WHD050` masked from the obesity head (both strong self-report proxies for current body size — an earlier run with only `WHD020` masked produced a suspicious isolated +0.160 obesity jump traced to `WHD050`'s 0.556 correlation with the label; see `report.md`). `WHD120` (weight at age 25, correlation 0.275) stays visible everywhere including obesity, as genuine history rather than a near-restatement of the label.

| condition | before (no WHQ) | after (+ WHQ) | delta | noise floor | real? |
|---|---:|---:|---:|---:|:---:|
| diabetes | 0.855 ± 0.002 | 0.854 ± 0.002 | -0.001 | 0.002 | no |
| prediabetes | 0.750 ± 0.002 | 0.754 ± 0.002 | +0.004 | 0.002 | **yes** |
| obesity **<- obesity** | 0.775 ± 0.003 | 0.834 ± 0.002 | +0.059 | 0.003 | **yes** |
| hypertension | 0.830 ± 0.001 | 0.830 ± 0.002 | -0.000 | 0.002 | no |
| metabolic_syndrome | 0.869 ± 0.002 | 0.870 ± 0.002 | +0.001 | 0.002 | no |
| **mean** | **0.816** | **0.828** | **+0.013** | | |

## Obesity, specifically

Obesity AUROC moved +0.059 (0.775 -> 0.834), against a noise floor of 0.003 — a real change. `WHD020` and `WHD050` are both masked for this head specifically (same as `BMXBMI`/`BMXWAIST`/`BMXWT`/`BMXHT`), so this reflects `WHD120` (visible, correlation 0.275 with the label) plus every other unmasked group — not the two proxy features obesity never sees.
