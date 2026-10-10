# FT-Transformer: before vs. after organ labs / alcohol / sleep / medications (10 seeds)

Same FT-Transformer, no pretraining, per-head leakage masking, weight-off, same 10 fixed seeds. New groups this round: `renal` (creatinine, BUN), `hepatic` (ALT/AST/ALP/GGT/bilirubin/albumin), `lifestyle` +alcohol (`ALQ130`) +sleep (`sleep_hours`/`SLQ050`), and `medication_use` (`RXDCOUNT`, aggregated from the one-row-per-drug `RXQ_RX` file). None read by `labels.py` — checked below for the same isolated-jump signature that caught `WHD050`.

| condition | before (+ WHQ only) | after (+ organ/lifestyle/meds) | delta | noise floor | real? |
|---|---:|---:|---:|---:|:---:|
| diabetes | 0.854 ± 0.002 | 0.866 ± 0.003 | +0.012 | 0.003 | **yes** |
| prediabetes | 0.754 ± 0.002 | 0.767 ± 0.004 | +0.013 | 0.004 | **yes** |
| obesity | 0.834 ± 0.002 | 0.848 ± 0.002 | +0.014 | 0.002 | **yes** |
| hypertension | 0.830 ± 0.002 | 0.835 ± 0.001 | +0.005 | 0.002 | **yes** |
| metabolic_syndrome | 0.870 ± 0.002 | 0.879 ± 0.002 | +0.009 | 0.002 | **yes** |
| **mean** | **0.828** | **0.839** | **+0.010** | | |

## Proxy-leak check (the WHD050 tell: isolated, disproportionate jump)

No condition shows an isolated jump bigger than +/-0.02 beyond its noise floor — gains (if any) look spread across conditions rather than concentrated on one, unlike the WHD050 case. No further per-head masking indicated by this check alone.
