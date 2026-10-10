# Test-set metrics — per-head (middle-ground) model

Held-out test split: **2,694 adults**. Best config by mean val AUROC: **pretrain-on_weight-off**.

## Part A — discrimination (best config)

Per condition, unweighted vs survey-weighted AUROC/AUPRC. Weight: `WTMEC2YR` for the four general heads, `WTSAF2YR` for metabolic syndrome.

| condition | n valid | prev. | AUROC (unw.) | AUROC (wt.) | AUPRC (unw.) | AUPRC (wt.) |
|---|---:|---:|---:|---:|---:|---:|
| diabetes | 2,559 | 17.1% | 0.845 | 0.857 | 0.527 | 0.457 |
| prediabetes | 2,559 | 34.2% | 0.728 | 0.746 | 0.550 | 0.566 |
| obesity | 2,515 | 35.9% | 0.778 | 0.791 | 0.668 | 0.670 |
| hypertension | 2,405 | 50.9% | 0.823 | 0.815 | 0.811 | 0.785 |
| metabolic_syndrome | 996 | 34.7% | 0.842 | 0.870 | 0.738 | 0.759 |

## Part B — subgroup calibration: weighted vs unweighted training

For each subgroup, survey-weighted **mean predicted probability** vs **observed prevalence**; the gap is the calibration error. Lower and more uniform gaps = better calibration. Two models compared: trained **with** survey weighting (`pretrain-on_weight-on`) vs **without** (`pretrain-on_weight-off`).


### Race / ethnicity

| model | mean |gap| across subgroups | max |gap| |
|---|---:|---:|
| trained WITH survey weight | 0.040 | 0.128 |
| trained WITHOUT survey weight | 0.037 | 0.100 |

Detail — metabolic syndrome by race / ethnicity (pred / observed / gap), weighted model | unweighted model:

| subgroup | n | W pred | W obs | W gap | U pred | U obs | U gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mexican American | 208 | 0.364 | 0.442 | 0.078 | 0.386 | 0.442 | 0.056 |
| NH Black | 175 | 0.391 | 0.263 | 0.128 | 0.364 | 0.263 | 0.100 |
| NH White | 264 | 0.282 | 0.335 | 0.054 | 0.293 | 0.335 | 0.042 |
| Other Hispanic | 145 | 0.285 | 0.302 | 0.017 | 0.314 | 0.302 | 0.012 |
| Other/Multi | 204 | 0.218 | 0.195 | 0.023 | 0.233 | 0.195 | 0.038 |

### Income category

| model | mean |gap| across subgroups | max |gap| |
|---|---:|---:|
| trained WITH survey weight | 0.034 | 0.088 |
| trained WITHOUT survey weight | 0.036 | 0.086 |

Detail — metabolic syndrome by income category (pred / observed / gap), weighted model | unweighted model:

| subgroup | n | W pred | W obs | W gap | U pred | U obs | U gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| High (>=3.5) | 304 | 0.280 | 0.287 | 0.007 | 0.291 | 0.287 | 0.004 |
| Low (<1.3) | 270 | 0.311 | 0.329 | 0.018 | 0.319 | 0.329 | 0.009 |
| Middle (1.3-3.5) | 331 | 0.313 | 0.381 | 0.068 | 0.324 | 0.381 | 0.057 |
| Unknown | 91 | 0.273 | 0.251 | 0.022 | 0.299 | 0.251 | 0.048 |

## Verdict — was survey weighting worth it?

- **Race / ethnicity:** more consistent calibration when trained **WITHOUT** survey weighting (mean |gap| 0.040 weighted vs 0.037 unweighted).
- **Income category:** more consistent calibration when trained **WITH** survey weighting (mean |gap| 0.034 weighted vs 0.036 unweighted).

1 of 2 subgroup analyses favor survey-weighted training on calibration consistency — the metric that actually reflects population fairness, unlike raw unweighted AUROC.
