# Four-way ablation — preliminary results

Model: physiology-structured backbone + five condition heads (`finetune.py`).
Cohort: adults (18+). Metric: **validation-split AUROC**, one seed (42).
Features: the 8 leakage-free predictors (strict leakage policy).

| pretrain | survey weight | best epoch | val loss | diabetes | prediabetes | obesity | hypertension | metab. syndrome | **mean** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| on  | on  | 19 | 0.5331 | 0.762 | 0.716 | 0.634 | 0.804 | 0.719 | **0.727** |
| on  | off | 19 | 0.5315 | 0.764 | 0.720 | 0.662 | 0.807 | 0.726 | **0.736** |
| off | on  | 16 | 0.5298 | 0.764 | 0.721 | 0.635 | 0.806 | 0.705 | **0.726** |
| off | off | 21 | 0.5314 | 0.763 | 0.718 | 0.675 | 0.807 | 0.722 | **0.737** |

**Marginal effect on mean val AUROC**
- pretraining (on − off), averaged over weight settings: **−0.0003**
- survey weighting (on − off), averaged over pretrain settings: **−0.0099**

## How to read this (caveats)

1. **Single seed, single val set.** The spread across variants (0.726–0.737) is
   within plausible run-to-run noise. No claim should be made without ≥5 seeds
   and error bars, plus held-out **test-set** numbers.

2. **The survey-weighting delta is largely a metric artifact.** AUROC on an
   *unweighted* validation set structurally favors an unweighted-trained model.
   Survey weighting should be judged with **survey-weighted metrics** and
   **subgroup calibration/fairness** (`metrics.py`), not raw val AUROC.

3. **Strict leakage removal undercuts the architecture's premise.** After
   dropping every label-defining measurement, 3 of 6 physiological groups
   (adiposity, glycemic, hemodynamic) are empty and only 8 features remain — so
   the physiology-structured layer and SSL pretraining have little structure to
   exploit. This is expected to blunt their benefit and is the key reason the
   novel ingredients look flat here. Revisit the leakage-strictness decision
   (e.g. per-head exclusion of only that label's own defining feature) before
   concluding the ideas don't help.

## Suggested next steps
- Multi-seed sweep (≥5) with mean ± std per cell.
- `metrics.py`: test-set AUROC/AUPRC + calibration, weighted, sliced by
  demographic subgroups.
- Leakage middle-ground experiment to repopulate the physiology groups.
- `xgboost_baseline.py` for a classical comparison.

Per-variant checkpoints/configs: `experiments/finetune/<variant>/`.


---

# Per-head (middle-ground) ablation — repopulated feature set

After restoring all 14 continuous features (6 physiology groups) and masking
each head's own label-defining features per forward pass (`finetune.py --per-head`).
Metric: **validation-split AUROC**, one seed (42).

| pretrain | survey weight | diabetes | prediabetes | obesity | hypertension | metab. syndrome | **mean** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| on  | on  | 0.796 | 0.711 | 0.724 | 0.829 | 0.838 | **0.780** |
| on  | off | 0.803 | 0.719 | 0.744 | 0.830 | 0.848 | **0.789** |
| off | on  | 0.785 | 0.713 | 0.718 | 0.827 | 0.844 | **0.777** |
| off | off | 0.807 | 0.702 | 0.739 | 0.827 | 0.852 | **0.785** |

**Best per-head config:** `pretrain-on / weight-off` (mean 0.789).

## Strict vs per-head (best config of each), val AUROC

| condition | strict | per-head | delta |
|---|---:|---:|---:|
| diabetes | 0.764 | 0.803 | +0.039 |
| prediabetes | 0.720 | 0.719 | -0.001 |
| obesity | 0.662 | 0.744 | +0.082 |
| hypertension | 0.807 | 0.830 | +0.023 |
| metabolic_syndrome | 0.726 | 0.848 | +0.122 |
| **mean** | **0.736** | **0.789** | **+0.053** |

The middle ground repopulates the physiology groups and recovers large amounts
of signal exactly where strict removal hurt most (metabolic syndrome, obesity),
while still preventing each head from seeing its own defining measurements.
Survey weighting still lowers raw val AUROC — see `test_metrics.md`, where it
wins on subgroup calibration, the metric that actually matters for it.

**Caveat, corrected by the multi-seed sweep below:** the single seed here (42)
happened to make pretraining look like a small positive. It isn't, on average —
see the next section.

---

# Multi-seed sweep — is any of this real, or noise?

The single-seed table above can't tell a real effect from a lucky/unlucky draw.
`finetune.py --n-seeds 10` reruns the same 4-way per-head ablation across 10
fixed seeds (0–9) — 40 runs total — and reports mean ± std val AUROC per variant
per condition. Full table: `results/seed_sweep.md`.

| variant | diabetes | prediabetes | obesity | hypertension | metab. syndrome | **mean** |
|---|---:|---:|---:|---:|---:|---:|
| pretrain-on / weight-on | 0.801 ± 0.006 | 0.701 ± 0.012 | 0.732 ± 0.006 | 0.824 ± 0.003 | 0.838 ± 0.005 | 0.779 |
| pretrain-on / weight-off | 0.802 ± 0.014 | 0.704 ± 0.026 | 0.727 ± 0.027 | 0.826 ± 0.006 | 0.843 ± 0.012 | 0.781 |
| pretrain-off / weight-on | 0.798 ± 0.005 | 0.716 ± 0.007 | 0.713 ± 0.010 | 0.826 ± 0.004 | 0.840 ± 0.005 | 0.779 |
| pretrain-off / weight-off | 0.805 ± 0.004 | 0.724 ± 0.007 | 0.731 ± 0.007 | 0.831 ± 0.002 | 0.849 ± 0.004 | **0.788** |

Marginal effects, per condition, checked against the noise floor (average
per-variant std across the 10 seeds):

| condition | weight-on − weight-off | pretrain-on − pretrain-off | noise floor | weight gap real? | pretrain gap real? |
|---|---:|---:|---:|:---:|:---:|
| diabetes | -0.004 | +0.000 | 0.007 | no | no |
| prediabetes | -0.006 | **-0.017** | 0.013 | no | **yes (hurts)** |
| obesity | -0.006 | +0.007 | 0.012 | no | no |
| hypertension | -0.003 | -0.003 | 0.004 | no | no |
| metabolic_syndrome | -0.006 | -0.004 | 0.006 | no | no |

**What this actually shows:**

1. **Pretraining does not help here, on average** — averaged across weight
   settings, mean AUROC is 0.780 with pretraining vs 0.784 without. The
   single-seed table's "pretraining wins" was noise from one lucky seed.
   Pretraining has exactly one *statistically distinguishable* effect across all
   40 runs, and it's negative: it **reliably hurts prediabetes** (-0.017, bigger
   than that condition's own seed-to-seed noise of 0.013). Every other
   condition's pretrain gap is smaller than its noise floor — not distinguishable
   from randomness at 10 seeds.
2. **Survey weighting's raw-AUROC cost is small and not individually
   significant, but consistent in direction** — weight-on scores lower than
   weight-off on all 5 conditions, every time, by 0.003–0006. No single
   condition's gap clears its own noise floor, but five-for-five same-direction
   is not what pure noise looks like either; call it a small, real, systematic
   cost. Judged on raw AUROC alone, unweighted training wins. It is *not*
   judged on raw AUROC alone — see `test_metrics.md`'s subgroup calibration
   comparison, which is the metric this project treats as authoritative for
   the weighting question, and which favors weighted training.
3. **Best mean AUROC variant across all 40 runs is `pretrain-off / weight-off`**
   (0.788) — plain per-head training, no pretraining, no survey weighting. That
   is a materially different conclusion than the single-seed table implied.
   Given point 2, the *deployed* choice should still weigh calibration, not
   just this table.

**Bottom line:** the per-head leakage fix (Section "Per-head (middle-ground)
ablation" above) is the one result in this whole ablation study that's large
enough to trust from a single seed (+0.05 mean AUROC, far past any plausible
noise floor here). Pretraining's benefit and survey-weighting's raw-AUROC cost
are both small enough that they needed the 10-seed sweep to even characterize
correctly — and pretraining's apparent benefit didn't survive it.

**Caveat for everything below:** the 10-seed sweep above was run on the
*2-cycle, 14-feature* per-head dataset, before the expansion in the next
section. It has not been re-run on the expanded 3-cycle/30-feature data — the
single-seed numbers below carry the same "could be noise" caveat the original
per-head table did before it got the multi-seed treatment.

---

# Expanded dataset — 3 cycles + 3 new modules (single seed 42)

The cohort was expanded from 2 cycles (2015-2018) to 3 (**2013-2014 added**,
NHANES suffix "H") and 3 new modules were merged in: **MCQ** (family history of
diabetes/heart attack: `MCQ300C`/`MCQ300A`), **DR1TOT** (Day-1 dietary recall
totals: energy, protein, carbs, sugar, fiber, fat, saturated fat, sodium — Day 2
was downloaded but not modeled, matching the existing "first reading only"
precedent for blood pressure), and **PAQ** (physical activity: sedentary
minutes/day + a derived intensity-weighted activity-minutes/week score). Two
new physiological groups: `lifestyle` (diet + activity) and `family_history`.
All 11 original modules' variable names were confirmed unchanged in the
2013-2014 codebook (no renames, no recoding) before merging — see project
memory. Cohort grew from 11,848 to **17,961 adults** (90 PSU clusters); splits
were regenerated from scratch, not appended, since the new cycle introduces
clusters the old split files never saw. Feature count: 14 continuous + 4
categorical -> **24 continuous + 6 categorical**.

| variant | diabetes | prediabetes | obesity | hypertension | metab. syndrome | **mean** |
|---|---:|---:|---:|---:|---:|---:|
| on / on   | 0.843 | 0.742 | 0.773 | 0.825 | 0.852 | 0.807 |
| on / off  | 0.853 | 0.746 | 0.775 | 0.830 | 0.868 | **0.814** |
| off / on  | 0.839 | 0.738 | 0.758 | 0.820 | 0.850 | 0.801 |
| off / off | 0.851 | 0.740 | 0.767 | 0.825 | 0.861 | 0.809 |

Best config unchanged: `pretrain-on / weight-off`.

## Val AUROC vs. every earlier version of this ablation (best config each time)

| condition | strict, 2-cycle, 8 feat. | per-head, 2-cycle, 14 feat. | per-head, 3-cycle, 30 feat. (**new**) |
|---|---:|---:|---:|
| diabetes | 0.764 | 0.803 | **0.853** |
| prediabetes | 0.720 | 0.719 | **0.746** |
| obesity | 0.662 | 0.744 | **0.775** |
| hypertension | 0.807 | 0.830 | **0.830** |
| metabolic_syndrome | 0.726 | 0.848 | **0.868** |
| **mean** | **0.736** | **0.789** | **0.814** |

Every condition improved or held steady versus the 2-cycle per-head numbers —
the extra ~6,100 adults and 16 new features (lifestyle + family history) added
real signal on top of the leakage-fix gain, most visibly for diabetes (+0.050)
and metabolic syndrome (+0.020). None of this is multi-seed-checked yet (see
caveat above), so treat the *size* of each condition's gain cautiously —
only the fact that per-head-on-more-data beats per-head-on-less-data is likely
to survive a seed sweep, not necessarily these exact deltas.
