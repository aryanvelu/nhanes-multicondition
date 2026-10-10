# NHANES Multi-Condition Prediction — Project Report

A plain-language explanation of what this project is, every term it uses, and
what each script does, in the order they run. If you read this top to bottom you
should understand where the project is and where it's going.

---

## 1. What are we building, in one paragraph?

We are training a single deep-learning model that looks at one person's health
profile — their age, sex, body measurements, a few blood-test results, and some
questionnaire answers — and predicts, all at once, whether they have (or are at
risk of) **five** chronic conditions: **diabetes, prediabetes, obesity, high
blood pressure, and metabolic syndrome**. The data comes from a big, free,
public US health survey called **NHANES**. The project tries to be genuinely
useful and novel (not just a stock classifier) through four design choices,
explained in Section 3.

---

## 2. Glossary — every term, defined

**Deep learning** — a family of machine-learning models ("neural networks")
made of layers of simple math units that learn patterns from examples.

**Neural network** — the model itself: numbers ("weights") arranged in
connected layers that transform inputs into outputs. Training = nudging the
weights until the outputs are good.

**Feature** — one input column describing a person (e.g. age, cholesterol).

**Label** — the answer we want to predict (e.g. "has diabetes: yes/no").

**Binary label** — a yes/no (1/0) label.

**Multi-label** — predicting several yes/no labels for the same person at once
(here, five conditions), rather than picking one class. A person can be positive
for several conditions simultaneously.

**NHANES** (National Health and Nutrition Examination Survey) — a long-running
US survey run by the CDC that interviews *and* physically examines a sample of
Americans, then publishes the (de-identified) data for free. Released in 2-year
"cycles" (we use **2013–2014, 2015–2016, and 2017–2018**).

**CDC** — US Centers for Disease Control and Prevention, who run NHANES.

**Cycle** — one 2-year batch of NHANES data.

**Module** — one NHANES data file covering one topic (e.g. `BMX` = body
measurements, `GLU` = fasting glucose). We download several and join them.

**SEQN** — the participant ID number in NHANES. **Important quirk:** SEQN is
only unique *within* a cycle; the same number means different people in
different cycles, so we always keep a `cycle` column alongside it.

**.XPT file** — the SAS "transport" file format the CDC publishes data in. We
read it with the `pyreadstat` library.

**Survey weight** — a number attached to each participant saying "how many
people in the real US population this person represents." NHANES deliberately
over-samples some groups, so weights are needed to get nationally correct
statistics. Column names: `WTINT2YR` (interview weight), `WTMEC2YR` (exam
weight), `WTSAF2YR` (fasting-subsample weight).

**Subsample** — a subset of participants who got an extra test. Key one here:
the **fasting subsample** — only people who came in the morning after not eating
got fasting-glucose and triglyceride blood draws (~6,000 of ~19,000). This is
why many lab columns are blank for most people.

**Fasting glucose** — blood sugar measured after not eating; a diabetes test.
Column `LBXGLU`, in mg/dL (milligrams per deciliter).

**HbA1c / glycohemoglobin** — a blood test reflecting average blood sugar over
~3 months. Column `LBXGH`, in %.

**Triglycerides** — a type of blood fat. Column `LBXTR`. **HDL** — "good"
cholesterol, column `LBDHDD`. **LDL** — "bad" cholesterol, `LBDLDL`. **Total
cholesterol** — `LBXTC`.

**BMI** (body mass index) — weight ÷ height², a standard obesity measure.
Column `BMXBMI`. **Waist circumference** — `BMXWAIST`.

**Systolic / diastolic blood pressure** — the two numbers in a BP reading
(e.g. 120/80). Columns `BPXSY1` / `BPXDI1`.

**Metabolic syndrome** — not one disease but a *cluster*: you have it if you
meet 3 of 5 risk criteria (big waist, high triglycerides, low HDL, high BP, high
fasting glucose). We use the standard **NCEP ATP III** definition (a widely-used
clinical rule set).

**Clinical threshold** — the cutoff value doctors use to call something
abnormal (e.g. fasting glucose ≥ 126 = diabetes).

**Family history** — whether a close blood relative had a condition. NHANES
asks this directly (`MCQ300A` = relative had a heart attack, `MCQ300C` =
relative had diabetes); it's a genuine risk factor, not a measurement of the
participant themselves, so it can never leak a label.

**24-hour dietary recall** — an interviewer asks everything the participant ate
in the last 24 hours, then software totals up nutrients. We use the Day-1
totals (`DR1TOT` module: calories, protein, carbs, sugar, fiber, fat, saturated
fat, sodium), discarded if NHANES itself flagged the recall as unreliable or
incomplete (`DR1DRSTZ != 1`).

**Physical Activity Questionnaire (PAQ / GPAQ)** — a standard WHO
questionnaire NHANES uses: do you do vigorous/moderate work, walk or bike
places, or do vigorous/moderate recreational exercise, and if so how often and
for how long. We combine the answers into one number, **intensity-weighted
activity minutes/week** (vigorous activities count double, matching the public
"150 min moderate or 75 min vigorous per week" guideline), plus sedentary
minutes/day on its own.

**Prevalence** — the fraction of people who have a condition. We print these to
sanity-check our labels against known medical statistics.

**One-hot encoding** — turning a category into a row of 0s with a single 1
(e.g. "race = 3 of 5" → `0 0 1 0 0`). Simple but wide and treats categories as
unrelated.

**Embedding** — a *learned* short list of numbers that stands in for a category.
Instead of one-hot, the model learns, say, a 3-number vector per race group, and
can discover that some groups behave similarly. More compact and more powerful.

**Categorical feature** — a feature that is a category/code (sex, race,
education, smoking status). Fed to the model as embeddings.

**Continuous feature** — a numeric feature that varies smoothly (age,
cholesterol). Fed to the model as (standardized) numbers.

**Standardization (z-score)** — rescaling a numeric column to have average 0 and
spread 1, so features on different scales (age ~0–80 vs cholesterol ~100–300)
are comparable to the network. Formula: (value − mean) ÷ standard deviation.

**Missing value** — a blank cell (person didn't get that test). **Imputation** —
filling blanks with a stand-in number so the math works. **Missingness flag** —
an extra 0/1 column we add *before* filling, recording "this was blank." In
NHANES, *what* is missing is itself informative (e.g. missing fasting glucose
usually means "not in the fasting subsample"), so we keep that as a signal.

**Physiological system / group** — a body subsystem we group features by:
*glycemic* (blood sugar), *lipid* (blood fats), *adiposity* (body fat/size),
*hemodynamic* (blood pressure), *demographic* (age, sex, income, etc.),
*behavioral* (smoking), *lifestyle* (diet, physical activity, alcohol, sleep),
*family_history* (relative had diabetes/heart attack), *weight_history*
(self-reported past weight), *renal* (kidney function), *hepatic* (liver
function), and *medication_use* (prescription drug count). Used by our
special first layer (Section 3).

**Kidney function markers** — **creatinine** (`LBXSCR`) and **BUN**, blood
urea nitrogen (`LBXSBU`): waste products the kidneys filter out of blood;
both rise when kidney function declines.

**Liver function markers** — **ALT** and **AST** (enzymes that leak into
blood when liver cells are damaged), **ALP** and **GGT** (elevated with bile-
duct problems or heavy alcohol use), **bilirubin** (a breakdown product the
liver processes — high levels cause jaundice), **albumin** (a protein the
liver makes — low levels can mean liver damage or malnutrition).

**Polypharmacy** — taking many medications at once; a common marker of
overall disease burden in health research. We use it as a simple count
(`RXDCOUNT`, aggregated from a file listing one row per medication a person
takes) rather than tracking which specific drugs.

**Leakage** — when an input feature basically *gives away* the label, making the
task fake-easy and the model useless in practice. Example: using BMI to predict
obesity, when obesity is *defined* as BMI ≥ 30. We remove these (Section 4).

**Train / validation / test split** — we divide people into three groups: the
model learns on **train**, we tune/early-stop on **validation**, and we get an
honest final score on **test** (which the model never saw).

**Cluster / PSU / stratum** — NHANES doesn't sample people independently; it
picks a few geographic areas (**PSUs**, "primary sampling units," column
`SDMVPSU`) inside **strata** (groups of areas, column `SDMVSTRA`) and surveys
people there. People in the same PSU are correlated. So when splitting, we keep
each whole PSU on one side — otherwise the model could "cheat" by recognizing a
cluster it saw in training. This is a **cluster-aware split**.

**Self-supervised pretraining** — training the model on a task that needs *no
labels*, to warm it up before the real task. Our task: hide some of a person's
features and make the model guess them back ("masked-feature reconstruction").

**Masking** — hiding chosen inputs on purpose so the model must predict them.

**Backbone** — the shared core network that converts a person's features into a
compact vector (the **embedding**). Different "heads" attach to the same backbone
for different jobs.

**Head** — a small output network bolted onto the backbone. The *pretraining
head* reconstructs hidden features; later, five *classification heads* each
output one condition's probability.

**MSE (mean squared error)** — a loss for numbers: average of (predicted −
true)². Used to score reconstructed continuous features.

**Cross-entropy (CE)** — the standard loss for predicting a category; low when
the model puts high probability on the correct class. Used for categorical
features and, later, for the yes/no condition heads.

**Loss** — a single number measuring how wrong the model is; training tries to
make it small.

**Epoch** — one full pass over the training data. **Batch / mini-batch** — a
small chunk of examples processed at once. **Batch size** — how many.

**Early stopping** — stop training when the validation loss stops improving, to
avoid **overfitting** (memorizing the training data instead of learning general
patterns).

**BatchNorm (batch normalization)** — a layer that re-scales activations using
the current batch's statistics; makes training faster and steadier.

**Dropout** — randomly switching off some units during training so the model
doesn't over-rely on any one of them; reduces overfitting.

**Token / tokenization** — turning one piece of input into its own small
learned vector. Normally used for words in a sentence; here, every *feature*
(age, BMI, sex, etc.) becomes its own token, so the model can pay attention
to feature-to-feature relationships directly.

**Self-attention / Transformer** — the architecture behind BERT/GPT: every
token looks at every other token and learns how much to "attend to" each one,
letting the model discover relationships between inputs on its own instead of
a human hard-coding which features belong together.

**`[CLS]` token** — an extra learned token added to the sequence purely to
collect a summary of everything else after attention runs; its final
representation is used as "the answer" for downstream predictions, the same
trick BERT uses to summarize a whole sentence in one vector.

**MPS** — Apple's "Metal Performance Shaders," the way PyTorch uses the GPU on
Apple Silicon Macs. `torch.device("mps")`. No NVIDIA/CUDA card needed.

**PyTorch (`torch`)** — the deep-learning library we build the model in.

**pandas / parquet / JSON** — pandas is the table-handling library; `.parquet`
is an efficient table file format; `.json` is a simple text format for storing
structured settings/metadata.

**AUROC / AUPRC / calibration** — evaluation metrics (used in Section 6):
AUROC = how well the model ranks sick above healthy (0.5 = coin flip, 1.0 =
perfect); AUPRC = same idea but focused on the positive class (good for rare
conditions; its "chance" level is the prevalence); **calibration** = do predicted
probabilities match reality — if the model says "40% risk" for a group of people,
do about 40% of them actually have the condition? The **calibration gap** is the
difference between average predicted risk and actual observed rate.

**Subgroup** — a slice of people sharing a trait (e.g. one race/ethnicity group,
or an income band). We check calibration *within* subgroups to see if the model
is equally trustworthy for everyone, not just on average.

**SMOTE** (Synthetic Minority Over-sampling TEchnique) — a way to fix a class
imbalance (too few positives) by *manufacturing* new synthetic positive
examples — interpolating between real positive examples' feature values —
rather than just duplicating existing rows. **SMOTENC** is the variant for
data with both continuous and categorical columns (ours has both): it
mode-selects categorical/missing-flag values from nearest neighbors instead of
producing nonsensical fractional category codes.

**Weighted vs unweighted metric** — an *unweighted* metric scores the model on
the survey sample as-is; a *survey-weighted* metric scores it as it would behave
on the real US population (each person counted by their survey weight). Survey
weighting during training is judged fairly by weighted calibration, not by raw
unweighted AUROC.

---

## 3. What makes this more than a plain classifier (the novel bits)

1. **Physiology-structured input layer.** Normally the first layer connects
   every input to every unit, and the network must *figure out* which features
   relate. Instead, we group features by body system and give each group its own
   private units with **no cross-group wiring in the first layer**. We bake in
   medical knowledge instead of hoping the model learns it.

2. **Self-supervised pretraining.** We first train on the hide-and-reconstruct
   task using *everyone* in the training split (labels not required), so the
   backbone learns how health variables relate before it ever sees a diagnosis.
   This helps because the fully-labelled group is small (~6k with fasting labs).

3. **Survey-weighted training.** Because NHANES over-samples some groups, we
   weight the training loss by each person's survey weight so the model reflects
   the real US population, not the survey's sampling quirks. Judged fairly by
   *calibration* across subgroups, not raw AUROC (Section 6).

4. **Four-way ablation, multi-seed.** We train the model with/without survey
   weighting and with/without pretraining (**ablation** = turning pieces off to
   measure their effect), repeated across 10 random seeds so we can tell a real
   effect from run-to-run noise — and compare against a tuned **XGBoost**
   baseline (a strong classical "gradient-boosted trees" model, the usual
   benchmark to beat on tabular data) to show the deep model earns its
   complexity.

---

## 4. The five conditions and how we label them

Computed in `labels.py` using standard clinical thresholds:

| Condition | Positive if… |
|---|---|
| **Diabetes** | fasting glucose ≥ 126, or HbA1c ≥ 6.5%, or self-reported diagnosis |
| **Prediabetes** | glucose 100–125, or HbA1c 5.7–6.4, or self-reported "borderline" (and not already diabetic) |
| **Obesity** | BMI ≥ 30 |
| **Hypertension** | BP ≥ 130/80, or on blood-pressure medication |
| **Metabolic syndrome** | 3+ of: big waist, triglycerides ≥ 150, low HDL, BP ≥ 130/85, glucose ≥ 100 (NCEP ATP III) |

**Validity mask.** Because many people lack the needed tests, each label also
gets a `_valid` companion column: 1 = "we could actually compute this for this
person," 0 = "not enough data." Diabetes/prediabetes are computable for
~28,200 people, of 29,400 total across all 3 cycles (self-report or HbA1c cover
almost everyone); metabolic syndrome needs the full fasting panel so it's only
valid for ~7,700. We keep everyone and let the model **ignore (mask) a label's
loss where it isn't valid**, instead of throwing away people with missing tests.

**Sanity check (prevalences we got):** diabetes ~11%, prediabetes ~22%,
obesity ~28%, hypertension ~42%, metabolic syndrome ~34% (of the fasting group).
These line up with published US figures — in particular metabolic syndrome ~34–36%
in adults — so the labels look medically reasonable.

### The leakage fix (important)

The measurements used to *define* the labels were also sitting in the model's
input features — e.g. BMI (defines obesity), BP (defines hypertension), glucose/
HbA1c (define diabetes), the waist/triglyceride/HDL panel (defines metabolic
syndrome). Predicting obesity from BMI is circular and would be dismissed by
reviewers. `remove_leakage.py` strips every such column out (plus raw weight and
height, which together rebuild BMI). It gets the "which columns" list *directly
from `labels.py`* so the two can never disagree.

**Consequence to know:** after removal, **20 predictive features** remain (age,
income, race, sex, education, smoking, total cholesterol, LDL, plus the newer
diet/activity/family-history features — none of which any label is defined
from), while three physiological groups (adiposity, glycemic, hemodynamic)
still become empty. That makes the prediction genuinely hard and honest for
those three groups' conditions — the model must find real risk signals rather
than read the answer off the input.

### 4b. The middle-ground fix — per-head leakage masking (the better answer)

Strict removal is *too* blunt: it deletes a feature for **every** head just
because it defines **one** label. Blood pressure gives away hypertension, but
it's a perfectly fair (non-leaking) input when predicting diabetes. The
middle-ground keeps all features and masks leakage **per head** instead:

- The backbone runs **once per condition**, and before each run we hide only
  that condition's own defining features (value zeroed, missing-flag set), while
  every other group stays visible. Same shared backbone weights every time.
- So diabetes never sees glucose/HbA1c but does see BMI, blood pressure, and
  lipids; obesity never sees any body-size feature but sees everything else;
  metabolic syndrome (defined across most systems) ends up masked the most.

This repopulates all eight physiological groups, giving the structured layer
and pretraining real structure to work with. It measurably beats strict removal
(mean validation AUROC 0.79 vs 0.74; metabolic syndrome +0.12, obesity +0.08),
while still preventing any head from reading its own answer. This is the version
the results below use. `lifestyle`, `family_history`, and `weight_history` are
never masked for any head — nothing in any of the three groups defines any of
the 5 labels — so every condition sees diet, activity, family-history, and
self-reported-weight features unconditionally.

**Update — the flagged risk materialized, and it's `WHD050`, not `WHD020`.**
`WHD020` (current self-reported weight) was masked from the obesity head
(same logic as `BMXWT`/`BMXHT`), leaving `WHD050` (weight a year ago) and
`WHD120` (weight at age 25) visible to every head including obesity, on the
reasoning that *history* isn't a restatement of *current state*. Re-running
the FT-Transformer (no pretraining, 10 seeds) on this feature set produced a
result that doesn't pass the smell test: **obesity AUROC jumped from 0.775 to
0.935 (+0.160, far past the 0.003 noise floor) while every other
condition stayed flat** (diabetes -0.002, prediabetes +0.001, hypertension
-0.000, metabolic_syndrome +0.002 — none real). A genuine improvement in
input richness should help several heads somewhat; an isolated ~50x-larger
effect on exactly the one head sitting next to the new features is the
signature of a proxy leak, not a modeling win. Checking why:
`WHD050`-to-obesity-label correlation is **0.556** — nearly as strong as
`WHD020`'s own 0.599 (the one we already excluded) — because most adults'
weight doesn't change much in a single year, so "what did you weigh 12
months ago" is nearly as informative as "what do you weigh now." `WHD120`
(weight at age 25, correlation 0.275 — decades of possible change intervene)
looks much more like genuine history and not implicated by these numbers.
**This directly contradicts the assumption "history isn't circular" for
`WHD050` specifically.** Recommendation given to the user (reversing part of
an earlier explicit instruction based on this new evidence) — **applied**:
`WHD050` now masked alongside `WHD020` for the obesity head;
`WHD120` remains the one weight-history feature obesity can legitimately see.

**After the fix, the picture is sane again.** Re-running the same 10-seed
FT-Transformer comparison with the corrected mask (`results/ft_whq_sweep.md`):

| condition | before (no WHQ) | after (+ WHQ, corrected mask) | delta |
|---|---:|---:|---:|
| diabetes | 0.855 | 0.854 | -0.001 |
| prediabetes | 0.750 | 0.754 | +0.004 (real) |
| **obesity** | 0.775 | **0.834** | **+0.059 (real)** |
| hypertension | 0.830 | 0.830 | -0.000 |
| metabolic_syndrome | 0.869 | 0.870 | +0.001 |
| **mean** | **0.816** | **0.828** | **+0.013** |

Obesity's gain shrank from the suspicious +0.160 down to a much more
plausible **+0.059**, still real (past its 0.003 noise floor) but now
consistent with what a single genuinely-informative feature (`WHD120`,
correlation 0.275 with the label — stronger than most other individual
predictors, but nowhere near `BMXBMI`'s 0.781 or the excluded proxies'
~0.6) could reasonably contribute, rather than a near-restatement of the
answer. Prediabetes also gained a small, real +0.004; the other three
conditions (which don't share as direct a connection to weight/diet history)
are unaffected, which is itself a good sign — a real signal should be
selective, not uniform. This is the trustworthy number for the WHQ addition.

---

## 5. The scripts, in run order

Run each as a module from the project root, e.g. `python -m src.data.download`.

| # | Script | Plain-language job | Key output |
|---|---|---|---|
| 1 | `src/data/download.py` | Download the raw NHANES `.XPT` files from the CDC (no login) for 3 cycles (2013-14, 2015-16, 2017-18) x 16 modules, and confirm each loads (prints rows × columns). | `data/raw/<cycle>/*.XPT` |
| 2 | `src/data/merge.py` | Join all modules for each person by `SEQN`, keeping cycles separate. Uses **left joins** onto demographics so people missing a lab become blanks, not deletions. Prints where people drop out. | `data/processed/merged.parquet` |
| 3 | `src/data/labels.py` | Compute the five yes/no labels + their validity masks from clinical thresholds. Prints prevalences. | `data/processed/labels.parquet` |
| 4 | `src/data/preprocess.py` | Standardize continuous features, keep categoricals as integer codes (for embeddings, not one-hot), add a missingness flag per feature *before* imputing, and record everything (including each feature's physiological group) in a schema file. Also derives `smoking_status` and `activity_min_per_week`, and nulls out unreliable dietary recalls. | `features.parquet`, `feature_schema.json` |
| 5a | `src/data/remove_leakage.py` | **Strict** leakage fix: drop every label-defining feature globally (see Section 4). | `features_no_leakage.parquet`, `feature_schema_no_leakage.json` |
| 5b | `src/data/perhead_leakage.py` | **Middle-ground** leakage fix: keep all features, but record a per-head mask so each condition's head hides only *its own* defining features (Section 4b). | `feature_schema_perhead.json` |
| 6 | `src/data/splits.py` | Make 70/15/15 train/val/test splits **by whole PSU cluster** (so no cluster is split across sets), on the **adult (18+)** cohort. Prints per-split prevalences to confirm they're balanced. | `data/splits/{train,val,test}.csv` |
| 7 | `src/models/structured_layer.py` | Defines the physiology-structured first layer and the exact input format (the three aligned arrays: values, missing-flags, category-codes). Not run directly — imported. | *(module)* |
| 8 | `src/models/backbone.py` | Assembles the reusable backbone: structured layer → 2–3 fully-connected layers with BatchNorm + Dropout → shared embedding. Imported by every training script. | *(module)* |
| 9 | `src/training/pretrain.py` | Self-supervised pretraining: hide some features, reconstruct them (MSE for numbers, cross-entropy for categories), early-stop on validation loss. Saves just the backbone weights. | `experiments/pretrain/backbone.pt` |
| 10 | `src/training/finetune.py` | Attach the five condition heads to the (pretrained) backbone; train with **valid-masked, survey-weighted** binary cross-entropy on the adult training split; early-stop on validation loss; report per-condition AUROC. `--no-pretrain` / `--no-survey-weight` run the ablation variants; `--per-head` runs the middle-ground mode (backbone runs once per head with that head's features masked); `--n-seeds N` re-runs all four variants across N fixed seeds and reports mean ± std per condition, to separate real effects from noise. Each variant saved to its own subfolder. | `experiments/finetune/{strict,perhead}/<variant>/model.pt`, `results/seed_sweep.md` |
| 11 | `src/evaluation/metrics.py` | Evaluate the best per-head config on the **held-out test set**: per-condition AUROC/AUPRC (unweighted + survey-weighted), then **subgroup calibration** (race, income) comparing the survey-weighted vs unweighted model — the real test of whether weighting was worth it. | `results/test_metrics.md` |
| 12 | `src/baselines/xgboost_baseline.py` | Classical comparison: one XGBoost tree ensemble per condition, given the *same* per-head-masked features, splits, validity masks, and survey-weighted metrics as the deep model, so the comparison is apples-to-apples. | `results/xgboost_baseline.md` |
| 12b | `src/baselines/lightgbm_baseline.py` | Second classical opinion: same setup as the XGBoost baseline, swapping in LightGBM, three-way table (deep/XGBoost/LightGBM). | `results/lightgbm_baseline.md` |
| 13 | `src/evaluation/stacking.py` | Fit a 2-input logistic-regression meta-model per condition on `[deep_prob, xgb_prob]` from the **validation** split, score the blend on test. | `results/stacking.md` |
| 14 | `src/training/hyperparam_search.py` | Random search over each model's key hyperparameters (deep: lr/width/dropout; XGBoost: max_depth/lr/n_estimators), picked by mean val AUROC, confirmed on test. | `results/hyperparam_search.md` |
| 15 | `src/training/smote.py` + `finetune.py --smote[-seeds N]` | SMOTE-balance prediabetes & metabolic_syndrome in the **training split only**; other heads ignore synthetic rows via `_valid=0`. `--smote-seeds N` compares on/off across N seeds. | `results/smote_sweep.md` |
| 16 | `src/models/ft_transformer.py` + `src/training/architecture_sweep.py` | Alternative backbone: per-feature tokens + `[CLS]` + self-attention instead of grouped-sparse MLP. Compared against the grouped-MLP across 10 seeds. | `results/architecture_sweep.md` |
| 17 | `src/training/pretrain_ft.py` + `src/training/ft_pretrain_sweep.py` | Masked-feature-**token** pretraining for the FT-Transformer (post-tokenization `[MASK]`, not pre-tokenization flat-vector masking). Pretrained vs. random-init compared across 10 seeds, val + test. | `experiments/pretrain_ft_transformer/backbone.pt`, `results/ft_pretrain_sweep.md` |
| 18 | `src/data/perhead_leakage.py` (`WHD020`+`WHD050`→obesity mask) + `src/training/ft_whq_sweep.py` | Masks both `WHD020` and `WHD050` from the obesity head's per-head mask (self-report proxies, same logic as `BMXWT`/`BMXHT` — `WHD050` added after an initial run surfaced a proxy-leakage red flag); re-runs FT-Transformer with `weight_history` in play. | `results/ft_whq_sweep.md` |
| 19 | `src/training/ft_organlab_sweep.py` | Adds kidney/liver labs (`BIOPRO`), alcohol (`ALQ`), sleep (`SLQ`), and medications (`RXQ_RX`, aggregated separately — one row per drug, not per person); re-runs FT-Transformer, checked for the same isolated-jump proxy-leak signature that caught `WHD050`. | `results/ft_organlab_sweep.md` |
| 20 | `src/training/train_ft_final.py` | Trains and saves the one deployable FT-Transformer checkpoint (seed 42, no pretrain, weight-off) on the current fully-enriched schema — the 10-seed sweeps never persist checkpoints. | `experiments/finetune/ft_transformer/model.pt` |

Note: `splits.py` filters the cohort to **adults (RIDAGEYR ≥ 18)** before
assigning clusters. This is safe to do at any stage — because a whole PSU cluster
goes to one split, dropping individuals can never move or split a cluster — but
doing it here keeps the 70/15/15 proportions honest for the adult cohort and
keeps pretraining and fine-tuning on the same people.

---

## 6. Where the project stands right now

**Done and verified end-to-end:** download (3 cycles x 16 modules) → merge →
labels → preprocess → leakage control (strict **and** per-head) → splits
(adults, regenerated from scratch for the 3-cycle cohort) → backbone →
self-supervised pretraining → supervised fine-tuning → four-way ablation →
test-set evaluation with subgroup calibration → XGBoost baseline. Everything
trains on Apple-Silicon GPU (MPS).

**Data scale:** the cohort now spans **3 NHANES cycles** (2013-14, 2015-16,
2017-18) after adding the `MCQ` (family history), `DR1TOT`/`DR2TOT` (diet), and
`PAQ` (physical activity) modules; 29,400 people total, adult cohort (18+) =
**17,961** (~12,600 train / ~2,700 val / ~2,700 test, 90 PSU clusters). Per-head
model uses **24 continuous + 6 categorical features across 8 physiological
groups** (up from 14+4 across 6 groups) — the two new groups are `lifestyle`
(diet + activity) and `family_history`.

**Since superseded:** the numbers immediately below (test-set table, and the
"vs. every earlier version" comparison) reflect the 24+6/8-group,
grouped-MLP-backbone snapshot and are kept for historical comparison. The
project has since moved past this in two ways that matter more: the
**FT-Transformer architecture** (a real, multi-seed-confirmed win) and **four
rounds of new feature modules** (`weight_history`, then kidney/liver labs +
alcohol + sleep + medications), each individually leakage-checked and
empirically verified against proxy leaks. See "Where things stand now" below
the FT-Transformer discussion for the current, trustworthy state of the art —
**mean val AUROC 0.839**, up from the 0.816 this section's numbers reflect.

**Test-set results — per-head model, survey-weighted AUROC / AUPRC** (honest,
leakage-controlled, on the expanded 3-cycle/30-feature data; from
`results/test_metrics.md`):

| condition | AUROC | AUPRC |
|---|---|---|
| metabolic syndrome | 0.87 | 0.76 |
| diabetes | 0.86 | 0.46 |
| obesity | 0.79 | 0.67 |
| prediabetes | 0.75 | 0.57 |
| hypertension | 0.82 | 0.79 |

**Validation AUROC vs. every earlier version of this experiment** (best config
each time, `pretrain-on / weight-off`; full table in `results/ablation.md`):

| condition | strict, 2-cycle, 8 feat. | per-head, 2-cycle, 14 feat. | per-head, 3-cycle, 30 feat. (now) |
|---|---:|---:|---:|
| diabetes | 0.764 | 0.803 | **0.853** |
| prediabetes | 0.720 | 0.719 | **0.746** |
| obesity | 0.662 | 0.744 | **0.775** |
| hypertension | 0.807 | 0.830 | **0.830** |
| metabolic_syndrome | 0.726 | 0.848 | **0.868** |
| **mean** | **0.736** | **0.789** | **0.814** |

Every condition improved or held steady after adding the extra cycle and the
16 new lifestyle/family-history features, on top of the earlier per-head
leakage-fix gain — most visibly diabetes (+0.05) and metabolic syndrome
(+0.02) versus the 2-cycle per-head numbers.

**Did the novel ingredients help?** (Checked properly on the *2-cycle* dataset:
single-seed numbers can be lucky or unlucky, so `finetune.py --n-seeds 10`
reran the whole 4-way per-head ablation across 10 fixed seeds — 40 training
runs — and reported mean ± std per condition. Full table:
`results/seed_sweep.md`, discussion in `results/ablation.md`. **This multi-seed
check has not yet been redone on the 3-cycle/30-feature data** — treat the new
single-seed deltas above with the same caution the 2-cycle numbers needed
before their seed sweep.)
- *Per-head leakage masking* — clearly yes, and this is the one result solid
  enough to trust from a single seed: mean val AUROC 0.79 vs 0.74 for strict
  removal on the 2-cycle data, biggest gains on metabolic syndrome (+0.12) and
  obesity (+0.08) — far larger than any noise floor seen in the 10-seed sweep.
- *Self-supervised pretraining* — **does not help, on average**, on the 2-cycle
  data the sweep was run on: mean AUROC is actually *higher without*
  pretraining (0.784 vs 0.780). The one effect that *is* statistically
  distinguishable from seed noise is negative: pretraining reliably **hurts
  prediabetes** prediction (-0.017, bigger than that condition's own
  run-to-run noise of 0.013). This corrected an earlier (single-seed)
  conclusion that it helped — a reminder to re-check once the 3-cycle sweep
  is run, since the single-seed 3-cycle table above still shows pretraining
  "on" winning.
- *Survey weighting* — the calibration verdict is **less clear-cut on the
  bigger, more diverse cohort.** On the original 2-cycle test set it gave more
  consistent calibration on *both* race and income (2/2). Re-run on the
  3-cycle test set, it now **only wins on income** (mean gap 0.034 weighted vs
  0.036 unweighted) and **loses on race** (0.040 weighted vs 0.037
  unweighted) — 1/2. Adding the older 2013-2014 cycle changed the population
  mix enough to shift this; it's not a free pass to conclude weighting no
  longer matters, but it is no longer a clean 2/2 win. Worth re-examining if
  this becomes an important claim for a paper.

**Does a classical model do just as well?** `results/xgboost_baseline.md`
trains one XGBoost tree ensemble per condition on the identical per-head-masked
features, splits, and metrics as the deep model (both unweighted, matching the
deep model's best-AUROC config), re-run on the expanded 3-cycle data. Result:
**still essentially a tie** on survey-weighted test AUROC — mean 0.816 (deep)
vs 0.817 (XGBoost), deep model ahead on 3/5 conditions (diabetes, obesity,
hypertension) and behind on 2/5 (prediabetes, metabolic syndrome), all by
≤0.012. Notably, the tie did **not** resolve itself with more data — both
models' test AUROC actually moved slightly *down* together versus the 2-cycle
numbers (0.823 → ~0.816), most likely reflecting a harder or more
heterogeneous test population after adding the 2013-2014 cycle, not a
regression in either model. The architecture-vs-classical-model conclusion is
unchanged: at this scale, raw discrimination is a wash: the deep model's case
still rests on the shared multi-task backbone and (partially) on
survey-weighting's calibration benefit, not on beating a well-tuned classical
model at ranking people.

**Is that a coincidence, or would any second classical model land here too?**
`results/lightgbm_baseline.md` runs LightGBM through the identical setup as
the XGBoost baseline — same per-head features, splits, metrics. It lands at
the **same place**: mean AUROC 0.817, matching XGBoost's 0.817 almost exactly
(diabetes 0.851, prediabetes 0.752 — LightGBM's best condition, obesity 0.785,
hypertension 0.813, metabolic syndrome 0.883 — also LightGBM's best). Deep
model still wins 3/5 conditions outright. **Three architecturally unrelated
model families — a physiology-structured neural net, and two different
gradient-boosting implementations — now agree on the same ~0.82 ceiling.**
That kind of convergence across genuinely different model classes is the
strongest evidence yet that this is a **data ceiling** for this leakage-safe
feature set, not a model-search problem: no fourth architecture is likely to
push mean AUROC toward 0.9 without richer input signal.

**Does blending the two models beat either one alone?** Since the tie means
neither model dominates, `src/evaluation/stacking.py` fits a tiny
**stacking ensemble** per condition: a 2-input logistic regression,
`P = sigmoid(a·deep_prob + b·xgb_prob + c)`, fit on **validation-split**
predictions from both base models (never on data either model actually
trained its weights on) and scored on test. Result
(`results/stacking.md`): the stack **beats both individual models on
AUROC in 4 of 5 conditions** (diabetes, prediabetes, obesity, hypertension —
mean AUROC 0.821 vs 0.816 deep / 0.817 XGBoost), losing only on metabolic
syndrome, where XGBoost alone (0.882) is hard to beat because its coefficient
is barely larger than the deep model's there too. Every condition's
logistic-regression coefficients came out positive and reasonably balanced
between the two models (e.g. diabetes: 3.16 deep / 3.10 xgb), confirming the
blend is genuinely combining two different sources of signal rather than
just parroting one model. **This is the strongest result in the project's
discrimination metrics so far** — for a "give clinicians a probability"
model, ship the stack, not either base model alone.

**Does tuning the hyperparameters change any of this?** `src/training/
hyperparam_search.py` ran a 10-trial random search from each model's
established winning config — deep model: learning rate / hidden-layer width
/ dropout; XGBoost: `max_depth` / `learning_rate` / `n_estimators` — selected
by mean unweighted validation AUROC, confirmed on test
(`results/hyperparam_search.md`). **Neither search moved the needle**: the
best deep config found was lr=0.001, width=64, dropout=0.2 — exactly the
defaults already in use — and every trial that changed the hidden width lost
access to the pretrained backbone (shape mismatch) and scored worse purely
from losing pretraining, not from the width itself. The best XGBoost config
was max_depth=4/lr=0.05/n_estimators=1000 — also exactly the existing
baseline, tied with two other configs at the same validation score. Tuned
test AUROC came back essentially identical to the untuned numbers for both
models (deep mean 0.816, XGBoost mean 0.817 — the same tie as before). This
is a legitimate, useful negative result: the hand-picked starting
configuration for both models was already close to a local optimum in this
search space, so the earlier reported numbers aren't an artifact of lazy
defaults.

**Does SMOTE oversampling help the two most imbalanced heads?**
`src/training/smote.py` SMOTE-balances **prediabetes and metabolic_syndrome**
specifically (the two conditions named in the request) to parity in the
**training split only** — val/test are never touched. Because all 5 heads
share one backbone, a synthetic row can't have real labels for the other 4
conditions, so it's given `_valid=0` there and only ever contributes to the
loss of the condition it was generated for. Checked the same way as every
other effect in this project: `finetune.py --smote-seeds 10` reruns SMOTE-on
vs SMOTE-off across the same 10 fixed seeds as the earlier sweep, at the
established best config. Result (`results/smote_sweep.md`): **SMOTE makes
things worse, and the effect survives the noise floor for 4 of 5
conditions** — prediabetes -0.010 (real, noise floor 0.006), metabolic_syndrome
-0.011 (real, noise floor 0.005), and it even drags down two *un-SMOTE'd*
conditions sharing the same backbone: diabetes -0.004 (real) and hypertension
-0.004 (real). Only obesity's -0.009 gap sits right at its noise floor
(0.009), too close to call. The likely explanation: neither target condition
was severely imbalanced to begin with (prediabetes ~31% positive,
metabolic_syndrome ~40% of its valid subsample), so SMOTE's synthetic rows
mostly add interpolation noise to the shared backbone without a real
imbalance problem to fix — and because the backbone is shared, that noise
leaks into every other head's performance too. **Verdict: SMOTE is not
adopted.**

**A second classical opinion: LightGBM.** `results/lightgbm_baseline.md` runs
LightGBM through the exact same per-head setup as the XGBoost baseline. It
lands in the same place: mean survey-weighted test AUROC **0.817**, matching
XGBoost's 0.817 almost exactly (best on prediabetes 0.752 and metabolic
syndrome 0.883, deep model still wins 3/5 conditions outright). **Three
architecturally unrelated model families — the physiology-structured net and
two different gradient-boosting implementations — now agree on the same
~0.82 ceiling.** That kind of cross-architecture convergence is strong
evidence this is a *data* ceiling for the current leakage-safe feature set,
not something a fourth classical model would break through.

**Does a fundamentally different deep architecture do better?**
`src/models/ft_transformer.py` replaces the grouped-sparse-by-physiology first
layer with an **FT-Transformer**: every feature (continuous or categorical)
becomes its own token via a per-feature learned projection/embedding, a
learned `[CLS]` token is prepended, several self-attention layers run over the
full token sequence with no hand-coded grouping at all, and the five heads
attach to the final `[CLS]` representation — same drop-in interface as the
grouped-MLP backbone, so every other piece of the pipeline (per-head leakage
masking, loss, splits) stays identical. Checked the same rigorous way as
everything else: `src/training/architecture_sweep.py` ran both backbones
across the same 10 fixed seeds, **both without pretraining** (FT-Transformer
has no pretrained checkpoint yet — pretraining was only ever run against the
grouped-MLP — so grouped-MLP was also run unpretrained here, to isolate
architecture from that confound). Result (`results/architecture_sweep.md`):

| architecture | diabetes | prediabetes | obesity | hypertension | metab. syndrome | mean |
|---|---:|---:|---:|---:|---:|---:|
| grouped-MLP (no pretrain) | 0.845 | 0.741 | 0.769 | 0.827 | 0.859 | 0.808 |
| FT-Transformer | 0.855 | 0.750 | 0.775 | 0.830 | 0.869 | **0.816** |

**The FT-Transformer wins, and the gap survives the noise floor on 4 of 5
conditions** (diabetes +0.010, prediabetes +0.010, hypertension +0.004,
metabolic syndrome +0.010 — all bigger than their seed-to-seed noise;
obesity's +0.006 is within noise, a tie). This is the **first architectural
change in the whole project that produced a real, multi-seed-confirmed
improvement** — everything else tried (pretraining, survey-weight-for-AUROC,
SMOTE, hyperparameter tuning) either didn't help or actively hurt. Worth
noting for scale: the FT-Transformer *without* pretraining already scores
close to (and on some conditions above) the grouped-MLP's best *pretrained*
numbers reported earlier in this section — hinting that if pretraining were
ever extended to this architecture, there could be more here.

**Does pretraining help the FT-Transformer, once it gets its own pretraining
routine?** `src/training/pretrain_ft.py` implements the FT-Transformer
analogue of `pretrain.py`: instead of masking a flat-vector element (zero the
value, set the flag) before the backbone ever sees it, this masks whole
feature **tokens** *after* tokenization — a shared learned `[MASK]` token
(same value regardless of which feature lived there) replaces a random subset
of positions, and the network must recover each masked feature purely from
self-attention over the *other* tokens (continuous positions reconstructed by
one shared regression head, since all continuous features share the same
z-scored scale; each categorical feature gets its own classification head).
Checked with the same rigor as everything else — `src/training/
ft_pretrain_sweep.py` compared pretrained vs. random-init across the same 10
seeds, reporting **both** validation (noise-floor check) and held-out test
(survey-weighted) numbers. Result (`results/ft_pretrain_sweep.md`):

| variant | val mean AUROC | test mean AUROC |
|---|---:|---:|
| pretrain-on | 0.815 | 0.814 |
| pretrain-off | **0.816** | **0.816** |

**Pretraining does not help the FT-Transformer either — and the pattern is
now the same story as the grouped-MLP backbone.** Of the 10 gap checks (5
conditions x val/test), only 2 clear the noise floor, and both are
*negative*: pretraining measurably **hurts** diabetes on val (-0.003) and
metabolic syndrome on test (-0.005). Every other condition's gap is
noise-floor-sized either direction. Combined with the grouped-MLP finding
earlier in this section (pretraining helped on one lucky seed, then reversed
under a 10-seed check), this is now **two different architectures, two
different masking schemes, same conclusion**: masked-feature self-supervised
pretraining doesn't reliably earn its keep for this fine-tuning task at this
data scale (~12.6k labeled training adults) — the reconstruction pretext task
doesn't appear to teach anything the labeled data doesn't already provide.

**Round 1 of feature enrichment — `weight_history` (`WHQ`).** Adding
current/1yr-ago/age-25 self-reported weight to the FT-Transformer (no
pretraining, same 10 seeds) surfaced and resolved a real proxy-leak issue
along the way (`WHD050`, weight a year ago, initially left visible to
obesity — see Section 4b for the full story: an isolated +0.160 obesity jump
traced to a 0.556 label correlation, masked, re-run). Corrected result
(`results/ft_whq_sweep.md`):

| condition | before | + WHQ (corrected) | delta |
|---|---:|---:|---:|
| diabetes | 0.855 | 0.854 | -0.001 |
| prediabetes | 0.750 | 0.754 | +0.004 |
| obesity | 0.775 | **0.834** | **+0.059** |
| hypertension | 0.830 | 0.830 | -0.000 |
| metabolic_syndrome | 0.869 | 0.870 | +0.001 |
| **mean** | **0.816** | **0.828** | **+0.013** |

**Round 2 — kidney/liver labs, alcohol, sleep, medications.** Added `BIOPRO`
(creatinine/BUN for kidney; ALT/AST/ALP/GGT/bilirubin/albumin for liver —
one file covers both, new `renal` and `hepatic` groups), `ALQ` (alcohol —
only `ALQ130`, avg drinks/day, is stable across all 3 cycles; the frequency
questions were restructured between cycles and excluded), `SLQ` (sleep hours
— `SLD010H`→`SLD012` renamed between cycles, combined into one feature; the
diagnosed-sleep-disorder variable only exists in one cycle, excluded), and
`RXQ_RX` (prescription medication count — structurally different from every
other module, one row per *drug* not per person, needing a dedicated
aggregation step in `merge.py` rather than the generic per-module join; also
surfaced an encoding bug, non-UTF-8 bytes in 2013-2014's drug-name field,
fixed with a latin1 fallback). CRP/inflammation was checked and confirmed
**unavailable** in the standard public-use files for these cycles.

This round caught a second proxy-leak, via a different detection method than
the first: **not** an isolated jump (every condition improved together this
time — the honest signature), but a **per-feature correlation check**, run
routinely rather than only when a result looked suspicious. `RXDCOUNT`
(medication count) correlated 0.476 with diabetes and 0.448 with
hypertension — both labels have "on medication for this condition" as one of
their own defining OR-conditions (`DIQ010` self-report tracks diabetes-med
use; `BPQ050A` *is* antihypertensive-med use) — so masked from those two
heads specifically, left visible to obesity/prediabetes/metabolic_syndrome
where its correlation (0.09-0.36) reads as genuine comorbidity signal.
Corrected result (`results/ft_organlab_sweep.md`):

| condition | + WHQ only | + organ/lifestyle/meds (corrected) | delta |
|---|---:|---:|---:|
| diabetes | 0.854 | 0.866 | +0.012 |
| prediabetes | 0.754 | 0.767 | +0.013 |
| obesity | 0.834 | 0.848 | +0.014 |
| hypertension | 0.830 | 0.835 | +0.005 |
| metabolic_syndrome | 0.870 | 0.879 | +0.009 |
| **mean** | **0.828** | **0.839** | **+0.010** |

Every delta is real (beyond its noise floor) and roughly proportional across
conditions — no isolated spike this time, the trustworthy pattern.

## Where things stand now

The **0.816-0.821 "data ceiling"** described earlier in this section (three
model families converging, stacking barely nudging past it) turned out not to
be a hard ceiling — it was the ceiling *of that feature set*. Two things that
didn't help: more model architectures (LightGBM added nothing XGBoost hadn't
already shown), more hyperparameter search (found nothing better than the
defaults). Two things that did: **the FT-Transformer architecture** (+0.008,
one-time, confirmed real) and, far more, **richer non-leaking features**
(+0.013, then +0.010, both confirmed real and — after catching and fixing two
separate proxy-leak issues along the way — trustworthy). Cumulative mean val
AUROC: 0.816 (deep-model plateau) → 0.816 (FT-Transformer, architecture only)
→ 0.828 (+ weight history) → **0.839** (+ kidney/liver/alcohol/sleep/meds).

**Best per-condition numbers (validation):** diabetes 0.866, prediabetes
0.767, obesity 0.848, hypertension 0.835, metabolic_syndrome 0.879 (mean
val AUROC 0.839).

**Rebuilding the stack on the current feature set — and the test-set
confirmation, in one step.** `src/training/train_ft_final.py` trained and
saved the first deployable FT-Transformer checkpoint on the fully-enriched
schema (the 10-seed sweeps never persisted checkpoints — they exist to
measure noise, not to ship a model). `xgboost_baseline.py` and
`lightgbm_baseline.py` were re-run too — both were still on the *pre-WHQ*
feature set until now. `stacking.py` was extended from a 2-model to a
**3-model blend** (FT-Transformer + XGBoost + LightGBM), fit on validation
predictions, scored on the held-out **test** split with survey-weighted
AUROC — which doubles as the test-set/survey-weighted confirmation this
section flagged as still owed. Result (`results/stacking.md`):

| condition | FT-Transformer | XGBoost | LightGBM | **stacked** |
|---|---:|---:|---:|---:|
| diabetes | 0.859 | 0.864 | 0.862 | **0.865** |
| prediabetes | 0.766 | 0.764 | 0.760 | **0.770** |
| obesity | 0.850 | 0.855 | 0.854 | **0.857** |
| hypertension | 0.821 | 0.817 | 0.818 | **0.821** |
| metabolic_syndrome | 0.880 | 0.887 | 0.892 | 0.889 |
| **mean** | **0.835** | **0.838** | **0.837** | **0.841** |

Test-set survey-weighted numbers landed close to (not dramatically above)
the validation numbers this time — FT-Transformer alone: 0.835 test vs
~0.837-0.839 val, a small, unremarkable difference, not the bigger jump seen
earlier in the project. The gain instead came from blending: the stack wins
4/5 conditions and pushes the mean from ~0.837 (best single model) to
**0.841** — every condition's meta-model coefficients are positive with no
single model dominating (e.g. diabetes: 1.78 FT / 2.42 xgb / 2.24 lgbm),
confirming genuine three-way blending rather than the stack just parroting
whichever base model happens to be best.

**Current best, honest number: mean test AUROC 0.841 (survey-weighted,
stacked, held-out).** That is **0.009 away from 0.85** — plausibly closeable
with another round of features or a second architecture in the stack;
**0.9 remains out of reach**, and prediabetes (0.770 even at its best here)
continues to be the clear reason why, exactly as flagged throughout this
project — a biological continuum, not a clean category.

**Round 3 — general self-rated health + CBC, and the returns clearly
diminished.** Added `HSQ` (`HSD010`, "how would you rate your health,"
1=excellent...5=poor — a single question, but a well-known strong summary
predictor in epidemiology) and `CBC` (`LBXWBCSI`/`LBXHGB`/`LBXHCT`/
`LBXPLTSI` — white cell count, hemoglobin, hematocrit, platelets, the
closest available substitute for hs-CRP, which was confirmed unavailable for
these cycles). Proactive correlation check (learned from `RXDCOUNT`, not
waiting for a suspicious result this time): `HSD010` reaches 0.362 with
obesity, the highest of the batch, but with no structural circularity —
unlike `WHD050`/`RXDCOUNT`, self-rated health isn't part of any label's
definition or a near-restatement of one; a moderate correlation between "I
feel unhealthy" and obesity status is exactly the well-documented,
legitimate epidemiological association it looks like. Not masked.

Retrained everything on the new schema (FT-Transformer, XGBoost, LightGBM,
restacked). Result: **the gain nearly vanished.**

| condition | round 2 stacked | round 3 stacked (HSQ+CBC) | delta |
|---|---:|---:|---:|
| diabetes | 0.865 | 0.869 | +0.004 |
| prediabetes | 0.770 | 0.775 | +0.005 |
| obesity | 0.857 | 0.859 | +0.002 |
| hypertension | 0.821 | 0.821 | +0.000 |
| metabolic_syndrome | 0.889 | 0.885 | -0.004 |
| **mean** | **0.841** | **0.842** | **+0.001** |

Compare to round 1 (+0.013 mean) and round 2 (+0.010 mean): **this round
added essentially nothing.** That's a real, useful data point in its own
right — general self-rated health and a basic blood count apparently
overlap heavily with what the model could already infer from everything
already in the schema (diet, activity, kidney/liver labs, lipids,
demographics), rather than contributing new independent signal. This is the
clearest sign yet that the "add another NHANES module" lever is running out
of room, not that it stopped working arbitrarily.

**Current best, honest number: mean test AUROC 0.842 (survey-weighted,
stacked, held-out).** 0.008 from 0.85 — still plausibly closeable, but each
further feature round should be expected to buy less than the last, not
more. **0.9 remains firmly out of reach**, and prediabetes (0.775 even at
its stacked best here) continues to be the clear, biologically-grounded
reason why.

**Adding a 4th base learner (grouped-MLP) — tested, essentially a wash.**
Following the architecture/ensemble recommendation above, `train_mlp_final.py`
retrained the grouped-MLP backbone from scratch on the current 51-token
schema (its only prior checkpoint predated all three feature rounds), and
`stacking.py` was extended to a 4-model blend (FT-Transformer + XGBoost +
LightGBM + grouped-MLP).

| condition | 3-model stacked | 4-model stacked | delta |
|---|---:|---:|---:|
| diabetes | 0.869 | 0.869 | +0.000 |
| prediabetes | 0.775 | 0.774 | -0.001 |
| obesity | 0.859 | 0.860 | +0.001 |
| hypertension | 0.821 | 0.822 | +0.001 |
| metabolic_syndrome | 0.885 | 0.883 | -0.002 |
| **mean** | **0.842** | **0.841** | **-0.001** |

The grouped-MLP was individually the weakest of the four base models on
every condition (e.g. diabetes 0.842 alone, vs. 0.862-0.870 for the other
three) — its hard physiology-group sparsity apparently costs more than its
different inductive bias buys back once FT-Transformer, XGBoost, and
LightGBM already cover the signal well. The meta-model does use it
non-trivially (e.g. prediabetes coef 2.17, metabolic_syndrome coef 1.91 —
not being ignored), but including a clearly-weaker 4th model nets out flat
to very slightly negative rather than additive. Contrast with the 2→3-model
step (deep+XGBoost → +LightGBM), which was a real gain because LightGBM was
competitive with, not weaker than, the existing models. **Conclusion: base
model diversity only helps the stack when the new model is roughly as
strong as the ones already in it — not diversity for its own sake.** Kept
the 3-model stack as the deployed configuration; `stacking.py`'s current
code path is the 4-model version but the honest verdict is "no improvement,"
not adopted as an upgrade.

**Redoing FT-Transformer pretraining on the current schema — also negative,
now confirmed on the enriched feature set.** `pretrain_ft.py` was re-run
from scratch against the current 51-token schema (its only prior checkpoint
predated all three feature rounds and was architecturally incompatible —
different tokenizer/embedding shapes). Converged cleanly (best val
reconstruction loss 1.667 @ epoch 125). `ft_pretrain_sweep.py` then compared
pretrained vs. randomly-initialized FT-Transformer across the same 10 fixed
seeds, both val (noise-floor check) and survey-weighted test:

| | diabetes | prediabetes | obesity | hypertension | metab. syndrome | mean |
|---|---:|---:|---:|---:|---:|---:|
| val, pretrain-on | 0.869 | 0.766 | 0.853 | 0.835 | 0.885 | 0.842 |
| val, pretrain-off | 0.872 | 0.771 | 0.852 | 0.837 | 0.886 | 0.844 |
| test, pretrain-on | 0.867 | 0.769 | 0.853 | 0.816 | 0.872 | 0.835 |
| test, pretrain-off | 0.862 | 0.772 | 0.854 | 0.820 | 0.878 | 0.837 |

Pretraining is net *worse*, not just unhelpful, on the current schema: mean
val -0.002, mean test -0.002. Where the gap clears the noise floor (2/5 val
conditions, 3/5 test conditions), it's negative in every case but one —
prediabetes val (-0.005) and hypertension val (-0.002) real; diabetes test
(+0.005), hypertension test (-0.004), and metabolic_syndrome test (-0.005)
real. Only diabetes improves; hypertension and metabolic_syndrome get
measurably worse. **Not adopted.** This replicates the earlier pre-enrichment
finding (`results/ft_pretrain_sweep.md`, first run) rather than overturning
it — masked-token reconstruction pretraining doesn't help this backbone,
on either the small or the fully-enriched schema.

**4th-base-learner attempts, both tried and rejected — the stack stays at
3 models.** Two candidates for extending `stacking.py` beyond FT-Transformer
+ XGBoost + LightGBM were tested this round: the grouped-MLP retrained on
the current schema (flat-to-negative, 0.842->0.841, see above) and the
pretrained FT-Transformer just discussed (net negative on its own, so folding
it into the stack in place of the current FT-Transformer checkpoint was not
attempted — swapping in a worse component cannot make the blend better).
`stacking.py` was reverted to its 3-model form. Re-confirmed: the 3-model
stack (mean test AUROC 0.842) still beats standalone FT-Transformer (0.836)
on **every one of the 5 conditions** — diabetes 0.869 vs 0.862, prediabetes
0.775 vs 0.770, obesity 0.859 vs 0.852, hypertension 0.821 vs 0.819,
metabolic_syndrome 0.885 vs 0.879 — so the stack remains clearly worth
keeping even though neither 4th-model candidate panned out.

**Current best, honest number: mean test AUROC 0.842 (survey-weighted,
3-model stack, held-out).** 0.008 from 0.85. Both obvious architecture-side
levers (more base-model diversity, richer pretraining) have now been tried
on the fully-enriched schema and come back negative — this is a meaningful,
not-yet-acted-on signal that the remaining gap to 0.85 is more likely to
close (if at all) through further feature engineering, hyperparameter
search, or accepting 0.842 as the honest ceiling for this data and these
leakage constraints, rather than through model/ensemble sophistication.

**Immediate next steps:** investigate the race-calibration regression for
survey weighting; reliability diagrams for the paper. Awaiting user
direction on whether to pursue a further feature round, hyperparameter
tuning, or treat 0.842 as the reported ceiling.
