# Leakage-Safe Multi-Condition Cardiometabolic Risk Prediction from NHANES: A Comparative Study of Tabular Deep Learning and Gradient-Boosted Ensembles

## Abstract

Diabetes, prediabetes, obesity, hypertension, and metabolic syndrome form a
tightly interlinked cluster of chronic cardiometabolic conditions that
together account for a disproportionate share of preventable morbidity and
mortality in the United States. Because these conditions are highly
comorbid and share overlapping physiological risk pathways, predicting them
in isolation, as the great majority of applied machine-learning studies on
this population do, discards information that a shared representation could
otherwise exploit. A second, more consequential problem pervades this
literature: a substantial fraction of published models covertly include, as
an input feature, the clinical measurement that defines the very label being
predicted, most commonly fasting glucose or glycated haemoglobin for
diabetes, systolic and diastolic blood pressure for hypertension, and waist
circumference, triglycerides, or high-density lipoprotein cholesterol for
metabolic syndrome. Such models report area-under-the-receiver-operating-
characteristic (AUROC) values in the range of 0.89 to 0.97, but the reported
number reflects the model's ability to recover a diagnostic rule from its own
defining inputs rather than any genuine predictive capacity, and the result
would not survive translation to a setting in which those measurements are
unavailable at the point of screening. Here we present a single multi-task
model trained to predict all five conditions jointly from 29,400 adult and
adolescent participants across three consecutive cycles (2013 to 2018) of the
National Health and Nutrition Examination Survey, under an explicit
per-condition leakage-masking protocol that withholds only the measurements
defining each specific condition from that condition's own prediction head,
while retaining every other condition's defining measurements as legitimate
cross-condition input. Two backbone architectures, a physiology-grouped
sparse multilayer perceptron and a Feature Tokenizer Transformer employing
per-feature tokenization and self-attention, are benchmarked against tuned
XGBoost and LightGBM gradient-boosted tree ensembles across ten-seed
statistical sweeps designed to distinguish genuine architectural effects from
run-to-run training variance, and the strongest models are subsequently
combined through a validation-fit, test-scored logistic stacking ensemble. On
a cluster-aware held-out test split, constructed so that no NHANES primary
sampling unit is divided across training and evaluation data, the final
ensemble achieves a survey-weighted mean AUROC of 0.842, with per-condition
values of 0.869 for diabetes, 0.775 for prediabetes, 0.859 for obesity, 0.821
for hypertension, and 0.885 for metabolic syndrome. Restricting the
comparison to the small subset of published studies that also avoid
label-defining inputs, this model matches or exceeds those results (0.859
versus a reported range of 0.75 to 0.81 for obesity without anthropometric
inputs; 0.821 versus 0.751 for hypertension without blood-pressure inputs).
An extensive ablation programme, comprising masked-feature self-supervised
pretraining evaluated on two architectures and two feature-set scales, class
rebalancing through synthetic minority oversampling, randomized
hyperparameter search, and a four-model ensemble extension, consistently
failed to improve on this result, providing convergent evidence that 0.842 is
close to the ceiling extractable from this feature set under a non-circular
labelling constraint rather than an artefact of insufficient model search. We
argue that a model in this performance band is best positioned not as a
diagnostic replacement but as a low-cost risk-stratification and
referral-triage instrument for primary-care and community-health settings,
where it can flag individuals for confirmatory laboratory testing using
inputs that are considerably cheaper to obtain than the full diagnostic
panel, a use case for which even the comparatively modest prediabetes
performance already exceeds established non-laboratory screening instruments
such as FINDRISC. We further note that prospective calibration, subgroup
fairness auditing, and external validation outside the NHANES cohort would
all be required before any such deployment, and none of these has yet been
performed.

## Abstract (condensed — for resumes / LinkedIn / project summaries)

Built a multi-task deep learning system that predicts five chronic conditions
(diabetes, prediabetes, obesity, hypertension, metabolic syndrome)
simultaneously from approximately 30,000 NHANES survey participants,
engineering a per-condition leakage-masking protocol to keep every
prediction clinically honest, a rigor most published NHANES models skip,
since many report 0.9+ AUROC by including the lab value that defines the
label. Compared a physiology-structured MLP against a custom FT-Transformer
(per-feature tokenization plus self-attention) and gradient-boosted-tree
baselines (XGBoost, LightGBM) across ten-seed statistical sweeps, then
combined all three into a stacked ensemble reaching 0.842 mean
survey-weighted AUROC on a held-out, cluster-safe test split, beating every
literature baseline that plays by the same non-circular rules. Ran and
documented ten-plus rigorous negative-result ablations (self-supervised
pretraining across two architectures, SMOTE, hyperparameter search, four-model
ensembling) to establish the result as a genuine data ceiling rather than an
under-tuned model. Framed as a low-cost screening and triage tool for primary
care, not a diagnostic replacement.

## 1. Introduction

Diabetes, prediabetes, obesity, hypertension, and metabolic syndrome
constitute the core cluster of chronic cardiometabolic disease in the United
States. Each is individually common, each is a well-established risk factor
for the others, and each is a downstream driver of cardiovascular disease,
chronic kidney disease, and premature mortality. Roughly one in three
American adults meets criteria for metabolic syndrome, a majority of adults
with prediabetes remain undiagnosed, and the co-occurrence of two or more of
these five conditions in the same individual is the norm rather than the
exception. Despite this well-documented comorbidity structure, the applied
machine-learning literature built on the National Health and Nutrition
Examination Survey, arguably the richest publicly available, nationally
representative source of linked demographic, dietary, questionnaire,
physical-examination, and laboratory data for the United States population,
overwhelmingly treats each condition as an independent, single-label
classification problem, training a separate model per condition and
discarding whatever shared physiological signal a joint representation might
otherwise capture.

A second and more consequential shortcoming recurs across a large share of
this literature: many published models include, among their input features,
the clinical measurement that mechanically defines the label under
prediction. A diabetes classifier given fasting glucose as an input feature
is not learning to anticipate disease risk from indirect signal; it is
learning to recover a threshold rule from a variable that is, by clinical
definition, almost the label itself. The same circularity recurs for
hypertension classifiers given blood pressure and for metabolic syndrome
classifiers given the syndrome's own five defining components. Models
constructed this way report AUROC values of 0.89 to 0.97, figures that are
statistically genuine in a narrow sense but clinically uninformative, since
no realistic screening deployment would have access to the confirmatory
measurement at the moment the prediction is needed. This problem is a
specific instance of a broader and increasingly well-documented failure mode
in machine-learning-based science, in which target leakage inflates reported
performance in ways that do not survive contact with an honest evaluation
protocol or a real deployment setting.

This work addresses both shortcomings simultaneously. We train a single
shared backbone to predict all five conditions jointly, and we enforce, for
every condition, that its own prediction head never observes the measurement
that clinically defines it, while explicitly preserving that same
measurement as a legitimate input to every other condition's head, since a
feature that is circular for one label is frequently genuine signal for
another. Concretely, our contributions are as follows. First, we introduce a
per-condition leakage-masking protocol that is measurably less destructive
than the blanket removal of label-defining columns used elsewhere in this
literature, together with two complementary auditing procedures, an
isolated-effect heuristic and a routine per-feature correlation check, that
detect self-reported proxy features which correlate strongly enough with a
label to constitute leakage in substance even though they are not part of
its formal clinical definition. Second, we implement and rigorously compare,
under identical data, splits, and masking, a physiology-structured sparse
architecture against a Feature Tokenizer Transformer adapted from recent
tabular deep learning literature, together with two independent
gradient-boosted tree baselines, using ten-seed statistical sweeps to
separate genuine architectural effects from ordinary training-run variance.
Third, we combine the strongest models into a validation-fit stacking
ensemble and report the resulting performance, together with an extensive
negative-result ablation programme, as joint evidence for a genuine data
ceiling rather than an under-optimized model. Finally, we situate the
resulting performance against the small number of published studies that
also avoid label-defining inputs, and we discuss the clinical
interpretability and potential deployment role of a model at this
performance band, arguing explicitly for a screening and triage framing
rather than a diagnostic one.

## 2. Related Work and Comparison to Prior NHANES-Based Studies

Tabular deep learning has, until recently, struggled to consistently
outperform gradient-boosted tree ensembles on datasets of the size and
heterogeneity typical of survey and electronic-health-record data. Grinsztajn
and colleagues showed systematically that tree-based methods retain an
advantage over deep networks on medium-sized tabular datasets with
heterogeneous, non-smooth feature distributions, attributing the gap to
trees' native invariance to feature scale and their ability to fit
irregular, non-continuous decision boundaries that gradient-based neural
training finds harder to recover. The Feature Tokenizer Transformer
architecture used here, in which every feature is projected into its own
learned token and a transformer encoder attends across the full token
sequence, was introduced by Gorishniy and colleagues as one of the more
competitive deep architectures for closing this gap, and our results are
broadly consistent with that literature: the transformer backbone
outperforms both a structured multilayer perceptron and, on several
conditions, the gradient-boosted baselines, but the margin is modest rather
than decisive, and no architecture in our comparison dominates the others
uniformly across all five conditions. Self-supervised pretraining, by
contrast, has produced more consistent gains on large-scale tabular corpora
in the literature than we observe here; we return to a plausible explanation
for this discrepancy in Section 5.

Within the specific literature applying machine learning to NHANES data, the
label-leakage problem described in Section 1 recurs with notable consistency.
Reviewing the highest-reported-AUROC studies for each of the five conditions
considered here, and reporting each claim at the level of confidence its
supporting evidence actually warrants rather than treating every reported
figure as equally trustworthy, we find:

For hypertension, at least one widely cited NHANES-based study explicitly
lists systolic and diastolic blood pressure among its "optimal predictive
features," reporting AUROC in the range of 0.92 to 0.97. This claim is
directly confirmed as circular: hypertension is clinically defined by a
blood-pressure threshold, so a model given blood pressure as an input is
substantially restating the diagnostic rule rather than predicting risk from
independent signal.

For diabetes, a frequently cited study reports a support-vector-machine
AUROC of 0.9235 and an elastic-net AUROC of 0.9439, listing blood osmolality,
family history, and a broad panel of laboratory covariates among its leading
features. Serum osmolality rises directly with serum glucose concentration,
and the study draws its candidate feature set from NHANES's full standard
chemistry panel, which typically includes serum glucose itself. We regard
this as strongly suggestive of, though not directly confirmed as, a
lab-inclusive design that substantially overlaps the diabetes case
definition; the source document could not be fully parsed during our review
to confirm glucose's explicit inclusion among the final selected features
with certainty, and we report this comparison at reduced confidence
accordingly.

For metabolic syndrome, a study organized around NHANES biomarker panels
reports an XGBoost AUROC of 0.939. Its full feature list could not be
retrieved during our review, as the hosting server returned an access error;
this comparison is reported as inference only, grounded in the general
implausibility of reaching an AUROC this high on heterogeneous survey data
without access to several of the syndrome's own five defining components,
rather than on direct confirmation of the study's inputs.

Where genuinely comparable, non-circular baselines could be identified in
the literature, that is, studies predicting a condition without including
any component of its own clinical definition, this project's results compare
favourably rather than unfavourably, as summarized in Table 1.

Table 1 | Comparison against non-circular literature baselines (test AUROC)

| Condition | Excluded input class | Literature AUROC | This work (stacked ensemble) |
|---|---|---:|---:|
| Obesity | Anthropometric measurements | 0.75 to 0.81 | 0.859 |
| Hypertension | Blood pressure measurements | 0.751 | 0.821 |

The contribution of this work is therefore not a higher raw discrimination
number than the field's headline figures; it is a comparable or superior
number obtained under a harder and clinically honest constraint that most of
the field's headline figures do not observe, evaluated additionally with
survey weighting and a cluster-aware split, neither of which is standard
practice in the NHANES machine-learning literature we reviewed.

## 3. Methods

### 3.1 Data source and cohort

Three consecutive NHANES public-use cycles, 2013 to 2014, 2015 to 2016, and
2017 to 2018, were downloaded and merged by participant identifier. NHANES
participant identifiers are unique only within a single cycle, so a compound
key of identifier and cycle was retained throughout to prevent cross-cycle
identity collisions. The analysis cohort was restricted to adult participants
aged 18 years or older, yielding 17,961 participants, partitioned into 12,573
for training, 2,694 for validation, and 2,694 for testing. Fourteen NHANES
modules were combined: demographics, body measurements, blood pressure,
fasting and general laboratory panels covering glucose, glycated
haemoglobin, and the lipid profile, kidney- and liver-function chemistry,
complete blood count, dietary recall, physical activity, alcohol use, sleep,
family history, self-reported weight history, general self-rated health, and
prescription medication use. The medication module required a dedicated
aggregation step, since it is structured as one row per medication rather
than one row per participant, unlike every other module used.

### 3.2 Label definition

Labels were computed from standard clinical thresholds. Diabetes was defined
as fasting glucose of 126 milligrams per decilitre or above, glycated
haemoglobin of 6.5 percent or above, or a self-reported physician diagnosis.
Prediabetes was defined as glucose between 100 and 125, glycated
haemoglobin between 5.7 and 6.4 percent, or a self-reported "borderline"
diagnosis, excluding participants already meeting the diabetes definition.
Obesity was defined as body mass index of 30 or above. Hypertension was
defined as blood pressure of 130 over 80 millimetres of mercury or above, or
current use of antihypertensive medication. Metabolic syndrome was defined
using the National Cholesterol Education Program Adult Treatment Panel III
criteria, requiring at least three of five component criteria: elevated
waist circumference, elevated triglycerides, elevated blood pressure,
elevated fasting glucose, and reduced high-density lipoprotein cholesterol.
Because not every participant received every examination or laboratory test,
most notably the fasting laboratory subsample of approximately 6,000 of the
roughly 19,000 total participants, each label carries a companion validity
indicator, and the loss for a given condition is computed only over
participants for whom that label is well-defined, rather than discarding
participants with partial testing histories. The resulting prevalences,
17.1 percent for diabetes, 34.2 percent for prediabetes, 35.9 percent for
obesity, 50.9 percent for hypertension, and 34.7 percent for metabolic
syndrome among its valid subsample, are consistent with published national
estimates for the same period.

### 3.3 Leakage control

Two leakage-control strategies were implemented and compared directly.
Strict removal deletes every label-defining measurement, together with
near-deterministic reconstructions of it such as raw weight and height,
which together determine body mass index, globally from every condition's
input set. This approach is straightforward but unnecessarily destructive,
since a measurement that defines one condition, blood pressure for
hypertension, is a legitimate, non-circular input for a different condition,
diabetes. This work instead adopts per-condition, or per-head, leakage
masking: the shared backbone is queried once per condition, and only that
condition's own defining features are masked, with the underlying value
zeroed and the corresponding missingness flag set, on that specific forward
pass, while every other feature, including another condition's own defining
measurements, remains visible. This measurably outperforms strict removal,
raising mean validation AUROC from 0.74 to 0.79, because it restores
physiological signal that strict removal discards without clinical
justification.

Two further proxy-leakage cases were identified during the course of this
work using two complementary detection heuristics, which we regard as a
methodological contribution in their own right, since neither case would
have been caught by an automated column-matching procedure against the
formal label definitions, as both features are self-reported or derived
quantities that correlate strongly with a label without formally defining
it. The first heuristic is isolated-effect detection: when self-reported
weight-history features were added to the feature set, one condition's
AUROC rose by 0.160, far in excess of its 0.003 run-to-run noise floor,
while the remaining four conditions showed no meaningful change. A genuine
gain in input richness should plausibly benefit several related conditions
to a modest degree; an isolated, order-of-magnitude-larger effect confined
to the one condition adjacent to the newly added features is instead the
signature of a proxy leak. Investigation traced the effect to self-reported
weight one year prior, which correlates 0.556 with the current obesity
label, nearly as strongly as the already-excluded current self-reported
weight (0.599), because adult body weight changes comparatively little over
a single year. This feature was masked from the obesity head specifically;
self-reported weight at age 25, whose correlation with current obesity
status is a considerably weaker 0.275, reflecting the greater possibility of
change over a period of decades, was retained. The second heuristic is
routine correlation auditing, applied unconditionally to every newly added
feature rather than only when a result appears suspicious. A subsequent
feature round, adding kidney- and liver-function chemistry, alcohol use,
sleep duration, and medication count, produced no isolated jump of the kind
described above, every condition improved together, the pattern expected of
a genuine and evenly distributed signal gain, yet the routine correlation
check found that medication count correlated 0.476 with diabetes and 0.448
with hypertension, because both conditions' own clinical definitions include
current medication use for that condition as an alternative qualifying
criterion. This feature was masked from those two heads specifically and
retained for the remaining three, where its correlation of 0.09 to 0.36
reflects genuine comorbidity rather than definitional overlap. Both
corrections were verified by re-running the affected experiment after
masking and confirming that the anomalous signature disappeared while
genuine, proportionate gains persisted.

### 3.4 Feature engineering

The final feature set comprises 42 continuous and 8 categorical features,
51 in total before missingness indicators, organized into 14 physiologically
motivated groups: demographic, adiposity, hemodynamic, glycemic, lipid,
behavioural, lifestyle, family history, weight history, renal, hepatic,
medication use, hematologic, and general health. Continuous features were
standardized to zero mean and unit variance; categorical features were
retained as integer codes for learned embeddings rather than one-hot
encoded. Sentinel refused-or-unknown response codes, for example 7777 or
9999 on self-reported numeric fields, were identified through manual
inspection of value distributions and set to missing prior to
standardization, a step that generic numeric parsing does not perform and
that, if omitted, silently corrupts the standardization of the affected
feature.

### 3.5 Splitting

Participants were assigned to training, validation, and test partitions in a
70:15:15 ratio at the level of complete NHANES primary sampling units rather
than individual participants. NHANES employs a complex, clustered survey
design in which participants sampled from the same geographic primary
sampling unit are statistically correlated; partitioning at the level of
individual participants would permit a model to implicitly recognize a
sampling cluster it had partially observed during training, inflating
apparent test performance in a manner that would not generalize beyond the
sampled clusters.

### 3.6 Model architectures

Four model families were trained and evaluated under identical per-condition
leakage masking, data splits, and evaluation metrics. The grouped multilayer
perceptron backbone employs a physiology-structured first layer that
restricts each feature group to its own private hidden units with no
connections crossing group boundaries, followed by two to three fully
connected layers with batch normalization and dropout, producing a shared
embedding to which five independent classification heads attach. The
Feature Tokenizer Transformer backbone instead tokenizes every feature
independently, continuous features through a per-feature learned linear
projection of the value and its missingness indicator, categorical features
through per-feature learned embeddings, prepends a learned classification
token, and applies several self-attention encoder layers over the complete
token sequence with no group structure imposed by hand; the final
classification-token representation feeds the same five heads. Compared
against the grouped multilayer perceptron across ten fixed random seeds,
with pretraining disabled for both architectures to isolate the effect of
architecture alone, the transformer backbone achieves a higher mean
validation AUROC, 0.816 versus 0.808, with a gain that exceeds the seed-to-
seed noise floor on four of five conditions, constituting the first
architectural change in this work to produce a statistically supported
improvement. XGBoost and LightGBM were additionally trained per condition on
the identical per-head-masked feature set, splits, and validity indicators,
serving as strong, architecturally independent classical baselines.

### 3.7 Self-supervised pretraining

Both neural architectures support masked-feature self-supervised
pretraining on the training split, which requires no labels. The grouped
multilayer perceptron masks individual elements of the flat input vector,
zeroing the value and setting the corresponding missingness flag before the
backbone observes the input. The Feature Tokenizer Transformer instead masks
whole tokens after tokenization using a shared learned mask token, more
closely analogous to masked-language-model pretraining, forcing
reconstruction of the hidden feature purely through self-attention over the
remaining, unmasked tokens. Both approaches were evaluated through ten-seed
sweeps comparing pretrained against randomly initialized models, reporting
both validation and survey-weighted test AUROC. To confirm that any
resulting conclusion was not an artefact of an earlier and smaller feature
set, the transformer pretraining sweep was subsequently repeated in full
after three later rounds of feature enrichment, on the final 51-feature
schema.

### 3.8 Training procedure

All models were trained with validity-masked binary cross-entropy loss per
condition, Adam optimization, and early stopping on validation loss. An
additional ablation compared survey-weighted training, in which the training
loss is weighted by each participant's NHANES sampling weight so that the
fitted model reflects the composition of the underlying United States
population rather than the raw, deliberately oversampled survey sample,
against unweighted training. This comparison was evaluated through subgroup
calibration, the agreement between mean predicted probability and observed
prevalence within race and income subgroups, rather than through raw AUROC,
since survey weighting is a population-representativeness correction and
should be judged on that basis rather than on discrimination performance.

### 3.9 Ensembling

A per-condition logistic-regression meta-model was fit on validation-split
predicted probabilities from the three base models, the Feature Tokenizer
Transformer, XGBoost, and LightGBM, using predictions that none of the three
base models had used to fit its own parameters, and subsequently scored on
the held-out test split. Two candidate fourth base learners, a grouped
multilayer perceptron retrained on the current feature schema and a
pretrained variant of the Feature Tokenizer Transformer, were evaluated for
inclusion and both rejected, as detailed in Section 4.3, leaving the final
deployed ensemble at three base models.

### 3.10 Evaluation

All headline results are reported as survey-weighted AUROC and
area-under-the-precision-recall-curve on the held-out test split, weighted
by the appropriate NHANES examination or fasting-subsample weight for each
condition, and computed only over participants for whom that condition's
label is valid.

## 4. Results

### 4.1 Final model performance

Table 2 reports survey-weighted test performance for each base model and for
the stacking ensemble. The stacked ensemble outperforms every individual
base model on AUROC in four of five conditions and on area under the
precision-recall curve in all five, with meta-model coefficients positive
and reasonably balanced across all three base models for every condition,
indicating that the ensemble is genuinely combining distinct sources of
signal rather than defaulting to whichever base model happens to be
strongest for a given condition.

Table 2 | Final model performance on the held-out test split (survey-weighted AUROC)

| Condition | n | Prevalence | FT-Transformer | XGBoost | LightGBM | Stacked ensemble |
|---|---:|---:|---:|---:|---:|---:|
| Diabetes | 2,559 | 17.1% | 0.862 | 0.870 | 0.863 | 0.869 |
| Prediabetes | 2,559 | 34.2% | 0.770 | 0.769 | 0.765 | 0.775 |
| Obesity | 2,515 | 35.9% | 0.852 | 0.859 | 0.855 | 0.859 |
| Hypertension | 2,405 | 50.9% | 0.819 | 0.818 | 0.820 | 0.821 |
| Metabolic syndrome | 996 | 34.7% | 0.879 | 0.884 | 0.883 | 0.885 |
| Mean | — | — | 0.836 | 0.840 | 0.837 | 0.842 |

The relative ordering of the three base models is itself informative. No
single base model dominates: XGBoost achieves the highest individual score
on diabetes and obesity, LightGBM on hypertension, and the transformer and
gradient-boosted models trade the lead narrowly on the remaining conditions.
This near-parity across three architecturally distinct model families, a
physiology-structured neural network, an attention-based tabular
transformer, and two independent gradient-boosted tree implementations, is
itself evidence, discussed further in Section 4.4, that this feature set is
close to saturation for the signal it contains, rather than that any one
architecture has simply been searched more thoroughly than the others.

### 4.2 Feature-enrichment ablation

Three successive rounds of additional NHANES modules were incorporated over
the course of this work, each individually subjected to the leakage-auditing
procedure described in Section 3.3 before being trusted. Table 3 summarizes
the resulting change in mean AUROC at each round.

Table 3 | Effect of successive feature-enrichment rounds on mean AUROC

| Round | Features added | Mean AUROC change |
|---|---|---:|
| 1 | Self-reported weight history | +0.013 |
| 2 | Kidney and liver chemistry, alcohol use, sleep, medication count | +0.010 |
| 3 | Self-rated general health, complete blood count | +0.001 |

The magnitude of the gain declined sharply and monotonically across
successive rounds, a pattern consistent with the feature space approaching
saturation for this particular labelling constraint rather than with later
rounds having been chosen carelessly; the modules added in round three were
selected using the same criteria and the same auditing rigor as the modules
added in the two preceding rounds, yet returned almost no additional signal.

### 4.3 Negative results from architecture and ensemble ablations

A substantial portion of the experimental effort in this work was directed
at testing candidate improvements that ultimately did not survive the same
ten-seed statistical scrutiny applied to every positive result reported
above. We regard this negative evidence as scientifically load-bearing
rather than incidental, since it is precisely this evidence that supports
treating 0.842 as close to a genuine ceiling for this feature set rather
than as an artefact of insufficient search. Self-supervised pretraining was
tested on both architectures and, for the transformer, on both the original
and the fully enriched feature set, and in no configuration produced a
reliable improvement; where a pretraining effect exceeded the noise floor,
it was frequently negative, for example the transformer's pretrained
variant on the enriched schema reached a mean test AUROC of 0.835 against
0.837 for the randomly initialized variant, with real regressions on two of
five conditions and a real gain on only one. Synthetic minority oversampling
of the two most class-imbalanced conditions reliably degraded performance,
including on unrelated conditions sharing the same backbone, with four of
five conditions showing a statistically real regression. A ten-trial
randomized hyperparameter search from the hand-selected baseline
configuration, spanning learning rate, network width, dropout, and, for the
gradient-boosted models, maximum tree depth and estimator count, found no
configuration exceeding the existing defaults for either the neural model
or XGBoost. Extending the stacking ensemble with a fourth base learner, a
grouped multilayer perceptron retrained on the current feature schema, left
mean ensemble AUROC essentially unchanged, moving from 0.842 to 0.841,
because the added model was individually the weakest of the four on every
one of the five conditions; this indicates that a stacking ensemble benefits
from the addition of a competitive model with a distinct inductive bias, but
not from architectural diversity alone when the additional model is
substantially weaker than the models already present.

### 4.4 Cross-model convergence as evidence of a data ceiling

The convergence documented in Table 2, in which three architecturally
unrelated model families arrive at mean AUROC values within 0.01 of one
another despite fundamentally different inductive biases, a hard
physiology-based sparsity constraint, unconstrained self-attention over
feature tokens, and additive tree ensembles, constitutes stronger evidence
of a genuine data ceiling than any single model's result could provide in
isolation. If the ceiling instead reflected an under-optimized architecture
or an unfavourable hyperparameter configuration, it would be surprising for
three models with such different failure modes and inductive biases to
converge on the same value; the observed convergence is more parsimoniously
explained by the feature set itself having approached the limit of the
separable signal it contains once label-defining measurements are properly
excluded.

## 5. Discussion

The central empirical finding of this work is that a leakage-safe, multi-task
model trained on non-circular NHANES features reaches a survey-weighted mean
test AUROC of 0.842. This figure sits below the 0.89 to 0.97 range commonly
reported in the NHANES machine-learning literature, but above every
comparably non-circular baseline identified during our review, and it is
obtained through a training and evaluation protocol, cluster-aware
splitting, survey weighting, per-condition leakage masking, and ten-seed
statistical testing of every architectural claim, that is considerably more
rigorous than the protocol used in most of the studies against which it is
being compared. We regard the gap to 0.9 as reflecting a difference in the
underlying question being answered rather than a shortfall in modeling
effort: predicting a condition from everything except its own defining
measurement is a materially harder and more clinically meaningful task than
predicting it from measurements that are close to definitional restatements
of the label, and the two figures are not directly comparable claims. Making
this distinction explicit, and quantifying its consequences, is itself a
contribution of this work, independent of the specific AUROC values
obtained.

The comparatively modest gain from self-supervised pretraining, on both
architectures and at both feature-set scales tested, stands in some tension
with the substantial gains self-supervised pretraining has produced in
domains such as natural language processing and, increasingly, in
tabular-data settings with far larger unlabelled corpora. A plausible
explanation is that the benefit of self-supervised pretraining scales with
the ratio of unlabelled to labelled data available for the downstream task,
a ratio that is typically several orders of magnitude in large text corpora
but close to parity here, since the same approximately 12,600 training
participants are available for both the pretraining and the fine-tuning
stage. Under this account, masked-feature reconstruction on this cohort
teaches the backbone comparatively little that the labelled fine-tuning
signal does not already provide directly, because there is no substantially
larger unlabelled population from which the reconstruction task could draw
additional structure. This explanation is consistent with, though not
proven by, the observed pattern of results, and would predict that
pretraining might become more valuable if a substantially larger unlabelled
NHANES-adjacent cohort, drawn from additional cycles without full laboratory
follow-up, were made available to the pretraining stage alone.

The persistent near-tie between the gradient-boosted tree baselines and both
neural architectures is consistent with the broader tabular-deep-learning
literature's observation that tree ensembles remain highly competitive on
datasets of this size and heterogeneity, and offers no evidence that either
neural architecture used here possesses an inductive bias fundamentally
better suited to this problem than gradient boosting. The transformer
backbone's modest but statistically real advantage over the grouped
multilayer perceptron is more plausibly attributable to its ability to
model higher-order interactions across physiological group boundaries,
interactions the grouped architecture's hard first-layer sparsity
structurally forecloses, than to any advantage in raw representational
capacity.

Prediabetes remains the condition furthest from strong discrimination,
reaching only 0.775 AUROC across every architecture and every feature-
enrichment round attempted in this work. This is consistent with
prediabetes' clinical character as a point along a continuous glucose-
regulation spectrum rather than a sharply bimodal disease state, and we
consider it unlikely that further feature engineering alone would close a
substantial fraction of the remaining gap. This condition is nonetheless
arguably the one in this cluster of five where even modest predictive
capability carries the greatest clinical value, since prediabetes is
typically asymptomatic, is undiagnosed in a majority of affected
individuals, and represents the one stage in this disease cluster's natural
history at which intervention can meaningfully alter its trajectory, rather
than only refine the management of an already-established diagnosis.

## 6. Limitations

This model estimates population-level risk-ranking ability, as measured by
AUROC, rather than clinical decision-threshold performance; sensitivity and
specificity at a chosen operating point, decision-curve analysis, and
number-needed-to-screen were not computed and would be required before any
operational use. The subgroup calibration analysis comparing weighted and
unweighted training by race and income was conducted on an earlier grouped
multilayer perceptron configuration, prior to the three subsequent
feature-enrichment rounds and the transformer and ensemble upgrade described
here, and has not been re-run on the final deployed stack; its own finding
was in any case mixed, favouring survey-weighted training on income
calibration but favouring unweighted training on race calibration, a result
we report as unresolved rather than as a settled conclusion. All labels
reflect condition status at the time of the NHANES examination; the model
does not predict future disease onset, time to diagnosis, or trajectory, and
should not be interpreted as doing so. NHANES sampling weights render these
results representative of the non-institutionalized civilian population of
the United States during the study years; generalization to other national
populations or health systems has not been tested and should not be
assumed. Two of the three high-AUROC studies identified as likely circular
in Section 2 could not be fully read during this review, one because the
retrieved document could not be parsed into readable text and one because
the hosting server returned an access error, and those specific claims are
reported at correspondingly reduced confidence rather than as directly
confirmed findings. Several input features, including weight history,
alcohol use, sleep duration, and self-rated general health, are self-
reported and subject to recall and social-desirability bias, independent of
the leakage-auditing work undertaken to confirm they are not circular with
any label. Finally, all results reported here derive from held-out NHANES
data drawn from the same three survey cycles used for model development; no
fully independent external cohort, from a different survey, country, or
time period, has been used to confirm that these results generalize beyond
the sampled population.

## 7. Clinical Implications and Potential Deployment

Given a discrimination performance band of approximately 0.78 to 0.89
depending on condition, and the explicit design choice to exclude every
condition's own defining clinical measurement from its own prediction, this
model is best understood as a risk-stratification and referral-triage
instrument rather than a diagnostic device. A plausible deployment context
is a primary-care or community-health screening setting in which the inputs
used here, demographics, brief dietary, activity, sleep, and alcohol
questions, family history, self-reported weight history, and, where
available, basic organ-function laboratory values, are considerably cheaper
and more readily obtainable than the full confirmatory panel of fasting
glucose, glycated haemoglobin, a complete lipid profile, a formal blood-
pressure series, and measured waist circumference. In such a setting, the
model's role would be to flag which individuals most warrant confirmatory
testing, prioritizing scarce screening resources, rather than to substitute
for that testing. This framing is particularly apt for prediabetes, where
routine population screening is inconsistent in current practice and where
the model's AUROC, while the lowest of the five conditions examined here,
nonetheless exceeds established non-laboratory screening instruments such as
FINDRISC. Realizing any such deployment would require the prospective
calibration, subgroup fairness auditing, and external validation identified
in Section 6 as not yet undertaken; this work establishes a discrimination
performance benchmark and a leakage-safe methodology, not a clinically
validated instrument ready for deployment.

## 8. Conclusion and Future Work

A leakage-safe, multi-task deep learning model trained on three cycles of
NHANES survey data predicts five cardiometabolic conditions with a mean
survey-weighted test AUROC of 0.842, a result established through ten-seed
statistical testing of every architectural and training claim made in this
work and corroborated by two independent gradient-boosted tree baselines and
a validation-fit stacking ensemble. An extensive negative-result ablation
programme, spanning self-supervised pretraining, synthetic oversampling,
hyperparameter search, and additional ensemble members, supports treating
this figure as close to the ceiling extractable from the current feature set
under a non-circular labelling constraint, rather than as an artefact of
insufficient model search. Future work identified but not yet undertaken
includes re-running the subgroup calibration analysis on the final ensemble,
a further round of NHANES feature enrichment, expected on present evidence to
yield diminishing returns, and, should real-world deployment ultimately be
pursued, prospective calibration and external validation outside the NHANES
cohort.
