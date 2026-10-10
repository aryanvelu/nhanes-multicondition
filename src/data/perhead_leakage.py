"""Per-head (middle-ground) leakage control.

The *strict* policy (`remove_leakage.py`) drops every label-defining measurement
from the whole model, which leaves only 8 features and empties three of the six
physiological groups — starving the structured layer and pretraining of the very
structure they exist to exploit.

The middle ground keeps **all** features in play, but masks leakage *per head*:
before a given condition's head runs, we hide only the features that define
*that* condition's label, leaving every other group visible. So:

* **diabetes / prediabetes** can't see glucose or HbA1c, but do see BMI, blood
  pressure, and lipids;
* **obesity** can't see any body-composition feature (BMI, waist, weight,
  height — the whole adiposity group, since weight+height rebuild BMI), but sees
  everything else;
* **hypertension** can't see blood pressure, but sees the rest;
* **metabolic syndrome** is masked across most groups, because its NCEP ATP III
  definition spans waist, triglycerides, HDL, blood pressure, and fasting
  glucose — so relatively little is left visible to it.

This module does not train anything. It validates the per-head mask against the
authoritative label-defining column list (`labels.label_defining_columns`) and
against the full feature schema, then writes ``feature_schema_perhead.json`` —
the full feature schema (all six groups repopulated) plus a ``head_masks`` map
that ``finetune.py --per-head`` reads. The feature matrix itself is the full
``features.parquet`` produced by ``preprocess.py`` (nothing to regenerate).

Usage
-----
    python -m src.data.perhead_leakage
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data import labels as labels_mod
from src.data.labels import LABEL_NAMES

PROCESSED_ROOT = Path("data/processed")
MERGED_PATH = PROCESSED_ROOT / "merged.parquet"
FULL_SCHEMA_PATH = PROCESSED_ROOT / "feature_schema.json"
OUT_SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_perhead.json"

# Features to hide before each head's forward pass. Derived from each label's
# clinical definition in labels.py. Obesity masks the whole adiposity group
# (weight+height reconstruct BMI); metabolic syndrome masks its full ATP III
# panel across groups. Only continuous features appear here — the sole
# label-defining categorical (sex) is a threshold modifier, not a leak.
#
# WHD020 (current self-reported weight) and WHD050 (weight a year ago) are
# masked for obesity only, same reasoning as BMXWT/BMXHT: neither is read by
# labels.py (label_defining_columns() never flags either), but both are
# strong non-deterministic proxies for current body size — too close to
# circular for the one head whose label BMI-defines "large body size."
# WHD050 was initially left visible on the theory that "a year ago" isn't
# "now," but an experiment (FT-Transformer, 10 seeds, results/ft_whq_sweep.md)
# showed obesity AUROC jumping +0.160 with every other head flat — the
# signature of a proxy leak, not a modeling gain — traced to WHD050's 0.556
# correlation with the obesity label (nearly as strong as WHD020's 0.599):
# most adults' weight barely changes in a year. WHD120 (weight at age 25,
# correlation 0.275 — decades of possible change intervene) stays visible to
# every head, including obesity, as the one weight-history feature that
# actually behaves like history rather than a near-restatement of the label.
#
# RXDCOUNT (total prescription medication count, aggregated in merge.py from
# the one-row-per-drug RXQ_RX file) is masked for diabetes and hypertension
# only. Not read by labels.py, but both labels have "on medication for this
# condition" as one of their own defining OR-conditions (DIQ010 self-report
# correlates with diabetes-medication use; BPQ050A *is* antihypertensive-
# medication use directly) — RXDCOUNT correlates 0.476 with diabetes and
# 0.448 with hypertension (vs. 0.03-0.26 for every other new feature this
# round), and self-reported diabetics average 1.64 std. medications vs -0.15
# for non-diabetics, so a meaningful share of the signal is treatment-status
# circularity, not risk prediction. Stays visible to obesity/prediabetes/
# metabolic_syndrome, where its correlation (0.09-0.36) looks like genuine
# comorbidity/disease-burden signal rather than a restatement of the label.
HEAD_MASKED_FEATURES: dict[str, list[str]] = {
    "diabetes":           ["LBXGLU", "LBXGH", "RXDCOUNT"],
    "prediabetes":        ["LBXGLU", "LBXGH"],
    "obesity":            ["BMXBMI", "BMXWAIST", "BMXWT", "BMXHT", "WHD020", "WHD050"],
    "hypertension":       ["BPXSY1", "BPXDI1", "RXDCOUNT"],
    "metabolic_syndrome": ["BMXWAIST", "LBXTR", "LBDHDD", "BPXSY1", "BPXDI1", "LBXGLU"],
}


def main() -> int:
    schema = json.loads(FULL_SCHEMA_PATH.read_text())
    cont = set(schema["continuous"])
    cat = set(schema["categorical"])
    merged = pd.read_parquet(MERGED_PATH)
    referenced = labels_mod.label_defining_columns(merged)

    # --- validation --------------------------------------------------------- #
    assert set(HEAD_MASKED_FEATURES) == set(LABEL_NAMES), "head map must cover all labels"

    # Every masked feature must be a real continuous feature.
    for label, feats in HEAD_MASKED_FEATURES.items():
        bad = [f for f in feats if f not in cont]
        assert not bad, f"{label}: masked features not in continuous schema: {bad}"

    # Completeness: every continuous label-defining feature must be masked by at
    # least one head, so nothing that defines a label is ever left fully visible.
    referenced_cont = referenced & cont
    union_masked = set().union(*HEAD_MASKED_FEATURES.values())
    missed = referenced_cont - union_masked
    assert not missed, f"continuous label-defining features never masked: {missed}"

    # Report what each head can and cannot see.
    cont_group = {c: schema["continuous"][c]["group"] for c in schema["continuous"]}
    print("Per-head leakage masks (features hidden before each head runs):\n")
    print(f"{'head':<20}{'# masked':>9}  masked features (group)")
    for label in LABEL_NAMES:
        feats = HEAD_MASKED_FEATURES[label]
        pretty = ", ".join(f"{f}[{cont_group[f]}]" for f in feats)
        print(f"{label:<20}{len(feats):>9}  {pretty}")

    all_groups = sorted(set(cont_group.values())
                        | {schema["categorical"][c]["group"] for c in schema["categorical"]})
    print("\nGroups still visible to each head:")
    for label in LABEL_NAMES:
        masked = set(HEAD_MASKED_FEATURES[label])
        visible = []
        for g in all_groups:
            g_cont = [c for c in cont if cont_group[c] == g]
            g_cat = [c for c in cat if schema["categorical"][c]["group"] == g]
            visible_cont = [c for c in g_cont if c not in masked]
            if visible_cont or g_cat:
                visible.append(g)
        print(f"  {label:<20} {visible}")

    print(f"\ncross-check: continuous label-defining features {sorted(referenced_cont)}")
    print(f"             all covered by some head's mask -> {not missed}")

    # --- write schema ------------------------------------------------------- #
    out = dict(schema)
    out["head_masks"] = HEAD_MASKED_FEATURES
    out["leakage_mode"] = "per-head (middle-ground)"
    out["feature_matrix"] = "data/processed/features.parquet"  # full repopulated set
    OUT_SCHEMA_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUT_SCHEMA_PATH}")
    print(f"  full feature set: {len(cont)} continuous + {len(cat)} categorical "
          f"across {len(all_groups)} groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
