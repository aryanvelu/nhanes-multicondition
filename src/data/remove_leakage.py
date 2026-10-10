"""Remove label-leaking features from the modeling matrix.

Every raw measurement that ``labels.py`` reads to *compute* a label is, by
definition, a giveaway for that label (BMI -> obesity, BP -> hypertension,
fasting glucose / HbA1c -> diabetes, the waist/triglyceride/HDL panel ->
metabolic syndrome). Keeping them as model inputs makes several heads trivial
and would not survive peer review.

The list of columns to drop is taken directly from ``labels.py`` — specifically
:func:`src.data.labels.label_defining_columns`, which returns exactly the
columns the label logic touched when it last ran. Using that as the source of
truth means this step can never fall out of sync with the thresholds: change a
criterion in labels.py and the dropped set updates automatically.

Two refinements on top of the raw referenced set:

* Only *continuous* features are dropped. ``RIAGENDR`` (sex) is referenced by
  labels.py, but only to pick sex-specific waist/HDL cutoffs — it is a
  legitimate demographic predictor, not a leaked measurement, so it stays.
  Restricting removal to continuous features excludes it cleanly (it, and the
  questionnaire columns DIQ010/BPQ050A, are not in the feature matrix anyway).
* ``BMXWT`` and ``BMXHT`` are dropped even though labels.py never reads them:
  weight and height together reconstruct BMI (weight / height^2) almost exactly,
  which would reintroduce the obesity leak through the back door.

Output: ``features_no_leakage.parquet`` and ``feature_schema_no_leakage.json``
(a schema matching the reduced matrix; the original files are left untouched).

Usage
-----
    python -m src.data.remove_leakage
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data import labels as labels_mod

PROCESSED_ROOT = Path("data/processed")
MERGED_PATH = PROCESSED_ROOT / "merged.parquet"
FEATURES_PATH = PROCESSED_ROOT / "features.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema.json"
OUT_FEATURES_PATH = PROCESSED_ROOT / "features_no_leakage.parquet"
OUT_SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_no_leakage.json"

# Weight + height reconstruct BMI, so they leak obesity even though labels.py
# never reads them directly.
BMI_RECONSTRUCTION = {"BMXWT", "BMXHT"}


def group_counts(schema: dict) -> Counter:
    """Count modeling features (continuous + categorical) per physiology group."""
    counts: Counter = Counter()
    for spec in schema["continuous"].values():
        counts[spec["group"]] += 1
    for spec in schema["categorical"].values():
        counts[spec["group"]] += 1
    return counts


def main() -> int:
    features = pd.read_parquet(FEATURES_PATH)
    schema = json.loads(SCHEMA_PATH.read_text())
    merged = pd.read_parquet(MERGED_PATH)

    # Source of truth: whatever labels.py actually read to build the labels.
    referenced = labels_mod.label_defining_columns(merged)
    continuous = set(schema["continuous"])

    # Drop referenced continuous features + the BMI-reconstruction pair.
    to_drop = (referenced & continuous) | (BMI_RECONSTRUCTION & continuous)

    print(f"labels.py referenced {len(referenced)} raw columns: "
          f"{sorted(referenced)}")
    print(f"of those, in the continuous feature set: {sorted(referenced & continuous)}")
    print(f"+ BMI-reconstruction pair: {sorted(BMI_RECONSTRUCTION & continuous)}")
    print(f"=> dropping {len(to_drop)} features: {sorted(to_drop)}\n")

    before = group_counts(schema)

    # Build the reduced schema.
    new_schema = {
        "n_rows": schema["n_rows"],
        "continuous": {k: v for k, v in schema["continuous"].items() if k not in to_drop},
        "categorical": dict(schema["categorical"]),
        "missing_flags": [
            f for f in schema["missing_flags"]
            if f.rsplit("_missing", 1)[0] not in to_drop
        ],
        "passthrough": schema["passthrough"],
        "removed_for_leakage": sorted(to_drop),
        "leakage_source": "src.data.labels.label_defining_columns + BMI reconstruction (BMXWT, BMXHT)",
    }

    # Drop the feature columns and their missing-flag partners.
    drop_cols = list(to_drop) + [f"{c}_missing" for c in to_drop
                                 if f"{c}_missing" in features.columns]
    reduced = features.drop(columns=drop_cols)

    OUT_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_parquet(OUT_FEATURES_PATH, index=False)
    OUT_SCHEMA_PATH.write_text(json.dumps(new_schema, indent=2))

    after = group_counts(new_schema)

    # ---- report ----------------------------------------------------------- #
    print("Feature count by physiological group (before -> after):")
    all_groups = sorted(set(before) | set(after))
    print(f"  {'group':<14}{'before':>8}{'after':>8}")
    for g in all_groups:
        print(f"  {g:<14}{before[g]:>8}{after[g]:>8}")
    print(f"  {'TOTAL':<14}{sum(before.values()):>8}{sum(after.values()):>8}")

    print(f"\n  feature matrix : {features.shape[1]} -> {reduced.shape[1]} columns "
          f"(incl. keys/flags/weights)")
    print(f"  saved -> {OUT_FEATURES_PATH}")
    print(f"  saved -> {OUT_SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
