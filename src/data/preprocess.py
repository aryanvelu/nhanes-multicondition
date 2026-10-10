"""Turn the merged NHANES table into a model-ready feature matrix.

Design choices that matter downstream:

* **Continuous features are z-scored** (mean 0, unit variance) using statistics
  computed over observed values only. The fitted mean/std for every column are
  written to the schema JSON so the exact same transform is reproducible (and
  can be re-fit on a train split later to avoid leakage).

* **Categorical features stay integer-coded, not one-hot.** Each category is
  mapped to a small contiguous integer so the model can learn an *embedding*
  per level (the README's design) instead of a sparse one-hot block. Index 0 is
  reserved for "missing / unknown", so an embedding table of size = cardinality
  always has a valid row for absent values.

* **Missingness is preserved as signal.** Before any imputation, every feature
  that has missing values gets a parallel ``<name>_missing`` 0/1 column. Only
  *then* are continuous NaNs filled (with 0 = the standardized mean) and
  categorical NaNs sent to index 0. NHANES missingness is highly structured
  (e.g. fasting labs exist only for the morning subsample), so the flags carry
  real information rather than noise.

* **Physiological grouping is recorded.** Each feature is tagged with the body
  system it belongs to (glycemic / lipid / adiposity / hemodynamic /
  demographic / behavioral / lifestyle / family_history / weight_history /
  renal / hepatic / medication_use / hematologic / general_health) so the
  physiology-structured input layer
  can wire its first layer by group instead of fully-connected.

Outputs (all under ``data/processed/``):
    features.parquet      SEQN, cycle, standardized continuous cols, missing
                          flags, integer categorical codes, survey weights
    labels.parquet        (produced by labels.py; re-aligned here by SEQN+cycle)
    feature_schema.json   continuous vs categorical, cardinalities, code maps,
                          scaler params, physiological groups, passthrough cols

Usage
-----
    python -m src.data.preprocess
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_ROOT = Path("data/processed")
MERGED_PATH = PROCESSED_ROOT / "merged.parquet"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"
FEATURES_PATH = PROCESSED_ROOT / "features.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema.json"

KEY = ["SEQN", "cycle"]

# Continuous predictors, tagged with physiological group for the structured
# input layer. Note: several of these also define labels (BMI -> obesity, BP ->
# hypertension, glucose/HbA1c -> diabetes, the lipid panel -> metabolic
# syndrome). They are kept here because the project frames prediction from the
# full clinical profile; see the leakage note in the README/summary before
# using a given head as a real predictive claim.
CONTINUOUS_GROUPS: dict[str, str] = {
    "RIDAGEYR": "demographic",   # age in years
    "INDFMPIR": "demographic",   # family income-to-poverty ratio
    "BMXBMI": "adiposity",       # body mass index
    "BMXWAIST": "adiposity",     # waist circumference, cm
    "BMXWT": "adiposity",        # weight, kg
    "BMXHT": "adiposity",        # standing height, cm
    "BPXSY1": "hemodynamic",     # systolic BP, 1st reading
    "BPXDI1": "hemodynamic",     # diastolic BP, 1st reading
    "LBXGLU": "glycemic",        # fasting plasma glucose
    "LBXGH": "glycemic",         # glycohemoglobin (HbA1c)
    "LBXTC": "lipid",            # total cholesterol
    "LBDHDD": "lipid",           # HDL cholesterol
    "LBXTR": "lipid",            # triglycerides
    "LBDLDL": "lipid",           # LDL cholesterol
    # Diet: Day-1 24-hour recall totals (DR1TOT). We use Day 1 only, not the
    # Day-2 recall (DR2TOT) also downloaded/merged — the same "first reading
    # only" precedent already used for blood pressure (BPXSY1/BPXDI1 over the
    # repeated readings). Nulled out below where the recall was flagged
    # unreliable (DR1DRSTZ != 1), same treat-as-missing-not-ground-truth
    # pattern used for label validity elsewhere in this project.
    "DR1TKCAL": "lifestyle",     # energy, kcal
    "DR1TPROT": "lifestyle",     # protein, g
    "DR1TCARB": "lifestyle",     # carbohydrate, g
    "DR1TSUGR": "lifestyle",     # total sugars, g
    "DR1TFIBE": "lifestyle",     # dietary fiber, g
    "DR1TTFAT": "lifestyle",     # total fat, g
    "DR1TSFAT": "lifestyle",     # saturated fat, g
    "DR1TSODI": "lifestyle",     # sodium, mg
    "PAD680": "lifestyle",       # sedentary minutes/day (always asked, no skip gate)
    "activity_min_per_week": "lifestyle",  # derived; see derive_activity_minutes()
    # Weight History Questionnaire (WHQ): self-reported *belief* about body
    # weight, not the measured BMI (BMXBMI) that defines the obesity label.
    # Own group — conceptually distinct from measured adiposity, diet, or
    # activity. Sentinel-cleaned below (7777=refused, 9999=don't know).
    "WHD020": "weight_history",  # current self-reported weight, lb
    "WHD050": "weight_history",  # self-reported weight 1 year ago, lb
    "WHD120": "weight_history",  # self-reported weight at age 25, lb
    # Kidney function (BIOPRO). Measured labs, not self-report — no sentinel
    # cleaning needed (verified: no refused/don't-know spikes in the data).
    "LBXSCR": "renal",           # creatinine, mg/dL
    "LBXSBU": "renal",           # blood urea nitrogen, mg/dL
    # Liver function (BIOPRO).
    "LBXSATSI": "hepatic",       # ALT, U/L
    "LBXSASSI": "hepatic",       # AST, U/L
    "LBXSAPSI": "hepatic",       # alkaline phosphatase, IU/L
    "LBXSGTSI": "hepatic",       # GGT, U/L
    "LBXSTB": "hepatic",         # total bilirubin, mg/dL
    "LBXSAL": "hepatic",         # albumin, g/dL
    # Alcohol: only ALQ130 (avg drinks/day) is stable across all 3 cycles —
    # the frequency questions (ALQ120Q/U in H/I vs ALQ121 in J) were
    # restructured and aren't comparable, so they're excluded.
    "ALQ130": "lifestyle",       # avg alcoholic drinks/day, past 12mo
    # Sleep: SLD010H (H cycle) was renamed SLD012 (I/J) for the same
    # quantity — derived below into one combined `sleep_hours` column so the
    # cross-cycle rename doesn't silently create two half-populated features.
    "sleep_hours": "lifestyle",
    # Medications: RXQ_RX aggregated to one count/person in merge.py (see
    # aggregate_medications() there) since the raw file is one row per drug.
    "RXDCOUNT": "medication_use",
    # Complete Blood Count (CBC) — the closest available substitute for
    # hs-CRP, confirmed unavailable in the standard public-use files for
    # these cycles. Measured labs, no sentinel cleaning needed.
    "LBXWBCSI": "hematologic",   # white blood cell count, 1000 cells/uL
    "LBXHGB": "hematologic",     # hemoglobin, g/dL
    "LBXHCT": "hematologic",     # hematocrit, %
    "LBXPLTSI": "hematologic",   # platelet count, 1000 cells/uL
}

# WHQ pound-valued fields use the same refused/don't-know sentinel convention
# as the PAQ minute fields (_null_sentinels, defined below).
WEIGHT_HISTORY_COLS = ["WHD020", "WHD050", "WHD120"]

# Day-1 dietary recall columns nulled out when the recall wasn't reliable
# (DR1DRSTZ != 1), before they enter the generic z-score loop above.
DIET_RECALL_COLS = [
    "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR",
    "DR1TFIBE", "DR1TTFAT", "DR1TSFAT", "DR1TSODI",
]

# Physical-activity domains: (do-you-do-this, days/week, minutes/day, MET weight).
# Vigorous-intensity domains count double, mirroring the standard public-health
# framing ("150 min/week moderate or 75 min/week vigorous").
PA_DOMAINS = [
    ("PAQ605", "PAQ610", "PAD615", 2.0),  # vigorous work
    ("PAQ620", "PAQ625", "PAD630", 1.0),  # moderate work
    ("PAQ635", "PAQ640", "PAD645", 1.0),  # walk/bike for transport
    ("PAQ650", "PAQ655", "PAD660", 2.0),  # vigorous recreation
    ("PAQ665", "PAQ670", "PAD675", 1.0),  # moderate recreation
]

# Categorical predictors. Each maps raw NHANES codes -> the set of *valid*
# codes; anything else (refused/don't-know/NaN) becomes index 0 = missing.
# smoking_status is derived below rather than read from a single column.
CATEGORICAL_VALID: dict[str, list[int]] = {
    "RIAGENDR": [1, 2],            # sex: male / female
    "RIDRETH1": [1, 2, 3, 4, 5],   # race/ethnicity (1996 categories)
    "DMDEDUC2": [1, 2, 3, 4, 5],   # education (adults 20+); 7/9 -> missing
    "smoking_status": [0, 1, 2],   # never / former / current (derived)
    "MCQ300A": [1, 2],             # close relative had a heart attack: yes/no
    "MCQ300C": [1, 2],             # close relative had diabetes: yes/no
    "SLQ050": [1, 2],               # ever told doctor about trouble sleeping: yes/no
    "HSD010": [1, 2, 3, 4, 5],       # self-rated general health: excellent...poor
}
CATEGORICAL_GROUPS: dict[str, str] = {
    "RIAGENDR": "demographic",
    "RIDRETH1": "demographic",
    "DMDEDUC2": "demographic",
    "smoking_status": "behavioral",
    "MCQ300A": "family_history",
    "MCQ300C": "family_history",
    "SLQ050": "lifestyle",
    "HSD010": "general_health",
}

# Survey weights carried through untouched (used for weighted training/eval,
# never fed to the network as a feature).
PASSTHROUGH = ["WTINT2YR", "WTMEC2YR", "WTSAF2YR"]


def derive_smoking_status(df: pd.DataFrame) -> pd.Series:
    """0 = never, 1 = former, 2 = current, NaN = unusable/not asked.

    SMQ020: smoked >=100 cigarettes in life (1 yes / 2 no).
    SMQ040: do you now smoke (1 every day / 2 some days / 3 not at all).
    """
    smq020 = df.get("SMQ020", pd.Series(np.nan, index=df.index))
    smq040 = df.get("SMQ040", pd.Series(np.nan, index=df.index))
    status = pd.Series(np.nan, index=df.index, dtype="float64")
    status[smq020 == 2] = 0                       # never smoked
    status[(smq020 == 1) & (smq040 == 3)] = 1     # former
    status[(smq020 == 1) & smq040.isin([1, 2])] = 2  # current
    return status


def _null_sentinels(s: pd.Series, sentinels: set[float]) -> pd.Series:
    """NHANES marks refused/don't-know on numeric fields with sentinel values
    (77/99 for day counts, 7777/9999 for minutes) rather than leaving them
    blank — coerce those to real NaN so they don't get treated as data."""
    s = pd.to_numeric(s, errors="coerce")
    return s.mask(s.isin(sentinels))


def derive_activity_minutes(df: pd.DataFrame) -> pd.Series:
    """Intensity-weighted physical-activity minutes/week across 5 domains.

    Each domain (e.g. "vigorous work") is asked as do-you-do-this-at-all
    (1=yes/2=no), then days/week and minutes/day *only if yes* (a skip
    pattern, so days/minutes are legitimately NaN for a "no" answer). Domain
    minutes = days*minutes if yes, 0 if no, NaN if not ascertained at all.
    Vigorous domains count double, matching the standard "150 min/week
    moderate or 75 min/week vigorous" public-health framing. The total is NaN
    only if every domain is unknown; otherwise known domains are summed and
    unknown domains simply don't contribute (documented approximation).
    """
    domain_minutes = []
    for do_col, days_col, min_col, weight in PA_DOMAINS:
        do = df.get(do_col, pd.Series(np.nan, index=df.index))
        days = _null_sentinels(df.get(days_col, pd.Series(np.nan, index=df.index)), {77, 99})
        mins = _null_sentinels(df.get(min_col, pd.Series(np.nan, index=df.index)), {7777, 9999})
        minutes = pd.Series(np.nan, index=df.index, dtype="float64")
        minutes[do == 2] = 0.0
        minutes[do == 1] = (days * mins)[do == 1]
        domain_minutes.append(minutes * weight)

    stacked = pd.concat(domain_minutes, axis=1)
    total = stacked.sum(axis=1, skipna=True)
    total[stacked.isna().all(axis=1)] = np.nan
    return total


def main() -> int:
    df = pd.read_parquet(MERGED_PATH)
    n = len(df)
    print(f"Loaded {MERGED_PATH}: {n:,} participants")

    df = df.copy()
    df["smoking_status"] = derive_smoking_status(df)
    df["PAD680"] = _null_sentinels(df["PAD680"], {7777, 9999})  # refused/don't-know
    df["activity_min_per_week"] = derive_activity_minutes(df)
    for col in WEIGHT_HISTORY_COLS:
        df[col] = _null_sentinels(df[col], {7777, 9999})  # refused/don't-know
    df["ALQ130"] = _null_sentinels(df["ALQ130"], {777, 999})  # refused/don't-know
    # SLD010H (H cycle) was renamed SLD012 (I/J) for the same "hours of
    # sleep" quantity — each cycle only populates one of the two columns, so
    # combine_first coalesces them into a single cross-cycle feature. 99 is
    # SLD010H's refused/don't-know sentinel (12 is a legitimate "12+" top
    # code, left alone); SLD012 has no sentinel (NCHS-computed, not a raw
    # interview response).
    df["SLD010H"] = _null_sentinels(df["SLD010H"], {99})
    df["sleep_hours"] = df["SLD012"].combine_first(df["SLD010H"])

    # Unreliable/incomplete Day-1 dietary recalls (DR1DRSTZ != 1, e.g. "not
    # reliable" or "not done") shouldn't be treated as real intake data — null
    # them out here so they get a missing flag + mean-impute below instead of
    # silently corrupting the z-score with bad values.
    reliable = df["DR1DRSTZ"] == 1
    n_unreliable = int((~reliable).sum())
    for col in DIET_RECALL_COLS:
        df[col] = df[col].where(reliable)
    print(f"Dietary recall reliability gate (DR1DRSTZ==1): nulled {n_unreliable:,} "
          f"of {n:,} rows' Day-1 nutrient totals (unreliable/incomplete/not-done recall)\n")

    # Guard against any expected column missing from the merge.
    missing_cols = [
        c for c in list(CONTINUOUS_GROUPS) + list(CATEGORICAL_VALID) + PASSTHROUGH
        if c not in df.columns
    ]
    if missing_cols:
        raise KeyError(f"expected columns absent from merged data: {missing_cols}")

    out = df[KEY].copy()
    schema: dict = {
        "n_rows": n,
        "continuous": {},
        "categorical": {},
        "missing_flags": [],
        "passthrough": PASSTHROUGH,
    }

    # ---- continuous: z-score, then flag + impute --------------------------- #
    print("\nContinuous features (z-scored; missing flags added pre-impute):")
    for col, group in CONTINUOUS_GROUPS.items():
        raw = pd.to_numeric(df[col], errors="coerce")
        mean = float(raw.mean())
        std = float(raw.std(ddof=0))
        scale = std if std > 0 else 1.0

        n_missing = int(raw.isna().sum())
        if n_missing:
            out[f"{col}_missing"] = raw.isna().astype("int8")
            schema["missing_flags"].append(f"{col}_missing")

        z = (raw - mean) / scale
        out[col] = z.fillna(0.0).astype("float32")  # 0 == standardized mean

        schema["continuous"][col] = {
            "group": group,
            "mean": mean,
            "std": std,
            "n_missing": n_missing,
        }
        print(f"  {col:<10} group={group:<12} mean={mean:8.2f} std={std:8.2f} "
              f"missing={n_missing:>6,}")

    # ---- categorical: contiguous integer codes (0 = missing) --------------- #
    print("\nCategorical features (integer-coded for embeddings; 0 = missing):")
    for col, valid in CATEGORICAL_VALID.items():
        raw = pd.to_numeric(df[col], errors="coerce")
        # Map each valid raw code -> 1..K (0 reserved for missing/unknown).
        code_map = {int(v): i + 1 for i, v in enumerate(valid)}
        codes = raw.map(code_map)

        n_missing = int(codes.isna().sum())
        if n_missing:
            out[f"{col}_missing"] = codes.isna().astype("int8")
            schema["missing_flags"].append(f"{col}_missing")

        out[col] = codes.fillna(0).astype("int64")
        cardinality = len(valid) + 1  # +1 for the reserved 0 = missing index

        schema["categorical"][col] = {
            "group": CATEGORICAL_GROUPS[col],
            "cardinality": cardinality,
            "code_map": {str(k): v for k, v in code_map.items()},
            "missing_index": 0,
            "n_missing": n_missing,
        }
        print(f"  {col:<15} group={CATEGORICAL_GROUPS[col]:<12} "
              f"cardinality={cardinality} (incl. 0=missing)  missing={n_missing:>6,}")

    # ---- survey weights (passthrough) ------------------------------------- #
    for col in PASSTHROUGH:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    # ---- align labels to the exact feature row order ---------------------- #
    labels = pd.read_parquet(LABELS_PATH)
    merged_check = out[KEY].merge(labels, on=KEY, how="left", validate="one_to_one")
    assert len(merged_check) == len(out), "label alignment changed row count"
    # Re-write labels in the same row order so features/labels are index-aligned.
    labels_aligned = merged_check
    labels_aligned.to_parquet(LABELS_PATH, index=False)

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEATURES_PATH, index=False)
    with open(SCHEMA_PATH, "w") as fh:
        json.dump(schema, fh, indent=2)

    # ---- summary ---------------------------------------------------------- #
    n_cont = len(schema["continuous"])
    n_cat = len(schema["categorical"])
    n_flags = len(schema["missing_flags"])
    print("\n=== Summary ===")
    print(f"  feature matrix : {out.shape[0]:,} rows x {out.shape[1]} cols")
    print(f"  continuous     : {n_cont}")
    print(f"  categorical    : {n_cat}")
    print(f"  missing flags  : {n_flags}")
    print(f"  passthrough    : {len(PASSTHROUGH)} (survey weights)")
    print(f"\n  saved -> {FEATURES_PATH}")
    print(f"  saved -> {SCHEMA_PATH}")
    print(f"  labels re-aligned -> {LABELS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
