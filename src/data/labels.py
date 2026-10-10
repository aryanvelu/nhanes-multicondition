"""Compute the five diagnostic labels from merged NHANES values.

For every participant this produces two aligned matrices:

* **labels**  — five binary columns: ``diabetes``, ``prediabetes``,
  ``obesity``, ``hypertension``, ``metabolic_syndrome``.
* **valid**   — five columns of the same names with a ``_valid`` suffix, where
  1 means the label could actually be computed from that participant's
  available fields and 0 means it could not.

The validity mask is the whole point of not dropping rows. NHANES only draws
fasting bloods (glucose, triglycerides) on a morning subsample, so metabolic
syndrome — which needs the full NCEP ATP III panel — is only computable for
~6k people, while obesity (BMI) or a self-reported diabetes diagnosis is
available for nearly everyone. Rather than throwing away the ~19k who lack
fasting labs, each label is computed from whatever criteria that person has,
and the mask records where it was and wasn't possible. Downstream training can
then mask the loss per-label instead of per-row.

A label is only marked positive where it is also valid; invalid rows are 0/0
(not positive, not computable) so the two matrices are always consistent.

Thresholds
----------
diabetes            fasting glucose >= 126 mg/dL, or HbA1c >= 6.5 %,
                    or self-reported diagnosis (DIQ010 == 1)
prediabetes         glucose 100-125, or HbA1c 5.7-6.4, or self-reported
                    "borderline" (DIQ010 == 3); excluded if already diabetic
obesity             BMI >= 30
hypertension        SBP >= 130 or DBP >= 80 (ACC/AHA 2017), or on
                    antihypertensive medication (BPQ050A == 1)
metabolic syndrome  >= 3 of 5 NCEP ATP III criteria (waist, triglycerides,
                    HDL, BP >= 130/85, fasting glucose >= 100); requires the
                    complete panel, so only valid for the fasting subsample

Usage
-----
    python -m src.data.labels
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROCESSED_ROOT = Path("data/processed")
MERGED_PATH = PROCESSED_ROOT / "merged.parquet"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"

LABEL_NAMES = [
    "diabetes",
    "prediabetes",
    "obesity",
    "hypertension",
    "metabolic_syndrome",
]

# DIQ010 self-report codes: 1=Yes, 2=No, 3=Borderline, 7=Refused, 9=Don't know.
# Only 1/2/3 are informative; 7/9/NaN are treated as "not asked / unusable".
DIQ_USABLE = [1, 2, 3]

# Every raw NHANES column the label logic reads passes through _col(), so this
# set is populated as a side effect of compute_labels() and becomes the
# authoritative list of *label-defining* columns. Downstream leakage removal
# uses this rather than a hand-maintained list, so it can never drift out of
# sync with the thresholds actually used here.
REFERENCED_COLUMNS: set[str] = set()


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column if present, else an all-NaN float column (schema-safe)."""
    REFERENCED_COLUMNS.add(name)
    if name in df.columns:
        return df[name]
    return pd.Series(pd.NA, index=df.index, dtype="Float64")


def label_defining_columns(df: pd.DataFrame) -> set[str]:
    """Return the exact set of raw columns the label logic reads from `df`.

    Runs the real label computation so the result reflects the current
    thresholds, not a copy of them. This is the source of truth for which
    features must be dropped to avoid label leakage.
    """
    REFERENCED_COLUMNS.clear()
    compute_labels(df)
    return set(REFERENCED_COLUMNS)


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of the 5 labels plus their 5 `_valid` masks."""
    out = pd.DataFrame(index=df.index)

    sex = _col(df, "RIAGENDR")          # 1 = male, 2 = female
    male = sex == 1
    female = sex == 2

    glu = _col(df, "LBXGLU")            # fasting plasma glucose, mg/dL
    a1c = _col(df, "LBXGH")             # glycohemoglobin (HbA1c), %
    diq = _col(df, "DIQ010")            # self-reported diabetes dx

    glu_ok = glu.notna()
    a1c_ok = a1c.notna()
    sr_ok = diq.isin(DIQ_USABLE)

    # ---- diabetes -------------------------------------------------------- #
    dia_valid = glu_ok | a1c_ok | sr_ok
    dia_pos = (
        (glu_ok & (glu >= 126))
        | (a1c_ok & (a1c >= 6.5))
        | (sr_ok & (diq == 1))
    )
    diabetes = (dia_pos & dia_valid).fillna(False)

    # ---- prediabetes (only among the non-diabetic) ----------------------- #
    pre_valid = glu_ok | a1c_ok | sr_ok
    pre_pos = (
        (glu_ok & (glu >= 100) & (glu < 126))
        | (a1c_ok & (a1c >= 5.7) & (a1c < 6.5))
        | (sr_ok & (diq == 3))
    )
    prediabetes = (pre_pos & pre_valid & ~diabetes).fillna(False)

    # ---- obesity --------------------------------------------------------- #
    bmi = _col(df, "BMXBMI")
    obe_valid = bmi.notna()
    obesity = (obe_valid & (bmi >= 30)).fillna(False)

    # ---- hypertension ---------------------------------------------------- #
    sbp = _col(df, "BPXSY1")
    dbp = _col(df, "BPXDI1")
    htn_med = _col(df, "BPQ050A") == 1  # currently taking antihypertensive rx
    sbp_ok, dbp_ok = sbp.notna(), dbp.notna()
    # Valid if BP was measured, or we have a positive medication report.
    htn_valid = sbp_ok | dbp_ok | htn_med
    htn_pos = (sbp_ok & (sbp >= 130)) | (dbp_ok & (dbp >= 80)) | htn_med
    hypertension = (htn_pos & htn_valid).fillna(False)

    # ---- metabolic syndrome: NCEP ATP III, needs the full panel ---------- #
    waist = _col(df, "BMXWAIST")
    trig = _col(df, "LBXTR")
    hdl = _col(df, "LBDHDD")
    gender_ok = male | female
    bp_ok = sbp_ok | dbp_ok

    mets_valid = (
        waist.notna() & trig.notna() & hdl.notna() & glu_ok & bp_ok & gender_ok
    )
    # Each of the 5 ATP III components (evaluated everywhere; only counted where
    # mets_valid, so component NaNs never leak into a positive call).
    c_waist = ((male & (waist > 102)) | (female & (waist > 88))).fillna(False)
    c_trig = (trig >= 150).fillna(False)
    c_hdl = ((male & (hdl < 40)) | (female & (hdl < 50))).fillna(False)
    c_bp = ((sbp >= 130) | (dbp >= 85) | htn_med).fillna(False)
    c_glu = (glu >= 100).fillna(False)
    mets_count = (
        c_waist.astype(int)
        + c_trig.astype(int)
        + c_hdl.astype(int)
        + c_bp.astype(int)
        + c_glu.astype(int)
    )
    metabolic_syndrome = (mets_valid & (mets_count >= 3)).fillna(False)

    # ---- assemble -------------------------------------------------------- #
    labels = {
        "diabetes": (diabetes, dia_valid),
        "prediabetes": (prediabetes, pre_valid),
        "obesity": (obesity, obe_valid),
        "hypertension": (hypertension, htn_valid),
        "metabolic_syndrome": (metabolic_syndrome, mets_valid),
    }
    for name, (label, valid) in labels.items():
        out[name] = label.astype("int8")
        out[f"{name}_valid"] = valid.fillna(False).astype("int8")

    return out


def main() -> int:
    df = pd.read_parquet(MERGED_PATH)
    n = len(df)
    print(f"Loaded {MERGED_PATH}: {n:,} participants\n")

    # Sanity check that the survey weights survived the merge intact.
    weight_cols = [c for c in ("WTINT2YR", "WTMEC2YR", "WTSAF2YR") if c in df.columns]
    print("Survey-weight columns present (NaN counts):")
    print(df[weight_cols].isna().sum().to_string())
    print()

    labels = compute_labels(df)
    labels.insert(0, "cycle", df["cycle"].values)
    labels.insert(0, "SEQN", df["SEQN"].values)

    print("Per-label validity and prevalence")
    print(f"{'label':<20}{'valid n':>9}{'positives':>11}{'prev(valid)':>13}"
          f"{'prev(all)':>11}")
    for name in LABEL_NAMES:
        valid = labels[f"{name}_valid"] == 1
        n_valid = int(valid.sum())
        pos = int(labels.loc[valid, name].sum())
        prev_valid = pos / n_valid if n_valid else 0.0
        prev_all = pos / n
        print(f"{name:<20}{n_valid:>9,}{pos:>11,}{prev_valid:>12.1%}{prev_all:>11.1%}")

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(LABELS_PATH, index=False)
    print(f"\nsaved -> {LABELS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
