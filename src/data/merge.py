"""Merge raw NHANES modules into one participant-level table per cycle.

Within each survey cycle, every module is a separate ``.XPT`` file keyed by the
participant sequence number ``SEQN``. This script joins them into a single wide
row per participant, then stacks the cycles into one long dataframe.

Two things matter for correctness:

1. **Cycle isolation.** ``SEQN`` is only unique *within* a cycle — the same
   integer identifies different people in 2015-2016 vs 2017-2018. We therefore
   merge strictly within a cycle and stamp an explicit ``cycle`` column before
   concatenating, so records from different survey periods can never collide.

2. **Demographics is the spine, joins are LEFT.** DEMO covers everyone who was
   interviewed. Lab modules like fasting glucose (GLU) and the LDL/triglyceride
   file (TRIGLY) only cover the morning fasting subsample. Joining those as
   INNER joins would silently delete every non-fasting participant. We LEFT-join
   everything onto DEMO instead, so subsample non-membership surfaces as NaN in
   the lab columns rather than as a dropped row — which is exactly the signal
   downstream label / missingness handling needs.

The per-step report prints, for each module, how many DEMO participants it
covers and how many fall out into NaN, so it is obvious where people are lost
(e.g. questionnaire respondents who were never in the fasting subsample).

Usage
-----
    python -m src.data.merge
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat

from src.data.download import CYCLES, MODULES, RAW_ROOT

SEQN = "SEQN"
BASE_MODULE = "DEMO"  # demographics is the left-hand spine for every join
PROCESSED_ROOT = Path("data/processed")
OUTPUT_PATH = PROCESSED_ROOT / "merged.parquet"

# RXQ_RX is one row per medication, not per person (SEQN repeats for anyone
# on multiple drugs) — the only module here shaped that way. It's excluded
# from the generic per-module loop below (which assumes one row per SEQN,
# enforced by validate="one_to_one") and aggregated separately instead.
MEDICATION_MODULE = "RXQ_RX"


def load_module(cycle: str, base: str) -> pd.DataFrame:
    """Read one module's .XPT for a cycle and return it with an integer SEQN."""
    suffix = CYCLES[cycle]
    path = RAW_ROOT / cycle / f"{base}_{suffix}.XPT"
    try:
        df, _meta = pyreadstat.read_xport(str(path))
    except UnicodeDecodeError:
        # Free-text fields (e.g. RXQ_RX drug names) occasionally have
        # non-UTF-8 bytes; latin1 never raises and every other module here
        # is numeric-coded, so this fallback only ever engages when needed.
        df, _meta = pyreadstat.read_xport(str(path), encoding="latin1")
    if SEQN not in df.columns:
        raise KeyError(f"{path} has no {SEQN} column")
    # SEQN is stored as a float in transport files; make it a clean integer key.
    df[SEQN] = df[SEQN].astype("int64")
    return df


def aggregate_medications(cycle: str) -> pd.DataFrame:
    """RXQ_RX -> one row per person: total prescription medication count.

    ``RXDUSE`` (1=yes/2=no/7=refused/9=don't know) is repeated on every one
    of a person's drug rows, so grouping by SEQN and taking the first value
    recovers it cleanly. In the raw file ``RXDCOUNT`` is only populated for
    RXDUSE==1 — a "no medications" participant still gets exactly one
    placeholder row, but with RXDCOUNT blank, not 0. Left as NaN, that would
    wrongly look identical to "didn't answer"; it's filled to a real 0 here.
    RXDUSE 7/9 (refused/don't know) stay NaN — genuinely unknown, not zero.
    """
    raw = load_module(cycle, MEDICATION_MODULE)
    first = raw.groupby(SEQN, as_index=False)[["RXDUSE", "RXDCOUNT"]].first()
    first["RXDCOUNT"] = first["RXDCOUNT"].mask(first["RXDUSE"] == 2, 0.0)
    first.loc[~first["RXDUSE"].isin([1, 2]), "RXDCOUNT"] = float("nan")
    return first[[SEQN, "RXDCOUNT"]]


def merge_cycle(cycle: str) -> pd.DataFrame:
    """Left-join every module onto DEMO for a single cycle, reporting drops."""
    print(f"\n=== Cycle {cycle} ===")
    base = load_module(cycle, BASE_MODULE)
    base_ids = set(base[SEQN])
    print(f"  base {BASE_MODULE:<7} {len(base):>6,} participants (spine)")

    merged = base
    for module in MODULES:
        if module.base in (BASE_MODULE, MEDICATION_MODULE):
            continue

        right = load_module(cycle, module.base)
        right_ids = set(right[SEQN])

        # Coverage relative to the DEMO spine.
        covered = len(base_ids & right_ids)      # DEMO people this module has
        missing = len(base) - covered            # DEMO people -> NaN after join
        orphan = len(right_ids - base_ids)        # in module but not DEMO (dropped)

        # Avoid non-key column collisions (e.g. WTSAF2YR appears in both GLU and
        # TRIGLY). Keep the first occurrence; drop later duplicates from the
        # right side so the merge doesn't create _x / _y suffixed columns.
        dupes = [c for c in right.columns if c != SEQN and c in merged.columns]
        if dupes:
            right = right.drop(columns=dupes)

        merged = merged.merge(right, on=SEQN, how="left", validate="one_to_one")

        added = right.shape[1] - 1  # columns contributed (minus SEQN)
        note = f" | dropped-dupe cols: {dupes}" if dupes else ""
        orphan_note = f" | {orphan:,} not in DEMO (dropped)" if orphan else ""
        print(
            f"  +{module.base:<7} {covered:>6,} matched, "
            f"{missing:>6,} -> NaN  (+{added} cols){orphan_note}{note}"
        )

    # RXQ_RX: aggregated separately (one row per medication upstream), then
    # left-joined onto DEMO the same way as every other module.
    meds = aggregate_medications(cycle)
    meds_ids = set(meds[SEQN])
    covered = len(base_ids & meds_ids)
    missing = len(base) - covered
    merged = merged.merge(meds, on=SEQN, how="left", validate="one_to_one")
    print(f"  +{MEDICATION_MODULE:<7} {covered:>6,} matched, "
         f"{missing:>6,} -> NaN  (+1 col, aggregated: RXDCOUNT = total meds/person)")

    merged.insert(1, "cycle", cycle)
    print(f"  result: {merged.shape[0]:,} rows x {merged.shape[1]} cols")
    return merged


def main() -> int:
    frames = [merge_cycle(cycle) for cycle in sorted(CYCLES)]

    combined = pd.concat(frames, ignore_index=True)

    # SEQN is only unique within a cycle, so uniqueness is (cycle, SEQN).
    assert not combined.duplicated([SEQN, "cycle"]).any(), "duplicate (SEQN, cycle)"

    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)

    print("\n=== Combined ===")
    print(f"  cycles concatenated : {len(frames)}")
    print(f"  total participants  : {combined.shape[0]:,}")
    print(f"  total columns       : {combined.shape[1]}")
    print("  per-cycle counts    :")
    for cycle, n in combined["cycle"].value_counts().sort_index().items():
        print(f"    {cycle}: {n:,}")
    # Quick label-availability probe: fasting-glucose coverage is the tightest
    # gate for the metabolic-syndrome / diabetes fasting criteria.
    if "LBXGLU" in combined.columns:
        n_glu = combined["LBXGLU"].notna().sum()
        print(f"  fasting glucose (LBXGLU) present: {n_glu:,} / {len(combined):,}")
    print(f"\n  saved -> {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
