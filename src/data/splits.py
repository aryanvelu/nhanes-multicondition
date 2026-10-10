"""Create cluster-aware train/val/test splits (70/15/15).

Why not a plain random split? NHANES is a *clustered* sample: participants are
drawn from a limited number of primary sampling units (PSUs) — roughly,
geographic areas — nested inside sampling strata. People in the same PSU are
correlated (same communities, clinics, environment). If members of one PSU land
in both the training and test sets, the model can memorize cluster-specific
quirks and the test score is optimistic. So we split at the **cluster** level:
an entire PSU goes wholly into one of train/val/test, never split apart.

A cluster is identified by ``(cycle, SDMVSTRA, SDMVPSU)``:
    SDMVSTRA — the masked design **stratum** id.
    SDMVPSU  — the masked **PSU** id, numbered *within* a stratum (so it is only
               meaningful together with its stratum, and cycle).

Clusters are shuffled with a fixed seed and greedily assigned to whichever split
is currently furthest below its participant-count target, which keeps the
70/15/15 proportions close despite unequal cluster sizes.

Cohort filter
-------------
The analytic cohort is adults (``RIDAGEYR >= 18``): the five conditions and
their clinical thresholds are defined for adults, and most children lack the
questionnaire fields anyway. Filtering happens *here*, before assignment, so the
70/15/15 proportions are correct for the adult cohort that actually gets
modelled. Note this is safe to do at any stage: because assignment is at whole-
PSU granularity, dropping individuals never moves a cluster across splits nor
splits one apart, so it can't invalidate the cluster property — filtering
upstream just keeps the reported proportions honest and the cohort consistent
with pretraining/fine-tuning.

Outputs (under ``data/splits/``):
    train.csv, val.csv, test.csv   — each has columns SEQN, cycle
                                     (SEQN alone is not unique across cycles).
    split_assignments.csv          — every participant with its split + cluster.

Usage
-----
    python -m src.data.splits            # seed 42
    python -m src.data.splits --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_ROOT = Path("data/processed")
MERGED_PATH = PROCESSED_ROOT / "merged.parquet"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"
SPLITS_ROOT = Path("data/splits")

CLUSTER_KEYS = ["cycle", "SDMVSTRA", "SDMVPSU"]
TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}
LABEL_NAMES = [
    "diabetes", "prediabetes", "obesity", "hypertension", "metabolic_syndrome",
]


def assign_clusters(cluster_sizes: pd.Series, seed: int) -> dict:
    """Greedily assign whole clusters to splits to hit participant targets."""
    total = int(cluster_sizes.sum())
    goals = {s: frac * total for s, frac in TARGETS.items()}
    counts = {s: 0 for s in TARGETS}
    assignment: dict = {}

    rng = np.random.default_rng(seed)
    order = list(cluster_sizes.index)
    rng.shuffle(order)
    # Assign biggest-first so large clusters don't overshoot a nearly-full split.
    order.sort(key=lambda c: cluster_sizes[c], reverse=True)

    for cluster in order:
        size = int(cluster_sizes[cluster])
        # Split with the largest remaining deficit (target minus current count).
        target_split = max(TARGETS, key=lambda s: goals[s] - counts[s])
        assignment[cluster] = target_split
        counts[target_split] += size

    return assignment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-age", type=int, default=18,
                        help="Keep only participants with RIDAGEYR >= this (adults).")
    args = parser.parse_args(argv)

    df = pd.read_parquet(MERGED_PATH, columns=["SEQN", "RIDAGEYR", *CLUSTER_KEYS])
    labels = pd.read_parquet(LABELS_PATH)

    # Restrict to the adult analytic cohort before clustering/assignment.
    n_all = len(df)
    df = df[df["RIDAGEYR"] >= args.min_age].reset_index(drop=True)
    print(f"cohort filter: RIDAGEYR >= {args.min_age} kept {len(df):,} of "
          f"{n_all:,} ({n_all - len(df):,} minors dropped)\n")

    # Build a hashable cluster id per participant.
    df["cluster"] = list(zip(df["cycle"], df["SDMVSTRA"].astype(int),
                             df["SDMVPSU"].astype(int)))
    cluster_sizes = df.groupby("cluster").size()

    assignment = assign_clusters(cluster_sizes, args.seed)
    df["split"] = df["cluster"].map(assignment)

    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Seed {args.seed} | {len(cluster_sizes)} clusters -> "
          f"{df['split'].nunique()} splits\n")
    print(f"{'split':<7}{'clusters':>10}{'participants':>14}{'share':>8}")
    n_total = len(df)
    n_clusters_by_split = {s: sum(1 for c in assignment.values() if c == s)
                           for s in TARGETS}
    for s in TARGETS:
        n = int((df["split"] == s).sum())
        print(f"{s:<7}{n_clusters_by_split[s]:>10}{n:>14,}{n / n_total:>8.1%}")

    # ---- label prevalence per split (among valid participants) ------------ #
    merged = df.merge(labels, on=["SEQN", "cycle"], how="left", validate="one_to_one")
    print("\nLabel prevalence by split (positives / valid participants):")
    header = f"{'label':<20}" + "".join(f"{s:>12}" for s in TARGETS)
    print(header)
    for name in LABEL_NAMES:
        row = f"{name:<20}"
        for s in TARGETS:
            sub = merged[merged["split"] == s]
            valid = sub[f"{name}_valid"] == 1
            n_valid = int(valid.sum())
            prev = sub.loc[valid, name].mean() if n_valid else float("nan")
            row += f"{prev:>11.1%} "
        print(row)

    # ---- write files ------------------------------------------------------ #
    for s in TARGETS:
        sub = df.loc[df["split"] == s, ["SEQN", "cycle"]].sort_values(["cycle", "SEQN"])
        sub.to_csv(SPLITS_ROOT / f"{s}.csv", index=False)
    df[["SEQN", "cycle", "SDMVSTRA", "SDMVPSU", "split"]].to_csv(
        SPLITS_ROOT / "split_assignments.csv", index=False)

    print(f"\n  saved -> {SPLITS_ROOT}/(train|val|test).csv + split_assignments.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
