"""FT-Transformer, before vs. after adding kidney/liver labs, alcohol, sleep,
and medication-use features.

Same protocol as `ft_whq_sweep.py`: FT-Transformer, no pretraining (settled
negative, `results/ft_pretrain_sweep.md`), per-head leakage masking,
weight-off, same 10 fixed seeds. New groups since the last checkpoint
(`results/ft_whq_sweep.md`): `renal` (creatinine, BUN), `hepatic` (ALT, AST,
ALP, GGT, bilirubin, albumin), `lifestyle` gains alcohol (`ALQ130`) and sleep
(`sleep_hours`, `SLQ050`), and a new `medication_use` group (`RXDCOUNT`,
total prescription medications/person, aggregated from the one-row-per-drug
`RXQ_RX` file in `merge.py`). None of these are read by `labels.py`
(confirmed via `label_defining_columns()`), and per the lesson from
`WHD050` — an automatic leakage-check pass is not the same as "safe" — this
script checks every head for an isolated, disproportionate AUROC jump versus
the pre-this-batch baseline, the same tell that caught `WHD050`.

The "before" numbers are the already-published mean +/- std from
`results/ft_whq_sweep.md` (the FT-Transformer + WHQ, corrected-mask result),
cited directly below rather than re-parsed, since that file's table layout
(before/after columns) differs from `architecture_sweep.md`'s.

Usage
-----
    python -m src.training.ft_organlab_sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.labels import LABEL_NAMES
from src.models.ft_transformer import FTTransformerBackbone
from src.training.finetune import build_head_masks, pick_device, train_and_evaluate

PROCESSED_ROOT = Path("data/processed")
FEATURES_PATH = PROCESSED_ROOT / "features.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_perhead.json"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"
PRETRAIN_WEIGHTS = Path("experiments/pretrain_ft_transformer/backbone.pt")  # unused (no pretrain)
OUT_MD = Path("results/ft_organlab_sweep.md")

# Cited from results/ft_whq_sweep.md ("after (+ WHQ)" column, corrected
# WHD020+WHD050 obesity mask) — the last checkpoint before this batch.
PREV_RESULTS: dict[str, tuple[float, float]] = {
    "diabetes": (0.854, 0.002),
    "prediabetes": (0.754, 0.002),
    "obesity": (0.834, 0.002),
    "hypertension": (0.830, 0.002),
    "metabolic_syndrome": (0.870, 0.002),
}


def run_sweep(n_seeds: int, epochs: int, batch_size: int, lr: float, patience: int) -> dict:
    device = pick_device()
    schema = json.loads(SCHEMA_PATH.read_text())
    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    head_masks = build_head_masks(schema, device)

    print(f"FT-Transformer + organ labs/alcohol/sleep/meds sweep: {n_seeds} seeds "
          f"(per-head, no pretrain, weight-off, device={device})")
    print(f"schema: {len(schema['continuous'])} continuous + "
          f"{len(schema['categorical'])} categorical across "
          f"{len(set(v['group'] for v in schema['continuous'].values()) | set(v['group'] for v in schema['categorical'].values()))} groups")

    results: dict[int, dict] = {}
    t0 = time.time()
    for seed in range(n_seeds):
        out = train_and_evaluate(
            features=features, labels=labels, schema=schema, head_masks=head_masks,
            pretrain_weights=PRETRAIN_WEIGHTS, use_pretrain=False, use_weights=False,
            seed=seed, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
            device=device, run_dir=None, save_checkpoint=False, verbose=False,
            backbone_cls=FTTransformerBackbone,
        )
        results[seed] = out["final_val_auroc"]
        elapsed = time.time() - t0
        aurocs = {k: round(v, 3) for k, v in out["final_val_auroc"].items()}
        print(f"  [{seed + 1:>2}/{n_seeds}] seed {seed}: {aurocs} "
              f"(best_epoch {out['best_epoch']}, {elapsed:.0f}s elapsed)")

    return results


def write_report(results: dict, n_seeds: int) -> None:
    new_stats = {}
    for name in LABEL_NAMES:
        vals = np.array([results[s][name] for s in results])
        new_stats[name] = (float(vals.mean()), float(vals.std(ddof=1)))
    new_mean = float(np.mean([new_stats[n][0] for n in LABEL_NAMES]))
    old_mean = float(np.mean([PREV_RESULTS[n][0] for n in LABEL_NAMES]))

    lines = [f"# FT-Transformer: before vs. after organ labs / alcohol / sleep / "
             f"medications ({n_seeds} seeds)\n",
             "Same FT-Transformer, no pretraining, per-head leakage masking, "
             "weight-off, same 10 fixed seeds. New groups this round: `renal` "
             "(creatinine, BUN), `hepatic` (ALT/AST/ALP/GGT/bilirubin/albumin), "
             "`lifestyle` +alcohol (`ALQ130`) +sleep (`sleep_hours`/`SLQ050`), "
             "and `medication_use` (`RXDCOUNT`, aggregated from the one-row-"
             "per-drug `RXQ_RX` file). None read by `labels.py` — checked below "
             "for the same isolated-jump signature that caught `WHD050`.\n",
             "| condition | before (+ WHQ only) | after (+ organ/lifestyle/meds) | delta | noise floor | real? |",
             "|---|---:|---:|---:|---:|:---:|"]
    flagged = []
    for name in LABEL_NAMES:
        old_m, old_s = PREV_RESULTS[name]
        new_m, new_s = new_stats[name]
        delta = new_m - old_m
        noise = max(old_s, new_s)
        real = abs(delta) > noise
        if real and abs(delta) > 0.02:  # "isolated disproportionate jump" bar
            flagged.append((name, delta))
        lines.append(f"| {name} | {old_m:.3f} ± {old_s:.3f} "
                     f"| {new_m:.3f} ± {new_s:.3f} | {delta:+.3f} | {noise:.3f} "
                     f"| {'**yes**' if real else 'no'} |")
    lines.append(f"| **mean** | **{old_mean:.3f}** | **{new_mean:.3f}** "
                f"| **{new_mean - old_mean:+.3f}** | | |")

    lines += ["\n## Proxy-leak check (the WHD050 tell: isolated, disproportionate jump)\n"]
    if flagged:
        lines.append("**Flagged for follow-up** — real change bigger than +/-0.02 "
                     "on a single condition, worth investigating like WHD050 was:\n")
        for name, delta in flagged:
            lines.append(f"- `{name}`: {delta:+.3f}")
    else:
        lines.append("No condition shows an isolated jump bigger than +/-0.02 "
                     "beyond its noise floor — gains (if any) look spread across "
                     "conditions rather than concentrated on one, unlike the "
                     "WHD050 case. No further per-head masking indicated by this "
                     "check alone.")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nsaved -> {OUT_MD}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args(argv)

    results = run_sweep(args.n_seeds, args.epochs, args.batch_size, args.lr, args.patience)
    Path("experiments/ft_organlab_sweep").mkdir(parents=True, exist_ok=True)
    Path("experiments/ft_organlab_sweep/results.json").write_text(json.dumps(results, indent=2))
    write_report(results, args.n_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
