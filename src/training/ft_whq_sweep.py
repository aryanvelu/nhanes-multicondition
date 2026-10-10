"""FT-Transformer, before vs. after adding the WHQ (weight-history) features.

Re-runs the FT-Transformer half of `architecture_sweep.py` (no pretraining —
settled negative in `results/ft_pretrain_sweep.md` — per-head leakage
masking, survey-weight off) across the same 10 fixed seeds, on the current
`feature_schema_perhead.json`, which now includes the `weight_history` group
(`WHD020`/`WHD050`/`WHD120`) with `WHD020` masked for the obesity head only
(see `perhead_leakage.py`).

The "before" numbers are read directly out of the already-written
`results/architecture_sweep.md` (the FT-Transformer row from the pre-WHQ
schema) rather than retyped, so there's no risk of transcription drift.

Usage
-----
    python -m src.training.ft_whq_sweep
    python -m src.training.ft_whq_sweep --n-seeds 5
"""

from __future__ import annotations

import argparse
import json
import re
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
PREV_RESULTS_PATH = Path("results/architecture_sweep.md")
OUT_MD = Path("results/ft_whq_sweep.md")


def parse_previous_ft_row(path: Path) -> dict[str, tuple[float, float]]:
    """Pull the FT-Transformer mean +/- std row out of the pre-WHQ sweep report."""
    text = path.read_text()
    m = re.search(
        r"^\| ft-transformer \| (.+) \|$",
        text, flags=re.MULTILINE,
    )
    if not m:
        raise ValueError(f"couldn't find ft-transformer row in {path}")
    cells = [c.strip() for c in m.group(1).split("|")]
    # cells: diabetes, prediabetes, obesity, hypertension, metab.syndrome, mean
    out = {}
    for name, cell in zip(LABEL_NAMES, cells[:5]):
        mean_s, std_s = cell.split("±")
        out[name] = (float(mean_s.strip()), float(std_s.strip()))
    return out


def run_sweep(n_seeds: int, epochs: int, batch_size: int, lr: float, patience: int) -> dict:
    device = pick_device()
    schema = json.loads(SCHEMA_PATH.read_text())
    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    head_masks = build_head_masks(schema, device)

    print(f"FT-Transformer + WHQ sweep: {n_seeds} seeds (per-head, no pretrain, "
          f"weight-off, device={device})")
    print(f"schema: {len(schema['continuous'])} continuous + "
          f"{len(schema['categorical'])} categorical, obesity mask: "
          f"{schema['head_masks']['obesity']}")

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

    old_stats = parse_previous_ft_row(PREV_RESULTS_PATH)
    old_mean = float(np.mean([old_stats[n][0] for n in LABEL_NAMES]))

    lines = [f"# FT-Transformer: before vs. after WHQ (weight-history) features "
             f"({n_seeds} seeds)\n",
             "Same FT-Transformer, no pretraining (settled negative — see "
             "`results/ft_pretrain_sweep.md`), per-head leakage masking, "
             "weight-off, same 10 fixed seeds. Only the feature set changed: "
             "`weight_history` (`WHD020`/`WHD050`/`WHD120`) added, with "
             "`WHD020` **and** `WHD050` masked from the obesity head "
             "(both strong self-report proxies for current body size — an "
             "earlier run with only `WHD020` masked produced a suspicious "
             "isolated +0.160 obesity jump traced to `WHD050`'s 0.556 "
             "correlation with the label; see `report.md`). `WHD120` (weight "
             "at age 25, correlation 0.275) stays visible everywhere "
             "including obesity, as genuine history rather than a "
             "near-restatement of the label.\n",
             "| condition | before (no WHQ) | after (+ WHQ) | delta | noise floor | real? |",
             "|---|---:|---:|---:|---:|:---:|"]
    for name in LABEL_NAMES:
        old_m, old_s = old_stats[name]
        new_m, new_s = new_stats[name]
        delta = new_m - old_m
        noise = max(old_s, new_s)
        real = abs(delta) > noise
        marker = " **<- obesity**" if name == "obesity" else ""
        lines.append(f"| {name}{marker} | {old_m:.3f} ± {old_s:.3f} "
                     f"| {new_m:.3f} ± {new_s:.3f} | {delta:+.3f} | {noise:.3f} "
                     f"| {'**yes**' if real else 'no'} |")
    lines.append(f"| **mean** | **{old_mean:.3f}** | **{new_mean:.3f}** "
                f"| **{new_mean - old_mean:+.3f}** | | |")

    obesity_delta = new_stats["obesity"][0] - old_stats["obesity"][0]
    obesity_noise = max(old_stats["obesity"][1], new_stats["obesity"][1])
    obesity_real = abs(obesity_delta) > obesity_noise
    lines += ["\n## Obesity, specifically\n",
             f"Obesity AUROC moved {obesity_delta:+.3f} ({old_stats['obesity'][0]:.3f} -> "
             f"{new_stats['obesity'][0]:.3f}), against a noise floor of "
             f"{obesity_noise:.3f} — {'a real change' if obesity_real else 'not distinguishable from seed noise'}. "
             f"`WHD020` and `WHD050` are both masked for this head specifically "
             f"(same as `BMXBMI`/`BMXWAIST`/`BMXWT`/`BMXHT`), so this reflects "
             f"`WHD120` (visible, correlation 0.275 with the label) plus every "
             f"other unmasked group — not the two proxy features obesity never sees."]

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
    Path("experiments/ft_whq_sweep").mkdir(parents=True, exist_ok=True)
    Path("experiments/ft_whq_sweep/results.json").write_text(json.dumps(results, indent=2))
    write_report(results, args.n_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
