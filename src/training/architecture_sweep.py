"""Architecture comparison: grouped-MLP backbone vs. FT-Transformer backbone.

Both trained with everything else held identical to the established best
config — per-head leakage masking, survey-weighting off — across the same 10
fixed seeds (0-9) used by every other seed sweep in this project, so a gap can
be checked against run-to-run noise the same way the pretrain/weight/SMOTE
effects were.

One deliberate asymmetry, stated plainly rather than hidden: the grouped-MLP
backbone *can* warm-start from `experiments/pretrain_perhead/backbone.pt`
(self-supervised pretraining), but the FT-Transformer has no pretrained
checkpoint of its own — pretraining was never run against this architecture.
Both variants therefore run **without** pretraining here, so the comparison
isolates architecture (grouped-sparse-MLP vs. full self-attention over
per-feature tokens), not "pretrained vs. not."

Usage
-----
    python -m src.training.architecture_sweep
    python -m src.training.architecture_sweep --n-seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.labels import LABEL_NAMES
from src.models.backbone import Backbone
from src.models.ft_transformer import FTTransformerBackbone
from src.training.finetune import build_head_masks, pick_device, train_and_evaluate

PROCESSED_ROOT = Path("data/processed")
FEATURES_PATH = PROCESSED_ROOT / "features.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_perhead.json"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"
PRETRAIN_WEIGHTS = Path("experiments/pretrain_perhead/backbone.pt")
OUT_MD = Path("results/architecture_sweep.md")

ARCHITECTURES = {
    "grouped-mlp": Backbone,
    "ft-transformer": FTTransformerBackbone,
}


def run_sweep(n_seeds: int, epochs: int, batch_size: int, lr: float, patience: int) -> dict:
    device = pick_device()
    schema = json.loads(SCHEMA_PATH.read_text())
    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    head_masks = build_head_masks(schema, device)

    seeds = list(range(n_seeds))
    n_runs = n_seeds * len(ARCHITECTURES)
    print(f"Architecture sweep: {n_seeds} seeds x {len(ARCHITECTURES)} architectures "
          f"= {n_runs} runs (per-head, no pretrain, weight-off, device={device})")

    results: dict[str, dict[int, dict]] = {}
    t0 = time.time()
    run_i = 0
    for arch_name, backbone_cls in ARCHITECTURES.items():
        results[arch_name] = {}
        for seed in seeds:
            run_i += 1
            out = train_and_evaluate(
                features=features, labels=labels, schema=schema, head_masks=head_masks,
                pretrain_weights=PRETRAIN_WEIGHTS, use_pretrain=False, use_weights=False,
                seed=seed, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
                device=device, run_dir=None, save_checkpoint=False, verbose=False,
                backbone_cls=backbone_cls,
            )
            results[arch_name][seed] = out["final_val_auroc"]
            elapsed = time.time() - t0
            aurocs = {k: round(v, 3) for k, v in out["final_val_auroc"].items()}
            print(f"  [{run_i:>2}/{n_runs}] {arch_name:<15} seed {seed}: {aurocs} "
                  f"(best_epoch {out['best_epoch']}, {elapsed:.0f}s elapsed)")

    return results


def write_report(results: dict, n_seeds: int) -> None:
    archs = list(results.keys())
    stats = {}
    for arch, per_seed in results.items():
        stats[arch] = {}
        for name in LABEL_NAMES:
            vals = np.array([per_seed[s][name] for s in per_seed])
            stats[arch][name] = (float(vals.mean()), float(vals.std(ddof=1)))

    lines = [f"# Architecture comparison — grouped-MLP vs. FT-Transformer "
             f"({n_seeds} seeds x 2 architectures = {n_seeds * 2} runs)\n",
             "Validation-split AUROC, mean +/- std across seeds "
             f"0..{n_seeds - 1}. Both per-head leakage masking, weight-off, "
             "**no pretraining for either** (FT-Transformer has no pretrained "
             "checkpoint, so grouped-MLP runs without one too, for a fair "
             "architecture-only comparison).\n",
             "| architecture | diabetes | prediabetes | obesity | hypertension | metab. syndrome | mean |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arch in archs:
        row_means = [stats[arch][n][0] for n in LABEL_NAMES]
        cells = " | ".join(f"{stats[arch][n][0]:.3f} ± {stats[arch][n][1]:.3f}" for n in LABEL_NAMES)
        lines.append(f"| {arch} | {cells} | {np.mean(row_means):.3f} |")

    lines += ["\n## Does either architecture win, or is the gap noise?\n",
             "Same rule as every other effect in this project: a gap only "
             "counts if it's bigger than the noise floor (the larger of the "
             "two architectures' per-condition standard deviation across "
             "seeds).\n",
             "| condition | ft-transformer − grouped-mlp | noise floor | gap real? | winner |",
             "|---|---:|---:|:---:|:---:|"]
    for name in LABEL_NAMES:
        mlp_mean, mlp_std = stats["grouped-mlp"][name]
        ft_mean, ft_std = stats["ft-transformer"][name]
        gap = ft_mean - mlp_mean
        noise = max(mlp_std, ft_std)
        real = abs(gap) > noise
        winner = ("ft-transformer" if gap > 0 else "grouped-mlp") if real else "tie (noise)"
        lines.append(f"| {name} | {gap:+.3f} | {noise:.3f} "
                     f"| {'**yes**' if real else 'no'} | {winner} |")

    out_path = OUT_MD
    out_path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nsaved -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args(argv)

    results = run_sweep(args.n_seeds, args.epochs, args.batch_size, args.lr, args.patience)
    Path("experiments/architecture_sweep").mkdir(parents=True, exist_ok=True)
    Path("experiments/architecture_sweep/results.json").write_text(json.dumps(results, indent=2))
    write_report(results, args.n_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
