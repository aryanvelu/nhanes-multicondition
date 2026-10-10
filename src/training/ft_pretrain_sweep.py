"""FT-Transformer: pretrained vs. non-pretrained, across the same 10 seeds,
with both validation (noise-floor check) and test (held-out confirmation)
results — train/val/test, all reported.

Mirrors the methodology of every other seed sweep in this project
(`results/seed_sweep.md`, `results/smote_sweep.md`, `results/architecture_sweep.md`):
same 10 fixed seeds (0-9), same per-head leakage masking, survey-weight off
(the established best-AUROC training regime). The only thing that changes
between the two variants is whether `train_and_evaluate` warm-starts the
FT-Transformer backbone from `experiments/pretrain_ft_transformer/backbone.pt`
(produced by `pretrain_ft.py`'s masked-feature-*token* reconstruction) or
trains from a random initialization.

Reports two tables:
* **Validation AUROC**, mean +/- std across the 10 seeds — checked against the
  noise floor, same rule as every other effect in this project.
* **Test AUROC/AUPRC** (survey-weighted), mean +/- std across the *same* 10
  seeds' trained models — the held-out confirmation, not just one cherry-picked
  seed.

Usage
-----
    python -m src.training.ft_pretrain_sweep
    python -m src.training.ft_pretrain_sweep --n-seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.labels import LABEL_NAMES
from src.evaluation.metrics import COND_WEIGHT, weighted_auprc, weighted_auroc
from src.evaluation.stacking import load_split_arrays
from src.models.ft_transformer import FTTransformerBackbone
from src.training.finetune import (
    MultiHead, build_head_masks, compute_logits, pick_device, train_and_evaluate,
)

PROCESSED_ROOT = Path("data/processed")
FEATURES_PATH = PROCESSED_ROOT / "features.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_perhead.json"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"
PRETRAIN_WEIGHTS = Path("experiments/pretrain_ft_transformer/backbone.pt")
OUT_ROOT = Path("experiments/ft_pretrain_sweep")
OUT_MD = Path("results/ft_pretrain_sweep.md")

VARIANTS = [("pretrain-on", True), ("pretrain-off", False)]


def predict_ft(model_path: Path, schema: dict, device, data: dict, head_masks) -> np.ndarray:
    backbone = FTTransformerBackbone(schema).to(device)
    heads = MultiHead(backbone.output_dim).to(device)
    ckpt = torch.load(model_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone"]); heads.load_state_dict(ckpt["heads"])
    backbone.eval(); heads.eval()
    xc, xm, xk = [torch.from_numpy(data[k]).to(device) for k in ("xc", "xm", "xk")]
    with torch.no_grad():
        logits = compute_logits(backbone, heads, xc, xm, xk, head_masks)
        return torch.sigmoid(logits).cpu().numpy()


def evaluate_test(probs: np.ndarray, test_data: dict) -> tuple[dict, dict]:
    out_auroc, out_auprc = {}, {}
    for k, name in enumerate(LABEL_NAMES):
        w = test_data["weights"][COND_WEIGHT[name]]
        mw = (test_data["valid"][:, k] == 1) & (w > 0)
        y, s, wt = test_data["y"][mw, k], probs[mw, k], w[mw]
        out_auroc[name] = weighted_auroc(y, s, wt)
        out_auprc[name] = weighted_auprc(y, s, wt)
    return out_auroc, out_auprc


def run_sweep(n_seeds: int, epochs: int, batch_size: int, lr: float, patience: int) -> tuple[dict, dict]:
    device = pick_device()
    schema = json.loads(SCHEMA_PATH.read_text())
    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    head_masks = build_head_masks(schema, device)
    test_data = load_split_arrays("test", schema)

    val_results: dict[str, dict[int, dict]] = {v: {} for v, _ in VARIANTS}
    test_results: dict[str, dict[int, dict]] = {v: {} for v, _ in VARIANTS}

    n_runs = n_seeds * len(VARIANTS)
    print(f"FT-Transformer pretrain sweep: {n_seeds} seeds x {len(VARIANTS)} variants "
          f"= {n_runs} runs (per-head, weight-off, device={device})")
    t0 = time.time()
    run_i = 0
    for vname, use_pretrain in VARIANTS:
        for seed in range(n_seeds):
            run_i += 1
            run_dir = OUT_ROOT / vname / f"seed_{seed}"
            out = train_and_evaluate(
                features=features, labels=labels, schema=schema, head_masks=head_masks,
                pretrain_weights=PRETRAIN_WEIGHTS, use_pretrain=use_pretrain, use_weights=False,
                seed=seed, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
                device=device, run_dir=run_dir, save_checkpoint=True, verbose=False,
                backbone_cls=FTTransformerBackbone,
            )
            val_results[vname][seed] = out["final_val_auroc"]

            probs = predict_ft(run_dir / "model.pt", schema, device, test_data, head_masks)
            t_auroc, t_auprc = evaluate_test(probs, test_data)
            test_results[vname][seed] = {"auroc": t_auroc, "auprc": t_auprc}

            elapsed = time.time() - t0
            val_r = {k: round(v, 3) for k, v in out["final_val_auroc"].items()}
            test_r = {k: round(v, 3) for k, v in t_auroc.items()}
            print(f"  [{run_i:>2}/{n_runs}] {vname:<13} seed {seed}: val={val_r} "
                  f"test={test_r} (best_epoch {out['best_epoch']}, {elapsed:.0f}s elapsed)")

    return val_results, test_results


def _mean_std(results: dict, seeds: range, key=None) -> dict:
    stats = {}
    for vname, per_seed in results.items():
        stats[vname] = {}
        for name in LABEL_NAMES:
            vals = np.array([
                (per_seed[s][key][name] if key else per_seed[s][name]) for s in seeds
            ])
            stats[vname][name] = (float(vals.mean()), float(vals.std(ddof=1)))
    return stats


def write_report(val_results: dict, test_results: dict, n_seeds: int) -> None:
    seeds = range(n_seeds)
    val_stats = _mean_std(val_results, seeds)
    test_auroc_stats = _mean_std(test_results, seeds, key="auroc")
    test_auprc_stats = _mean_std(test_results, seeds, key="auprc")

    lines = [f"# FT-Transformer: pretrained vs. non-pretrained "
             f"({n_seeds} seeds x 2 variants = {n_seeds * 2} runs)\n",
             "Masked-feature-**token** pretraining (`pretrain_ft.py`) vs. random "
             "init, same per-head leakage masking / weight-off training regime, "
             "same 10 fixed seeds used by every other sweep in this project. "
             "Both val (noise-floor check) and test (held-out, survey-weighted) "
             "results below — train/val/test.\n",
             "## Validation AUROC (unweighted, mean +/- std across seeds)\n",
             "| variant | diabetes | prediabetes | obesity | hypertension | metab. syndrome | mean |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for vname, _ in VARIANTS:
        row_means = [val_stats[vname][n][0] for n in LABEL_NAMES]
        cells = " | ".join(f"{val_stats[vname][n][0]:.3f} ± {val_stats[vname][n][1]:.3f}"
                          for n in LABEL_NAMES)
        lines.append(f"| {vname} | {cells} | {np.mean(row_means):.3f} |")

    lines += ["\n### Does the val AUROC gap survive multiple seeds?\n",
             "| condition | pretrain-on − pretrain-off | noise floor | gap real? |",
             "|---|---:|---:|:---:|"]
    for name in LABEL_NAMES:
        on_mean, on_std = val_stats["pretrain-on"][name]
        off_mean, off_std = val_stats["pretrain-off"][name]
        gap = on_mean - off_mean
        noise = max(on_std, off_std)
        lines.append(f"| {name} | {gap:+.3f} | {noise:.3f} "
                     f"| {'**yes**' if abs(gap) > noise else 'no'} |")

    lines += ["\n## Test AUROC (survey-weighted, mean +/- std across the same seeds)\n",
             "| variant | diabetes | prediabetes | obesity | hypertension | metab. syndrome | mean |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for vname, _ in VARIANTS:
        row_means = [test_auroc_stats[vname][n][0] for n in LABEL_NAMES]
        cells = " | ".join(f"{test_auroc_stats[vname][n][0]:.3f} ± {test_auroc_stats[vname][n][1]:.3f}"
                          for n in LABEL_NAMES)
        lines.append(f"| {vname} | {cells} | {np.mean(row_means):.3f} |")

    lines += ["\n### Does the test AUROC gap survive multiple seeds?\n",
             "| condition | pretrain-on − pretrain-off | noise floor | gap real? |",
             "|---|---:|---:|:---:|"]
    for name in LABEL_NAMES:
        on_mean, on_std = test_auroc_stats["pretrain-on"][name]
        off_mean, off_std = test_auroc_stats["pretrain-off"][name]
        gap = on_mean - off_mean
        noise = max(on_std, off_std)
        lines.append(f"| {name} | {gap:+.3f} | {noise:.3f} "
                     f"| {'**yes**' if abs(gap) > noise else 'no'} |")

    lines += ["\n## Test AUPRC (survey-weighted, mean +/- std across the same seeds)\n",
             "| variant | diabetes | prediabetes | obesity | hypertension | metab. syndrome |",
             "|---|---:|---:|---:|---:|---:|"]
    for vname, _ in VARIANTS:
        cells = " | ".join(f"{test_auprc_stats[vname][n][0]:.3f} ± {test_auprc_stats[vname][n][1]:.3f}"
                          for n in LABEL_NAMES)
        lines.append(f"| {vname} | {cells} |")

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

    val_results, test_results = run_sweep(
        args.n_seeds, args.epochs, args.batch_size, args.lr, args.patience)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "val_results.json").write_text(json.dumps(val_results, indent=2))
    (OUT_ROOT / "test_results.json").write_text(json.dumps(test_results, indent=2))
    write_report(val_results, test_results, args.n_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
