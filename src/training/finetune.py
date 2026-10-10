"""Supervised fine-tuning: five condition heads on the shared backbone.

This is the stage that actually predicts disease. It takes the ``Backbone``
(optionally warm-started from self-supervised pretraining) and attaches **five
classification heads**, one per condition (diabetes, prediabetes, obesity,
hypertension, metabolic syndrome). All five share the same backbone, so the
representation learned for one condition helps the others — the point of a
multi-label model.

Two NHANES-specific details are handled in the loss:

* **Per-label valid-masking.** Not every label is computable for every person
  (metabolic syndrome needs the full fasting panel, so it exists for ~40% of
  adults). Each ``<label>_valid`` flag zeroes out that label's loss where it
  can't be computed, so a person still trains every *other* head. No rows are
  dropped for having partial labels.

* **Survey-weighted loss.** NHANES over-samples some groups, so an unweighted
  model would drift toward those groups. Each participant's contribution to the
  loss is scaled by their exam survey weight (``WTMEC2YR``), normalized to mean 1
  within the training set, so the fitted model reflects the US adult population.

The loss is therefore a survey-weighted, validity-masked average binary
cross-entropy over the (person, label) pairs that are actually observed.

The pretraining and survey-weighting can each be switched off (``--no-pretrain``,
``--no-survey-weight``) to support the README's four-way ablation.

Usage
-----
    python -m src.training.finetune
    python -m src.training.finetune --no-pretrain --no-survey-weight
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.backbone import Backbone
from src.models.structured_layer import assemble_model_inputs

PROCESSED_ROOT = Path("data/processed")
FEATURES_PATH = PROCESSED_ROOT / "features_no_leakage.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_no_leakage.json"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"
SPLITS_ROOT = Path("data/splits")
PRETRAIN_WEIGHTS = Path("experiments/pretrain/backbone.pt")
OUT_DIR = Path("experiments/finetune")

LABEL_NAMES = [
    "diabetes", "prediabetes", "obesity", "hypertension", "metabolic_syndrome",
]
WEIGHT_COL = "WTMEC2YR"  # exam weight; our features come from the MEC exam/labs


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MultiHead(nn.Module):
    """Five independent binary heads sharing the backbone embedding."""

    def __init__(self, embedding_dim: int, n_labels: int = 5, dropout: float = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(embedding_dim // 2, 1),
            )
            for _ in range(n_labels)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(z) for head in self.heads], dim=1)  # [B, 5] logits


def load_split(name: str, features: pd.DataFrame, labels: pd.DataFrame,
               schema: dict, use_weights: bool):
    """Return tensors (x_cont, x_miss, x_cat, y, valid, weight) for a split."""
    ids = pd.read_csv(SPLITS_ROOT / f"{name}.csv")
    feat = ids.merge(features, on=["SEQN", "cycle"], how="left", validate="one_to_one")
    lab = ids.merge(labels, on=["SEQN", "cycle"], how="left", validate="one_to_one")

    x_cont, x_miss, x_cat, _, _ = assemble_model_inputs(feat, schema)
    y = lab[LABEL_NAMES].to_numpy(dtype=np.float32)
    valid = lab[[f"{n}_valid" for n in LABEL_NAMES]].to_numpy(dtype=np.float32)

    if use_weights:
        w = feat[WEIGHT_COL].to_numpy(dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0)
        w = w / w.mean()  # normalize to mean 1 so loss scale is stable
    else:
        w = np.ones(len(feat), dtype=np.float64)

    return (torch.from_numpy(x_cont), torch.from_numpy(x_miss), torch.from_numpy(x_cat),
            torch.from_numpy(y), torch.from_numpy(valid),
            torch.from_numpy(w.astype(np.float32)))


def masked_weighted_bce(logits, y, valid, w):
    """Survey-weighted, validity-masked mean binary cross-entropy."""
    per = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    wmask = valid * w.unsqueeze(1)          # [B, 5] weight, zero where invalid
    denom = wmask.sum().clamp(min=1e-8)
    return (per * wmask).sum() / denom


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties averaged — enough for a numpy AUROC."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    a_sorted = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under ROC via the Mann-Whitney statistic; NaN if one class absent."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(score)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compute_logits(backbone, heads, x_cont, x_miss, x_cat, head_masks=None):
    """Head logits. If head_masks is given, run the backbone once PER head with
    that head's own label-defining features masked (value 0, missing-flag 1),
    so no head can see its own answer while every other group stays visible."""
    if head_masks is None:
        z = backbone(x_cont, x_miss, x_cat)
        return torch.cat([h(z) for h in heads.heads], dim=1)
    logits = []
    for k, h in enumerate(heads.heads):
        m = head_masks[k]  # [C] bool over continuous features
        xc = x_cont.masked_fill(m, 0.0)
        xm = torch.maximum(x_miss, m.to(x_miss.dtype))
        logits.append(h(backbone(xc, xm, x_cat)))
    return torch.cat(logits, dim=1)


def build_head_masks(schema, device):
    """One boolean [n_continuous] mask per label from schema['head_masks']."""
    cont = list(schema["continuous"])
    idx = {c: i for i, c in enumerate(cont)}
    masks = []
    for name in LABEL_NAMES:
        feats = schema.get("head_masks", {}).get(name, [])
        m = torch.zeros(len(cont), dtype=torch.bool)
        for f in feats:
            m[idx[f]] = True
        masks.append(m.to(device))
    return masks


def evaluate(model, backbone, tensors, device, head_masks=None):
    """Return (val_loss, {label: auroc}) on a split, over valid samples only."""
    xc, xm, xk, y, valid, w = [t.to(device) for t in tensors]
    backbone.eval(); model.eval()
    with torch.no_grad():
        logits = compute_logits(backbone, model, xc, xm, xk, head_masks)
        loss = masked_weighted_bce(logits, y, valid, w).item()
        probs = torch.sigmoid(logits).cpu().numpy()
    y_np, valid_np = y.cpu().numpy(), valid.cpu().numpy()
    aurocs = {}
    for k, name in enumerate(LABEL_NAMES):
        m = valid_np[:, k] == 1
        aurocs[name] = auroc(y_np[m, k], probs[m, k])
    return loss, aurocs


def train_and_evaluate(*, features: pd.DataFrame, labels: pd.DataFrame, schema: dict,
                       head_masks, pretrain_weights: Path, use_pretrain: bool,
                       use_weights: bool, seed: int, epochs: int, batch_size: int,
                       lr: float, patience: int, device, run_dir: Path | None,
                       save_checkpoint: bool = True, verbose: bool = True,
                       backbone_kwargs: dict | None = None,
                       head_dropout: float = 0.2,
                       use_smote: bool = False,
                       smote_labels: tuple[str, ...] = ("prediabetes", "metabolic_syndrome"),
                       backbone_cls: type = Backbone) -> dict:
    """Train one finetune run to convergence; return best_epoch/best_val_loss/
    final_val_auroc. Keeps the best checkpoint in memory (not on disk) until the
    end, so this is cheap to call many times in a row (e.g. a seed sweep or a
    hyperparameter search).

    ``backbone_kwargs`` (e.g. ``hidden_dims``, ``dropout``, ``embedding_dim``)
    override the Backbone's architecture defaults. If they change the
    architecture's tensor shapes and ``use_pretrain`` is requested anyway, the
    pretrained checkpoint (fit at the *default* architecture) simply won't
    load — callers doing an architecture sweep must fall back to
    ``use_pretrain=False`` themselves for incompatible shapes.

    ``use_smote`` SMOTE-balances the training split (never val/test) for
    ``smote_labels`` specifically — see ``src.training.smote`` for why a
    joint multi-head model needs each synthetic row's *other* labels marked
    invalid rather than guessed.

    ``backbone_cls`` swaps the whole architecture (e.g.
    ``src.models.ft_transformer.FTTransformerBackbone``) while keeping every
    other piece of the training loop — loss, masking, splits, evaluation —
    identical, so an architecture comparison isn't confounded by anything else
    changing at the same time.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if save_checkpoint:
        run_dir.mkdir(parents=True, exist_ok=True)

    train_t = load_split("train", features, labels, schema, use_weights)
    val_t = load_split("val", features, labels, schema, use_weights)

    if use_smote:
        from src.training.smote import augment_with_smote
        for name in smote_labels:
            idx = LABEL_NAMES.index(name)
            n_before = train_t[0].shape[0]
            train_t = augment_with_smote(train_t, idx, seed=seed)
            if verbose:
                print(f"  SMOTE[{name}]: {n_before:,} -> {train_t[0].shape[0]:,} rows "
                      f"(+{train_t[0].shape[0] - n_before:,} synthetic)")

    backbone = backbone_cls(schema, **(backbone_kwargs or {})).to(device)
    if use_pretrain:
        if not pretrain_weights.exists():
            raise FileNotFoundError(f"{pretrain_weights} missing; run pretrain first")
        backbone.load_state_dict(torch.load(pretrain_weights, map_location=device))
    model = MultiHead(backbone.output_dim, dropout=head_dropout).to(device)

    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(model.parameters()), lr=lr)
    train_dev = [t.to(device) for t in train_t]
    loader = DataLoader(TensorDataset(*train_dev), batch_size=batch_size,
                        shuffle=True, drop_last=True)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        backbone.train(); model.train()
        tot, nb = 0.0, 0
        for xc, xm, xk, y, valid, w in loader:
            loss = masked_weighted_bce(compute_logits(backbone, model, xc, xm, xk, head_masks), y, valid, w)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item(); nb += 1
        train_loss = tot / max(nb, 1)

        val_loss, aurocs = evaluate(model, backbone, val_t, device, head_masks)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_auroc": aurocs})

        marker = ""
        if val_loss < best_val - 1e-4:
            best_val, best_epoch = val_loss, epoch
            best_state = {"backbone": {k: v.detach().clone() for k, v in backbone.state_dict().items()},
                         "heads": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            marker = " *"
        if verbose:
            auroc_str = "".join(
                f"{aurocs[n]:>7.3f}" if aurocs[n] == aurocs[n] else f"{'na':>7}"
                for n in LABEL_NAMES)
            print(f"{epoch:>5}{train_loss:>9.4f}{val_loss:>9.4f}   {auroc_str}{marker}")
        if epoch - best_epoch >= patience:
            if verbose:
                print(f"early stop: no val improvement for {patience} epochs")
            break

    backbone.load_state_dict(best_state["backbone"]); model.load_state_dict(best_state["heads"])
    _, final_aurocs = evaluate(model, backbone, val_t, device, head_masks)

    if save_checkpoint:
        torch.save(best_state, run_dir / "model.pt")
        (run_dir / "finetune_history.json").write_text(json.dumps(history, indent=2))

    return {"best_epoch": best_epoch, "best_val_loss": best_val, "final_val_auroc": final_aurocs}


SWEEP_VARIANTS = [(True, True), (True, False), (False, True), (False, False)]  # (pretrain, weight)


def variant_name(use_pretrain: bool, use_weights: bool) -> str:
    return f"pretrain-{'on' if use_pretrain else 'off'}_weight-{'on' if use_weights else 'off'}"


def run_seed_sweep(args, device) -> int:
    """Re-run the 4-way per-head ablation across N fixed seeds (0..N-1) and
    report mean +/- std val AUROC per variant per condition, so the earlier
    single-seed ablation deltas can be checked against run-to-run noise."""
    n_seeds = args.n_seeds
    seeds = list(range(n_seeds))
    features_path = PROCESSED_ROOT / "features.parquet"
    schema_path = PROCESSED_ROOT / "feature_schema_perhead.json"
    pretrain_weights = Path("experiments/pretrain_perhead/backbone.pt")
    sweep_root = OUT_DIR / "perhead" / "seed_sweep"

    schema = json.loads(schema_path.read_text())
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(LABELS_PATH)
    head_masks = build_head_masks(schema, device)

    n_runs = n_seeds * len(SWEEP_VARIANTS)
    print(f"seed sweep: {n_seeds} seeds x {len(SWEEP_VARIANTS)} variants = "
          f"{n_runs} runs (per-head leakage setup)")
    results: dict[str, dict[int, dict]] = {}
    t0 = time.time()
    run_i = 0
    for use_pretrain, use_weights in SWEEP_VARIANTS:
        vname = variant_name(use_pretrain, use_weights)
        results[vname] = {}
        for seed in seeds:
            run_i += 1
            out = train_and_evaluate(
                features=features, labels=labels, schema=schema, head_masks=head_masks,
                pretrain_weights=pretrain_weights, use_pretrain=use_pretrain,
                use_weights=use_weights, seed=seed, epochs=args.epochs,
                batch_size=args.batch_size, lr=args.lr, patience=args.patience,
                device=device, run_dir=None, save_checkpoint=False, verbose=False,
            )
            results[vname][seed] = out["final_val_auroc"]
            elapsed = time.time() - t0
            aurocs = {k: round(v, 3) for k, v in out["final_val_auroc"].items()}
            print(f"  [{run_i:>2}/{n_runs}] {vname} seed {seed}: {aurocs} "
                  f"(best_epoch {out['best_epoch']}, {elapsed:.0f}s elapsed)")

    sweep_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / "results.json").write_text(json.dumps(results, indent=2))
    write_sweep_report(results, n_seeds)
    return 0


def write_sweep_report(results: dict, n_seeds: int) -> None:
    variants = list(results.keys())
    stats = {}
    for vname, per_seed in results.items():
        stats[vname] = {}
        for name in LABEL_NAMES:
            vals = np.array([per_seed[s][name] for s in per_seed])
            stats[vname][name] = (float(vals.mean()), float(vals.std(ddof=1)))

    lines = [f"# Multi-seed ablation — per-head leakage setup "
             f"({n_seeds} seeds x 4 variants = {n_seeds * len(variants)} runs)\n",
             f"Validation-split AUROC, mean +/- std across seeds 0..{n_seeds - 1}. "
             "Same per-head leakage masking, features, and splits as the single-seed "
             "ablation in `results/ablation.md` — this only adds repeated seeds.\n",
             "| variant | diabetes | prediabetes | obesity | hypertension | metab. syndrome | mean |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for vname in variants:
        row_means = [stats[vname][n][0] for n in LABEL_NAMES]
        cells = " | ".join(f"{stats[vname][n][0]:.3f} ± {stats[vname][n][1]:.3f}" for n in LABEL_NAMES)
        lines.append(f"| {vname} | {cells} | {np.mean(row_means):.3f} |")

    lines += ["\n## Is the weighting / pretraining effect real, or noise?\n",
             "Marginal effect of survey weighting (mean over pretrain on/off) and of "
             "pretraining (mean over weight on/off), per condition, compared against "
             "the noise floor — the average per-variant standard deviation across "
             "seeds for that condition. If a gap is smaller than the noise floor, it "
             "isn't distinguishable from seed-to-seed randomness.\n",
             "| condition | weight-on − weight-off | pretrain-on − pretrain-off | "
             "noise floor (avg std) | weight gap real? | pretrain gap real? |",
             "|---|---:|---:|---:|:---:|:---:|"]
    for name in LABEL_NAMES:
        w_on = np.mean([stats["pretrain-on_weight-on"][name][0], stats["pretrain-off_weight-on"][name][0]])
        w_off = np.mean([stats["pretrain-on_weight-off"][name][0], stats["pretrain-off_weight-off"][name][0]])
        p_on = np.mean([stats["pretrain-on_weight-on"][name][0], stats["pretrain-on_weight-off"][name][0]])
        p_off = np.mean([stats["pretrain-off_weight-on"][name][0], stats["pretrain-off_weight-off"][name][0]])
        noise = np.mean([stats[v][name][1] for v in variants])
        weight_gap = w_on - w_off
        pretrain_gap = p_on - p_off
        lines.append(f"| {name} | {weight_gap:+.3f} | {pretrain_gap:+.3f} | {noise:.3f} "
                     f"| {'**yes**' if abs(weight_gap) > noise else 'no'} "
                     f"| {'**yes**' if abs(pretrain_gap) > noise else 'no'} |")

    out_path = Path("results/seed_sweep.md")
    out_path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nsaved -> {out_path}")


SMOTE_LABELS = ("prediabetes", "metabolic_syndrome")


def run_smote_sweep(args, device) -> int:
    """SMOTE on/off across the same N fixed seeds (0..N-1) used by the main
    seed sweep, at the established best config (per-head, pretrain-on,
    weight-off). Reports mean +/- std val AUROC per condition for both
    variants, and whether the SMOTE-vs-no-SMOTE gap for prediabetes/
    metabolic_syndrome (the two SMOTE'd conditions) survives the noise floor
    seen across seeds — the same bar the pretrain/weight effects were held to."""
    n_seeds = args.smote_seeds
    seeds = list(range(n_seeds))
    features_path = PROCESSED_ROOT / "features.parquet"
    schema_path = PROCESSED_ROOT / "feature_schema_perhead.json"
    pretrain_weights = Path("experiments/pretrain_perhead/backbone.pt")
    sweep_root = OUT_DIR / "perhead" / "smote_sweep"

    schema = json.loads(schema_path.read_text())
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(LABELS_PATH)
    head_masks = build_head_masks(schema, device)

    variants = [("smote-on", True), ("smote-off", False)]
    n_runs = n_seeds * len(variants)
    print(f"SMOTE sweep: {n_seeds} seeds x {len(variants)} variants = {n_runs} runs "
          f"(per-head, pretrain-on, weight-off; SMOTE targets: {SMOTE_LABELS})")
    results: dict[str, dict[int, dict]] = {}
    t0 = time.time()
    run_i = 0
    for vname, use_smote in variants:
        results[vname] = {}
        for seed in seeds:
            run_i += 1
            out = train_and_evaluate(
                features=features, labels=labels, schema=schema, head_masks=head_masks,
                pretrain_weights=pretrain_weights, use_pretrain=True, use_weights=False,
                seed=seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                patience=args.patience, device=device, run_dir=None, save_checkpoint=False,
                verbose=False, use_smote=use_smote, smote_labels=SMOTE_LABELS,
            )
            results[vname][seed] = out["final_val_auroc"]
            elapsed = time.time() - t0
            aurocs = {k: round(v, 3) for k, v in out["final_val_auroc"].items()}
            print(f"  [{run_i:>2}/{n_runs}] {vname} seed {seed}: {aurocs} "
                  f"(best_epoch {out['best_epoch']}, {elapsed:.0f}s elapsed)")

    sweep_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / "results.json").write_text(json.dumps(results, indent=2))
    write_smote_report(results, n_seeds)
    return 0


def write_smote_report(results: dict, n_seeds: int) -> None:
    stats = {}
    for vname, per_seed in results.items():
        stats[vname] = {}
        for name in LABEL_NAMES:
            vals = np.array([per_seed[s][name] for s in per_seed])
            stats[vname][name] = (float(vals.mean()), float(vals.std(ddof=1)))

    lines = [f"# SMOTE ablation — prediabetes & metabolic_syndrome "
             f"({n_seeds} seeds x 2 variants = {n_seeds * 2} runs)\n",
             "Validation-split AUROC, mean +/- std across seeds "
             f"0..{n_seeds - 1}, at the established best config (per-head, "
             "pretrain-on, weight-off — same seeds used in `results/seed_sweep.md`). "
             f"SMOTE (`SMOTENC`, training split only, never val/test) balances "
             f"{' and '.join(SMOTE_LABELS)} to parity; every other head trains "
             "on real data only and ignores the synthetic rows via `_valid=0`.\n",
             "| variant | diabetes | prediabetes | obesity | hypertension | metab. syndrome | mean |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for vname in ("smote-on", "smote-off"):
        row_means = [stats[vname][n][0] for n in LABEL_NAMES]
        cells = " | ".join(f"{stats[vname][n][0]:.3f} ± {stats[vname][n][1]:.3f}" for n in LABEL_NAMES)
        lines.append(f"| {vname} | {cells} | {np.mean(row_means):.3f} |")

    lines += ["\n## Does the SMOTE gap survive multiple seeds?\n",
             "Same rule as the pretrain/weight ablation: a gap only counts if "
             "it's bigger than the noise floor (the larger of the two "
             "variants' per-condition standard deviation across seeds).\n",
             "| condition | smote-on − smote-off | noise floor (max of the two std) | gap real? | SMOTE'd this condition? |",
             "|---|---:|---:|:---:|:---:|"]
    for name in LABEL_NAMES:
        on_mean, on_std = stats["smote-on"][name]
        off_mean, off_std = stats["smote-off"][name]
        gap = on_mean - off_mean
        noise = max(on_std, off_std)
        targeted = "yes" if name in SMOTE_LABELS else "no (control)"
        lines.append(f"| {name} | {gap:+.3f} | {noise:.3f} "
                     f"| {'**yes**' if abs(gap) > noise else 'no'} | {targeted} |")

    out_path = Path("results/smote_sweep.md")
    out_path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nsaved -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Random-init backbone instead of loading pretrained weights.")
    parser.add_argument("--no-survey-weight", action="store_true",
                        help="Use uniform weights instead of survey weights.")
    parser.add_argument("--per-head", action="store_true",
                        help="Middle-ground leakage: full feature set, mask each label own defining features per head.")
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="Instead of one run, re-run the 4-way per-head ablation "
                             "across N fixed seeds (0..N-1) and report mean+-std val "
                             "AUROC per variant per condition to results/seed_sweep.md.")
    parser.add_argument("--smote", action="store_true",
                        help="SMOTE-balance prediabetes & metabolic_syndrome in the "
                             "training split only (never val/test) before fine-tuning.")
    parser.add_argument("--smote-seeds", type=int, default=None,
                        help="Instead of one run, compare SMOTE on/off across N fixed "
                             "seeds (0..N-1) at the best per-head/pretrain-on config, "
                             "reporting mean+-std val AUROC to results/smote_sweep.md.")
    args = parser.parse_args(argv)

    device = pick_device()
    if args.smote_seeds is not None:
        print(f"device: {device}")
        return run_smote_sweep(args, device)
    if args.n_seeds is not None:
        print(f"device: {device}")
        return run_seed_sweep(args, device)

    use_weights = not args.no_survey_weight
    tag = (f"mode={'per-head' if args.per_head else 'strict'} "
           f"pretrain={'off' if args.no_pretrain else 'on'} "
           f"survey_weight={'off' if not use_weights else 'on'}")
    print(f"device: {device} | {tag} | seed: {args.seed}")

    # Per-head (middle-ground) mode uses the full repopulated feature set + the
    # perhead schema (with head_masks) + a backbone pretrained on all 6 groups.
    if args.per_head:
        features_path = PROCESSED_ROOT / "features.parquet"
        schema_path = PROCESSED_ROOT / "feature_schema_perhead.json"
        pretrain_weights = Path("experiments/pretrain_perhead/backbone.pt")
        run_base = OUT_DIR / "perhead"
    else:
        features_path, schema_path = FEATURES_PATH, SCHEMA_PATH
        pretrain_weights = PRETRAIN_WEIGHTS
        run_base = OUT_DIR / "strict"

    # Each ablation variant writes to its own subdir so runs never clobber.
    run_dir = run_base / (f"pretrain-{'off' if args.no_pretrain else 'on'}"
                          f"_weight-{'off' if not use_weights else 'on'}")
    run_dir.mkdir(parents=True, exist_ok=True)

    schema = json.loads(schema_path.read_text())
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(LABELS_PATH)
    head_masks = build_head_masks(schema, device) if args.per_head else None

    backbone_preview = Backbone(schema)
    print(backbone_preview.describe())
    if not args.no_pretrain:
        print(f"loading pretrained backbone <- {pretrain_weights}")

    print(f"\n{'epoch':>5}{'train':>9}{'val':>9}   " +
          "".join(f"{n[:4]:>7}" for n in LABEL_NAMES) + "  (val AUROC)")
    out = train_and_evaluate(
        features=features, labels=labels, schema=schema, head_masks=head_masks,
        pretrain_weights=pretrain_weights, use_pretrain=not args.no_pretrain,
        use_weights=use_weights, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, patience=args.patience,
        device=device, run_dir=run_dir, save_checkpoint=True, verbose=True,
        use_smote=args.smote,
    )
    best_epoch, best_val, final_aurocs = out["best_epoch"], out["best_val_loss"], out["final_val_auroc"]

    config = vars(args) | {"device": str(device), "best_epoch": best_epoch,
                           "best_val_loss": best_val, "use_survey_weight": use_weights,
                           "final_val_auroc": final_aurocs}
    (run_dir / "finetune_config.json").write_text(json.dumps(config, indent=2))

    print(f"\nbest val loss {best_val:.4f} @ epoch {best_epoch}")
    print("final val AUROC:", {k: round(v, 3) for k, v in final_aurocs.items()})
    print(f"saved -> {run_dir}/model.pt (+ config, history)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
