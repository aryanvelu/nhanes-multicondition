"""Masked-feature self-supervised pretraining of the shared backbone.

Idea (borrowed from masked language modeling, adapted to tabular health data):
before showing the model any diagnosis labels, teach it the *internal structure*
of an NHANES profile. Each mini-batch, we randomly hide a fraction of every
participant's features and ask the backbone to fill them back in. To do that
well the network must learn how the variables relate (e.g. older age tracks with
higher cholesterol), which is exactly the prior knowledge that helps the later
supervised heads — especially since only ~6k people have the full fasting panel.

Because no labels are used, pretraining can run on the *entire* training split,
not just the fully-labelled subset.

What gets hidden and scored
---------------------------
* A masked **continuous** feature has its value zeroed and its "missing" flag set
  to 1 (so corruption looks exactly like real missingness). It is scored with
  mean-squared error (MSE) — average squared gap between predicted and true value.
* A masked **categorical** feature has its code set to 0 (the model's existing
  "unknown" token) and is scored with cross-entropy (CE) — the standard
  classification loss over its possible categories.
* Loss is computed only where a feature was masked *and* was genuinely observed
  to begin with, so the model is never graded on reconstructing an imputed hole.

The network = the shared ``Backbone`` (imported, not redefined) + a lightweight
reconstruction head that is discarded afterwards. Only the backbone weights are
saved, to ``experiments/pretrain/backbone.pt``. Reconstruction loss on the
validation split drives early stopping.

Usage
-----
    python -m src.training.pretrain
    python -m src.training.pretrain --epochs 200 --mask-rate 0.3 --batch-size 512
"""

from __future__ import annotations

import argparse
import json
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
SPLITS_ROOT = Path("data/splits")
OUT_DIR = Path("experiments/pretrain")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(name: str, features: pd.DataFrame, schema: dict):
    """Return (x_cont, x_miss, x_cat) tensors for the participants in a split."""
    ids = pd.read_csv(SPLITS_ROOT / f"{name}.csv")
    sub = ids.merge(features, on=["SEQN", "cycle"], how="left", validate="one_to_one")
    x_cont, x_miss, x_cat, _, _ = assemble_model_inputs(sub, schema)
    return (torch.from_numpy(x_cont), torch.from_numpy(x_miss), torch.from_numpy(x_cat))


class ReconstructionHead(nn.Module):
    """Maps the shared embedding back to continuous values + categorical logits."""

    def __init__(self, embedding_dim: int, n_continuous: int,
                 categorical_cardinalities: dict[str, int]):
        super().__init__()
        self.continuous = nn.Linear(embedding_dim, n_continuous)
        self.categorical = nn.ModuleList(
            [nn.Linear(embedding_dim, card) for card in categorical_cardinalities.values()]
        )

    def forward(self, z: torch.Tensor):
        return self.continuous(z), [head(z) for head in self.categorical]


def mask_and_loss(backbone, head, x_cont, x_miss, x_cat, mask_rate, generator):
    """Corrupt a batch, reconstruct, and return the combined MSE + CE loss."""
    device = x_cont.device
    # Independent Bernoulli(mask_rate) masks over every feature.
    m_cont = (torch.rand(x_cont.shape, device=device, generator=generator) < mask_rate)
    m_cat = (torch.rand(x_cat.shape, device=device, generator=generator) < mask_rate)

    # Corrupt: hidden continuous -> value 0 + flag 1; hidden categorical -> code 0.
    xc_in = x_cont.masked_fill(m_cont, 0.0)
    xm_in = torch.maximum(x_miss, m_cont.float())
    xk_in = x_cat.masked_fill(m_cat, 0)

    z = backbone(xc_in, xm_in, xk_in)
    cont_pred, cat_logits = head(z)

    # Continuous MSE, only where masked AND originally observed (flag == 0).
    cont_scored = m_cont & (x_miss == 0)
    denom = cont_scored.sum().clamp(min=1)
    mse = (((cont_pred - x_cont) ** 2) * cont_scored).sum() / denom

    # Categorical CE, only where masked AND originally observed (code != 0).
    ce_sum = x_cont.new_zeros(())
    ce_count = x_cont.new_zeros(())
    for k, logits in enumerate(cat_logits):
        target = x_cat[:, k]
        scored = m_cat[:, k] & (target != 0)
        if scored.any():
            per = nn.functional.cross_entropy(logits, target, reduction="none")
            ce_sum = ce_sum + (per * scored).sum()
            ce_count = ce_count + scored.sum()
    ce = ce_sum / ce_count.clamp(min=1)
    return mse + ce, mse.detach(), ce.detach()


def run_epoch(loader, backbone, head, mask_rate, generator, optimizer=None):
    train = optimizer is not None
    backbone.train(train)
    head.train(train)
    totals = np.zeros(3)  # total, mse, ce
    n_batches = 0
    for xc, xm, xk in loader:
        loss, mse, ce = mask_and_loss(backbone, head, xc, xm, xk, mask_rate, generator)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        totals += [loss.item(), mse.item(), ce.item()]
        n_batches += 1
    return totals / max(n_batches, 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mask-rate", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", default=str(FEATURES_PATH),
                        help="Feature parquet to pretrain on.")
    parser.add_argument("--schema", default=str(SCHEMA_PATH),
                        help="Feature schema JSON matching --features.")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help="Where to save backbone weights + history.")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"device: {device} | mask-rate: {args.mask_rate} | seed: {args.seed}")

    schema = json.loads(Path(args.schema).read_text())
    features = pd.read_parquet(args.features)

    train_t = [t.to(device) for t in load_split("train", features, schema)]
    val_t = [t.to(device) for t in load_split("val", features, schema)]
    print(f"train participants: {train_t[0].shape[0]:,} | "
          f"val: {val_t[0].shape[0]:,}")

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=args.batch_size)

    backbone = Backbone(schema).to(device)
    head = ReconstructionHead(backbone.output_dim, len(backbone.continuous_features),
                              backbone.categorical_cardinalities).to(device)
    print(backbone.describe())

    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()), lr=args.lr
    )
    # Separate generators so train masking is stochastic but val masking is fixed
    # (same masks every epoch => comparable validation numbers for early stopping).
    train_gen = torch.Generator(device=device).manual_seed(args.seed)
    val_gen_seed = args.seed + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = -1
    history = []

    print(f"\n{'epoch':>5}{'train':>10}{'val':>10}{'val_mse':>10}{'val_ce':>10}")
    for epoch in range(1, args.epochs + 1):
        run_epoch(train_loader, backbone, head, args.mask_rate, train_gen, optimizer)
        with torch.no_grad():
            val_gen = torch.Generator(device=device).manual_seed(val_gen_seed)
            tr = run_epoch(train_loader, backbone, head, args.mask_rate,
                           torch.Generator(device=device).manual_seed(args.seed))
            val = run_epoch(val_loader, backbone, head, args.mask_rate, val_gen)
        history.append({"epoch": epoch, "train_loss": tr[0], "val_loss": val[0],
                        "val_mse": val[1], "val_ce": val[2]})
        marker = ""
        if val[0] < best_val - 1e-4:
            best_val, best_epoch = val[0], epoch
            torch.save(backbone.state_dict(), out_dir / "backbone.pt")
            marker = "  *saved"
        print(f"{epoch:>5}{tr[0]:>10.4f}{val[0]:>10.4f}{val[1]:>10.4f}"
              f"{val[2]:>10.4f}{marker}")
        if epoch - best_epoch >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs")
            break

    config = vars(args) | {"device": str(device), "best_epoch": best_epoch,
                           "best_val_loss": best_val,
                           "embedding_dim": backbone.output_dim}
    (out_dir / "pretrain_config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "pretrain_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest val loss {best_val:.4f} @ epoch {best_epoch}")
    print(f"saved -> {out_dir}/backbone.pt (+ config, history)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
