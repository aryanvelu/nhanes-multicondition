"""Masked-feature-TOKEN self-supervised pretraining for the FT-Transformer backbone.

Same idea as ``pretrain.py`` — hide some of a participant's features, teach
the network to fill them back in, so it learns how NHANES variables relate to
each other before ever seeing a diagnosis label — but adapted to how the
FT-Transformer represents a person: as a *sequence of feature tokens* rather
than one flat vector.

What's different from ``pretrain.py``'s masking
-------------------------------------------------
``pretrain.py`` masks a **flat-vector element**: it zeros a continuous
feature's *value* and sets its missing-flag to 1 *before* the backbone ever
sees it, so masking and real missingness look identical at the input.

Here, masking happens **after tokenization, before the encoder**: every
feature is first turned into its own token as usual
(``FTTransformerBackbone.tokenize()``), then a random subset of *token
positions* gets **replaced wholesale** by one shared, learned ``[MASK]``
token — the same token value regardless of which feature lived there,
exactly like BERT's ``[MASK]``. The network can only recover what was there
by attending to the *other*, unmasked tokens through self-attention; unlike
``pretrain.py``, it can't lean on "value near 0 with flag=1 probably means
missing," because the masked token carries no per-feature identity at all.

Reconstruction heads
---------------------
* All **continuous** masked positions share ONE regression head
  (``Linear(d_model, 1)``) — reasonable because every continuous feature is
  already z-scored to the same mean-0/std-1 scale, so one shared head sees a
  consistent target distribution regardless of which feature is at that
  position (the same reason BERT can use one shared vocabulary-sized output
  head for every masked word position).
* Each **categorical** feature gets its own classification head (cardinalities
  differ), applied only at that feature's own token position.
* Loss is scored only where a token was masked *and* genuinely observed to
  begin with (same "never grade a reconstruction of an imputed hole" rule as
  ``pretrain.py``): MSE for continuous, cross-entropy for categorical.

Trains on the **training split only**; validation-split reconstruction loss
drives early stopping, exactly like ``pretrain.py``. Only the backbone
(tokenizers, embeddings, ``[CLS]``/``[MASK]`` tokens, encoder) is saved —
the reconstruction heads are discarded, same convention as ``pretrain.py``.

Usage
-----
    python -m src.training.pretrain_ft
    python -m src.training.pretrain_ft --epochs 200 --mask-rate 0.3
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

from src.models.ft_transformer import FTTransformerBackbone
from src.models.structured_layer import assemble_model_inputs

PROCESSED_ROOT = Path("data/processed")
FEATURES_PATH = PROCESSED_ROOT / "features.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_perhead.json"
SPLITS_ROOT = Path("data/splits")
OUT_DIR = Path("experiments/pretrain_ft_transformer")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(name: str, features: pd.DataFrame, schema: dict):
    ids = pd.read_csv(SPLITS_ROOT / f"{name}.csv")
    sub = ids.merge(features, on=["SEQN", "cycle"], how="left", validate="one_to_one")
    x_cont, x_miss, x_cat, _, _ = assemble_model_inputs(sub, schema)
    return (torch.from_numpy(x_cont), torch.from_numpy(x_miss), torch.from_numpy(x_cat))


class TokenReconstructionHeads(nn.Module):
    """One shared head for all continuous-token positions; one classification
    head per categorical feature (their cardinalities differ)."""

    def __init__(self, d_model: int, categorical_cardinalities: dict[str, int]):
        super().__init__()
        self.continuous_head = nn.Linear(d_model, 1)
        self.categorical_heads = nn.ModuleDict({
            c: nn.Linear(d_model, card) for c, card in categorical_cardinalities.items()
        })


def mask_and_loss(backbone: FTTransformerBackbone, heads: TokenReconstructionHeads,
                  x_cont, x_miss, x_cat, mask_rate, generator):
    """Mask a random subset of feature TOKENS (not flat-vector elements),
    reconstruct each masked position from the others via self-attention."""
    device = x_cont.device
    n_cont = x_cont.shape[1]

    tokens = backbone.tokenize(x_cont, x_miss, x_cat)  # [B, n_features, d_model]
    mask = (torch.rand((tokens.shape[0], tokens.shape[1]), device=device,
                       generator=generator) < mask_rate)
    mask_token = backbone.mask_token.expand(tokens.shape[0], tokens.shape[1], -1)
    masked_tokens = torch.where(mask.unsqueeze(-1), mask_token, tokens)

    encoded = backbone.encode(masked_tokens)  # [B, 1+n_features, d_model]
    feature_out = encoded[:, 1:, :]           # drop [CLS], align with tokens/mask

    cont_out, cat_out = feature_out[:, :n_cont, :], feature_out[:, n_cont:, :]
    cont_mask, cat_mask = mask[:, :n_cont], mask[:, n_cont:]

    # Continuous: MSE, only where masked AND originally observed (flag == 0).
    cont_pred = heads.continuous_head(cont_out).squeeze(-1)
    cont_scored = cont_mask & (x_miss == 0)
    denom = cont_scored.sum().clamp(min=1)
    mse = (((cont_pred - x_cont) ** 2) * cont_scored).sum() / denom

    # Categorical: CE, only where masked AND originally observed (code != 0).
    ce_sum = x_cont.new_zeros(())
    ce_count = x_cont.new_zeros(())
    for k, c in enumerate(backbone.categorical_features):
        logits = heads.categorical_heads[c](cat_out[:, k, :])
        target = x_cat[:, k]
        scored = cat_mask[:, k] & (target != 0)
        if scored.any():
            per = nn.functional.cross_entropy(logits, target, reduction="none")
            ce_sum = ce_sum + (per * scored).sum()
            ce_count = ce_count + scored.sum()
    ce = ce_sum / ce_count.clamp(min=1)

    return mse + ce, mse.detach(), ce.detach()


def run_epoch(loader, backbone, heads, mask_rate, generator, optimizer=None):
    train = optimizer is not None
    backbone.train(train)
    heads.train(train)
    totals = np.zeros(3)
    n_batches = 0
    for xc, xm, xk in loader:
        loss, mse, ce = mask_and_loss(backbone, heads, xc, xm, xk, mask_rate, generator)
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
    parser.add_argument("--features", default=str(FEATURES_PATH))
    parser.add_argument("--schema", default=str(SCHEMA_PATH))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"device: {device} | mask-rate (per TOKEN): {args.mask_rate} | seed: {args.seed}")

    schema = json.loads(Path(args.schema).read_text())
    features = pd.read_parquet(args.features)

    train_t = [t.to(device) for t in load_split("train", features, schema)]
    val_t = [t.to(device) for t in load_split("val", features, schema)]
    print(f"train participants: {train_t[0].shape[0]:,} | val: {val_t[0].shape[0]:,}")

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=args.batch_size)

    backbone = FTTransformerBackbone(schema).to(device)
    heads = TokenReconstructionHeads(backbone.output_dim, backbone.categorical_cardinalities).to(device)
    print(backbone.describe())

    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(heads.parameters()), lr=args.lr
    )
    train_gen = torch.Generator(device=device).manual_seed(args.seed)
    val_gen_seed = args.seed + 1

    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = -1
    history = []

    print(f"\n{'epoch':>5}{'train':>10}{'val':>10}{'val_mse':>10}{'val_ce':>10}")
    for epoch in range(1, args.epochs + 1):
        run_epoch(train_loader, backbone, heads, args.mask_rate, train_gen, optimizer)
        with torch.no_grad():
            val_gen = torch.Generator(device=device).manual_seed(val_gen_seed)
            tr = run_epoch(train_loader, backbone, heads, args.mask_rate,
                          torch.Generator(device=device).manual_seed(args.seed))
            val = run_epoch(val_loader, backbone, heads, args.mask_rate, val_gen)
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
                           "best_val_loss": best_val, "embedding_dim": backbone.output_dim}
    (out_dir / "pretrain_config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "pretrain_history.json").write_text(json.dumps(history, indent=2))
    print(f"\nbest val loss {best_val:.4f} @ epoch {best_epoch}")
    print(f"saved -> {out_dir}/backbone.pt (+ config, history)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
