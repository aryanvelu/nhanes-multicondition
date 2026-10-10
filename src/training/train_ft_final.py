"""Train and save the canonical FT-Transformer checkpoint on the current
(fully-enriched) per-head feature set, for use as the deep-model base
learner in the stacking ensemble.

The 10-seed sweeps (`architecture_sweep.py`, `ft_whq_sweep.py`,
`ft_organlab_sweep.py`) all used ``save_checkpoint=False`` — they exist to
measure mean +/- std across seeds, not to produce a deployable model. This
script does the single canonical run (seed 42, matching every other "the"
checkpoint in this project — e.g. the grouped-MLP's
``experiments/finetune/perhead/pretrain-on_weight-off/model.pt``), at the
settings this session established as best: per-head leakage masking,
**no** pretraining (settled negative — see ``results/ft_pretrain_sweep.md``),
survey-weight off (the best-AUROC config every other baseline in this
project is compared against).

Usage
-----
    python -m src.training.train_ft_final
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.models.ft_transformer import FTTransformerBackbone
from src.training.finetune import LABEL_NAMES, build_head_masks, pick_device, train_and_evaluate

PROCESSED_ROOT = Path("data/processed")
FEATURES_PATH = PROCESSED_ROOT / "features.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_perhead.json"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"
PRETRAIN_WEIGHTS = Path("experiments/pretrain_ft_transformer/backbone.pt")  # unused (no pretrain)
RUN_DIR = Path("experiments/finetune/ft_transformer")
SEED = 42


def main() -> int:
    device = pick_device()
    schema = json.loads(SCHEMA_PATH.read_text())
    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    head_masks = build_head_masks(schema, device)

    print(f"Training canonical FT-Transformer (seed={SEED}, no pretrain, "
          f"weight-off, per-head) on {len(schema['continuous'])} continuous + "
          f"{len(schema['categorical'])} categorical features, device={device}")

    out = train_and_evaluate(
        features=features, labels=labels, schema=schema, head_masks=head_masks,
        pretrain_weights=PRETRAIN_WEIGHTS, use_pretrain=False, use_weights=False,
        seed=SEED, epochs=150, batch_size=256, lr=1e-3, patience=20,
        device=device, run_dir=RUN_DIR, save_checkpoint=True, verbose=True,
        backbone_cls=FTTransformerBackbone,
    )

    config = {"seed": SEED, "use_pretrain": False, "use_weights": False,
             "backbone": "FTTransformerBackbone", "best_epoch": out["best_epoch"],
             "best_val_loss": out["best_val_loss"], "final_val_auroc": out["final_val_auroc"]}
    (RUN_DIR / "finetune_config.json").write_text(json.dumps(config, indent=2))

    print(f"\nbest val loss {out['best_val_loss']:.4f} @ epoch {out['best_epoch']}")
    print("final val AUROC:", {k: round(v, 3) for k, v in out["final_val_auroc"].items()})
    print(f"saved -> {RUN_DIR}/model.pt (+ config)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
