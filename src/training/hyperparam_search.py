"""Hyperparameter search for the deep model and the XGBoost baseline.

For each model, a random search **starting from the previously-established
winning configuration** — per-head leakage masking with pretraining on for
the deep model (`pretrain-on_weight-off`, the best config from the 4-way
ablation), the existing baseline settings for XGBoost — varying exactly the
hyperparameters requested:

* Deep model: learning rate, hidden-layer width (`Backbone`'s `hidden_dims`,
  applied uniformly to both FC layers), dropout (shared by the backbone MLP
  and the classification heads).
* XGBoost: `max_depth`, `learning_rate`, `n_estimators`.

Selection metric: mean **unweighted validation AUROC** across the 5
conditions — the same metric the earlier 4-way ablation used to pick its
winning variant. The winning config of each model is then confirmed on the
untouched **test** split with survey-weighted AUROC/AUPRC.

Architecture note: the pretrained backbone checkpoint
(`experiments/pretrain_perhead/backbone.pt`) was fit at `hidden_dims=(64,64)`.
Any trial that changes the hidden width no longer matches those weight
shapes, so pretraining is automatically skipped (random init instead) for
those trials — flagged plainly in the trial log rather than silently reusing
mismatched or default weights.

Usage
-----
    python -m src.training.hyperparam_search                # both models
    python -m src.training.hyperparam_search --model deep
    python -m src.training.hyperparam_search --model xgboost
    python -m src.training.hyperparam_search --n-trials 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from src.baselines.xgboost_baseline import build_head_frame
from src.baselines.xgboost_baseline import load_split as load_xgb_split
from src.data.labels import LABEL_NAMES
from src.evaluation.metrics import COND_WEIGHT, weighted_auprc, weighted_auroc
from src.evaluation.stacking import load_split_arrays
from src.models.backbone import Backbone
from src.training.finetune import (
    MultiHead, auroc, build_head_masks, compute_logits, pick_device, train_and_evaluate,
)

PROCESSED_ROOT = Path("data/processed")
FEATURES_PATH = PROCESSED_ROOT / "features.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_perhead.json"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"
PRETRAIN_WEIGHTS = Path("experiments/pretrain_perhead/backbone.pt")
OUT_DIR = Path("experiments/hyperparam_search")
OUT_MD = Path("results/hyperparam_search.md")

DEFAULT_ARCH_WIDTH = 64  # matches the pretrained backbone's hidden_dims=(64,64)
SEED = 42


# --------------------------------------------------------------------------- #
# Deep model search
# --------------------------------------------------------------------------- #

def sample_deep_configs(n: int, rng: np.random.Generator) -> list[dict]:
    lrs = [5e-4, 1e-3, 2e-3, 3e-3]
    widths = [32, 64, 128]
    dropouts = [0.1, 0.2, 0.3, 0.4]
    configs = [{"lr": 1e-3, "width": 64, "dropout": 0.2}]  # the known starting point
    seen = {(1e-3, 64, 0.2)}
    while len(configs) < n:
        c = (float(rng.choice(lrs)), int(rng.choice(widths)), float(rng.choice(dropouts)))
        if c not in seen:
            seen.add(c)
            configs.append({"lr": c[0], "width": c[1], "dropout": c[2]})
    return configs


def load_and_predict_deep(model_path, schema, device, data, head_masks,
                          hidden_dims, dropout) -> np.ndarray:
    backbone = Backbone(schema, hidden_dims=hidden_dims).to(device)
    heads = MultiHead(backbone.output_dim, dropout=dropout).to(device)
    ckpt = torch.load(model_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone"]); heads.load_state_dict(ckpt["heads"])
    backbone.eval(); heads.eval()
    xc, xm, xk = [torch.from_numpy(data[k]).to(device) for k in ("xc", "xm", "xk")]
    with torch.no_grad():
        logits = compute_logits(backbone, heads, xc, xm, xk, head_masks)
        return torch.sigmoid(logits).cpu().numpy()


def confirm_deep_on_test(schema, features, labels, head_masks, device, best) -> tuple[dict, dict]:
    run_dir = OUT_DIR / "deep_best"
    hidden_dims = (best["width"], best["width"])
    train_and_evaluate(
        features=features, labels=labels, schema=schema, head_masks=head_masks,
        pretrain_weights=PRETRAIN_WEIGHTS, use_pretrain=best["pretrain_used"],
        use_weights=False, seed=SEED, epochs=150, batch_size=256,
        lr=best["lr"], patience=20, device=device, run_dir=run_dir,
        save_checkpoint=True, verbose=False,
        backbone_kwargs={"hidden_dims": hidden_dims}, head_dropout=best["dropout"],
    )
    test = load_split_arrays("test", schema)
    probs = load_and_predict_deep(run_dir / "model.pt", schema, device, test,
                                  head_masks, hidden_dims, best["dropout"])
    test_auroc, test_auprc = {}, {}
    for k, name in enumerate(LABEL_NAMES):
        w = test["weights"][COND_WEIGHT[name]]
        mw = (test["valid"][:, k] == 1) & (w > 0)
        y, s, wt = test["y"][mw, k], probs[mw, k], w[mw]
        test_auroc[name] = weighted_auroc(y, s, wt)
        test_auprc[name] = weighted_auprc(y, s, wt)
    return test_auroc, test_auprc


def run_deep_search(n_trials: int) -> dict:
    device = pick_device()
    schema = json.loads(SCHEMA_PATH.read_text())
    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    head_masks = build_head_masks(schema, device)

    rng = np.random.default_rng(SEED)
    configs = sample_deep_configs(n_trials, rng)

    print(f"\n=== Deep model hyperparameter search: {len(configs)} trials "
          f"(device={device}) ===")
    results = []
    for i, cfg in enumerate(configs, 1):
        can_pretrain = cfg["width"] == DEFAULT_ARCH_WIDTH
        out = train_and_evaluate(
            features=features, labels=labels, schema=schema, head_masks=head_masks,
            pretrain_weights=PRETRAIN_WEIGHTS, use_pretrain=can_pretrain,
            use_weights=False, seed=SEED, epochs=150, batch_size=256,
            lr=cfg["lr"], patience=20, device=device, run_dir=None,
            save_checkpoint=False, verbose=False,
            backbone_kwargs={"hidden_dims": (cfg["width"], cfg["width"])},
            head_dropout=cfg["dropout"],
        )
        mean_val_auroc = float(np.mean(list(out["final_val_auroc"].values())))
        note = "" if can_pretrain else "  (pretrain unavailable: width != 64, random init)"
        print(f"  [{i:>2}/{len(configs)}] lr={cfg['lr']:<7} width={cfg['width']:<4} "
              f"dropout={cfg['dropout']:<4} -> mean val AUROC {mean_val_auroc:.4f}{note}")
        results.append({**cfg, "pretrain_used": can_pretrain,
                        "val_auroc": out["final_val_auroc"], "mean_val_auroc": mean_val_auroc})

    best = max(results, key=lambda r: r["mean_val_auroc"])
    print(f"\nBest deep config: lr={best['lr']} width={best['width']} "
          f"dropout={best['dropout']} pretrain_used={best['pretrain_used']} "
          f"-> mean val AUROC {best['mean_val_auroc']:.4f}")

    test_auroc, test_auprc = confirm_deep_on_test(schema, features, labels, head_masks, device, best)
    print("test AUROC (best deep config):", {k: round(v, 3) for k, v in test_auroc.items()})
    return {"trials": results, "best": best, "test_auroc": test_auroc, "test_auprc": test_auprc}


# --------------------------------------------------------------------------- #
# XGBoost search
# --------------------------------------------------------------------------- #

def sample_xgb_configs(n: int, rng: np.random.Generator) -> list[dict]:
    depths = [3, 4, 5, 6]
    lrs = [0.01, 0.03, 0.05, 0.1]
    n_ests = [300, 600, 1000]
    configs = [{"max_depth": 4, "learning_rate": 0.05, "n_estimators": 1000}]  # baseline
    seen = {(4, 0.05, 1000)}
    while len(configs) < n:
        c = (int(rng.choice(depths)), float(rng.choice(lrs)), int(rng.choice(n_ests)))
        if c not in seen:
            seen.add(c)
            configs.append({"max_depth": c[0], "learning_rate": c[1], "n_estimators": c[2]})
    return configs


def fit_xgb_label(name, cfg, schema, head_masks, train_feat, train_lab, val_feat, val_lab):
    masked = head_masks[name]
    Xtr = build_head_frame(train_feat, schema, masked)
    Xva = build_head_frame(val_feat, schema, masked)
    vtr = train_lab[f"{name}_valid"].to_numpy() == 1
    vva = val_lab[f"{name}_valid"].to_numpy() == 1
    ytr = train_lab[name].to_numpy()[vtr]
    yva = val_lab[name].to_numpy()[vva]

    model = XGBClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"], subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, enable_categorical=True, tree_method="hist",
        eval_metric="auc", early_stopping_rounds=30, random_state=SEED,
    )
    model.fit(Xtr[vtr], ytr, eval_set=[(Xva[vva], yva)], verbose=False)
    return model, Xva[vva], yva


def run_xgb_search(n_trials: int) -> dict:
    schema = json.loads(SCHEMA_PATH.read_text())
    head_masks = schema["head_masks"]
    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    train_feat, train_lab = load_xgb_split("train", features, labels)
    val_feat, val_lab = load_xgb_split("val", features, labels)
    test_feat, test_lab = load_xgb_split("test", features, labels)

    rng = np.random.default_rng(SEED)
    configs = sample_xgb_configs(n_trials, rng)

    print(f"\n=== XGBoost hyperparameter search: {len(configs)} trials ===")
    results = []
    for i, cfg in enumerate(configs, 1):
        per_label_val_auroc = {}
        for name in LABEL_NAMES:
            model, Xva_valid, yva = fit_xgb_label(
                name, cfg, schema, head_masks, train_feat, train_lab, val_feat, val_lab)
            probs_va = model.predict_proba(Xva_valid)[:, 1]
            per_label_val_auroc[name] = auroc(yva, probs_va)
        mean_val_auroc = float(np.mean(list(per_label_val_auroc.values())))
        print(f"  [{i:>2}/{len(configs)}] depth={cfg['max_depth']} "
              f"lr={cfg['learning_rate']:<5} n_est={cfg['n_estimators']:<5} "
              f"-> mean val AUROC {mean_val_auroc:.4f}")
        results.append({**cfg, "val_auroc": per_label_val_auroc, "mean_val_auroc": mean_val_auroc})

    best = max(results, key=lambda r: r["mean_val_auroc"])
    print(f"\nBest XGBoost config: max_depth={best['max_depth']} "
          f"learning_rate={best['learning_rate']} n_estimators={best['n_estimators']} "
          f"-> mean val AUROC {best['mean_val_auroc']:.4f}")

    test_auroc, test_auprc = {}, {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in LABEL_NAMES:
        model, _, _ = fit_xgb_label(
            name, best, schema, head_masks, train_feat, train_lab, val_feat, val_lab)
        model.save_model(OUT_DIR / f"xgb_best_{name}.json")

        masked = head_masks[name]
        Xte = build_head_frame(test_feat, schema, masked)
        vte = test_lab[f"{name}_valid"].to_numpy() == 1
        w = test_feat.loc[vte, COND_WEIGHT[name]].to_numpy(dtype=np.float64)
        y = test_lab[name].to_numpy()[vte]
        mw = w > 0
        y, w = y[mw], w[mw]
        Xte_valid = Xte[vte].reset_index(drop=True)
        probs = model.predict_proba(Xte_valid[mw])[:, 1]
        test_auroc[name] = weighted_auroc(y, probs, w)
        test_auprc[name] = weighted_auprc(y, probs, w)

    print("test AUROC (best XGBoost config):", {k: round(v, 3) for k, v in test_auroc.items()})
    return {"trials": results, "best": best, "test_auroc": test_auroc, "test_auprc": test_auprc}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def write_report(deep_result: dict | None, xgb_result: dict | None) -> None:
    lines = ["# Hyperparameter search — deep model and XGBoost\n",
             "Random search from each model's previously-established winning "
             "config, selected by mean **unweighted validation AUROC** across "
             "the 5 conditions (same method the 4-way ablation used), then "
             "confirmed on the untouched test split with survey-weighted "
             "AUROC/AUPRC.\n"]

    if deep_result:
        lines += ["## Deep model\n",
                 "Swept: learning rate, hidden-layer width, dropout. Width != 64 "
                 "means the pretrained backbone's weights don't fit, so those "
                 "trials fall back to random init (flagged below).\n",
                 "| lr | width | dropout | pretrain used | mean val AUROC |",
                 "|---:|---:|---:|:---:|---:|"]
        for r in sorted(deep_result["trials"], key=lambda r: -r["mean_val_auroc"]):
            lines.append(f"| {r['lr']} | {r['width']} | {r['dropout']} "
                        f"| {'yes' if r['pretrain_used'] else 'no'} "
                        f"| {r['mean_val_auroc']:.4f} |")
        best = deep_result["best"]
        lines += [f"\n**Best config:** lr={best['lr']}, width={best['width']}, "
                 f"dropout={best['dropout']}, pretrain_used={best['pretrain_used']} "
                 f"(mean val AUROC {best['mean_val_auroc']:.4f})\n",
                 "| condition | val AUROC (best config) | test AUROC (wt.) | test AUPRC (wt.) |",
                 "|---|---:|---:|---:|"]
        for name in LABEL_NAMES:
            lines.append(f"| {name} | {best['val_auroc'][name]:.3f} "
                        f"| {deep_result['test_auroc'][name]:.3f} "
                        f"| {deep_result['test_auprc'][name]:.3f} |")

    if xgb_result:
        lines += ["\n## XGBoost\n",
                 "Swept: `max_depth`, `learning_rate`, `n_estimators` "
                 "(early stopping, 30 rounds, stays on throughout).\n",
                 "| max_depth | learning_rate | n_estimators | mean val AUROC |",
                 "|---:|---:|---:|---:|"]
        for r in sorted(xgb_result["trials"], key=lambda r: -r["mean_val_auroc"]):
            lines.append(f"| {r['max_depth']} | {r['learning_rate']} "
                        f"| {r['n_estimators']} | {r['mean_val_auroc']:.4f} |")
        best = xgb_result["best"]
        lines += [f"\n**Best config:** max_depth={best['max_depth']}, "
                 f"learning_rate={best['learning_rate']}, "
                 f"n_estimators={best['n_estimators']} "
                 f"(mean val AUROC {best['mean_val_auroc']:.4f})\n",
                 "| condition | val AUROC (best config) | test AUROC (wt.) | test AUPRC (wt.) |",
                 "|---|---:|---:|---:|"]
        for name in LABEL_NAMES:
            lines.append(f"| {name} | {best['val_auroc'][name]:.3f} "
                        f"| {xgb_result['test_auroc'][name]:.3f} "
                        f"| {xgb_result['test_auprc'][name]:.3f} |")

    if deep_result and xgb_result:
        mean_deep = np.mean(list(deep_result["test_auroc"].values()))
        mean_xgb = np.mean(list(xgb_result["test_auroc"].values()))
        lines += ["\n## Tuned deep model vs. tuned XGBoost (test, survey-weighted AUROC)\n",
                 "| condition | deep (tuned) | XGBoost (tuned) |",
                 "|---|---:|---:|"]
        for name in LABEL_NAMES:
            lines.append(f"| {name} | {deep_result['test_auroc'][name]:.3f} "
                        f"| {xgb_result['test_auroc'][name]:.3f} |")
        lines.append(f"| **mean** | **{mean_deep:.3f}** | **{mean_xgb:.3f}** |")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nsaved -> {OUT_MD}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["deep", "xgboost", "both"], default="both")
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args(argv)

    deep_result = run_deep_search(args.n_trials) if args.model in ("deep", "both") else None
    xgb_result = run_xgb_search(args.n_trials) if args.model in ("xgboost", "both") else None

    write_report(deep_result, xgb_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
