"""Classical baseline: one XGBoost classifier per condition (5 independent models).

This exists to answer a question the deep model's own ablation can't: is the
physiology-structured backbone + SSL pretraining actually earning its
complexity, or would a much simpler, off-the-shelf tree ensemble do just as
well? Gradient-boosted trees are the standard strong baseline for tabular
health data, so this is the comparison a reviewer would ask for first.

To keep the comparison fair, every design choice mirrors the deep model:

* **Same features.** Each label's XGBoost model sees the full repopulated
  feature set (continuous value + "was this missing" flag, categorical codes)
  from ``features.parquet``, exactly as the neural backbone does.
* **Same per-head leakage masking.** Before training/predicting a label's
  model, that label's own defining features are zeroed and their missing-flag
  set to 1 (the ``head_masks`` map in ``feature_schema_perhead.json``) — the
  identical mask ``finetune.py --per-head`` applies inside the neural
  network's forward pass, just applied once to a static dataframe instead of
  per batch.
* **Same splits.** Train/val/test come from ``data/splits/`` (the PSU-cluster
  splits), val is used for early stopping, test is untouched until scoring.
* **Same label validity masks.** Rows where a label isn't computable
  (``{label}_valid == 0``) are excluded from that label's training and scoring,
  same as the neural heads.
* **Same evaluation metrics.** ``src.evaluation.metrics.weighted_auroc`` /
  ``weighted_auprc`` (the exact functions used for the deep model's test-set
  report), with the exam weight ``WTMEC2YR`` for the four general conditions
  and the fasting-subsample weight ``WTSAF2YR`` for metabolic syndrome.
* **Same training regime as the deep model's reported config.** The per-head
  ablation's best/reported config trains *without* survey weighting
  (``pretrain-on_weight-off``), so these XGBoost models are also fit
  unweighted — an apples-to-apples comparison of architectures, not of
  weighting schemes (that question is already answered in ``test_metrics.md``).

Output: ``experiments/xgboost/<condition>.json`` (model),
``experiments/xgboost/test_results.json`` (raw metrics), and
``results/xgboost_baseline.md`` (the side-by-side table against the deep
model, whose numbers are read directly from the existing ``test_metrics.md``
so there is no risk of transcription drift).

Usage
-----
    python -m src.baselines.xgboost_baseline
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data.labels import LABEL_NAMES
from src.evaluation.metrics import COND_WEIGHT, weighted_auprc, weighted_auroc

PROCESSED_ROOT = Path("data/processed")
FEATURES_PATH = PROCESSED_ROOT / "features.parquet"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_perhead.json"
SPLITS_ROOT = Path("data/splits")
OUT_DIR = Path("experiments/xgboost")
DEEP_METRICS_PATH = Path("results/test_metrics.md")
OUT_MD = Path("results/xgboost_baseline.md")


def load_split(name: str, features: pd.DataFrame, labels: pd.DataFrame):
    ids = pd.read_csv(SPLITS_ROOT / f"{name}.csv")
    feat = ids.merge(features, on=["SEQN", "cycle"], how="left", validate="one_to_one")
    lab = ids.merge(labels, on=["SEQN", "cycle"], how="left", validate="one_to_one")
    return feat, lab


def build_head_frame(feat: pd.DataFrame, schema: dict, masked_features: list[str]) -> pd.DataFrame:
    """Value + missing-flag columns for every continuous feature, categorical
    codes for every categorical feature, with ``masked_features`` hidden
    (value -> 0, flag -> 1) exactly like the neural per-head mask."""
    cont_cols = list(schema["continuous"].keys())
    cat_cols = list(schema["categorical"].keys())
    df = feat[cont_cols + cat_cols].copy()
    # A feature with zero missingness never got a `_missing` column from
    # preprocess.py; synthesize an all-zero one so every continuous feature
    # has an aligned flag, same as assemble_model_inputs does for the net.
    for c in cont_cols:
        flag = f"{c}_missing"
        df[flag] = feat[flag].to_numpy() if flag in feat.columns else 0.0
    for f in masked_features:
        df[f] = 0.0
        df[f"{f}_missing"] = 1.0
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df


def main() -> int:
    schema = json.loads(SCHEMA_PATH.read_text())
    head_masks = schema["head_masks"]
    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)

    train_feat, train_lab = load_split("train", features, labels)
    val_feat, val_lab = load_split("val", features, labels)
    test_feat, test_lab = load_split("test", features, labels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    print(f"{'condition':<20}{'n train':>9}{'n val':>7}{'n test':>8}{'best_iter':>10}"
          f"{'AUROC(u)':>10}{'AUROC(w)':>10}{'AUPRC(u)':>10}{'AUPRC(w)':>10}")

    for name in LABEL_NAMES:
        masked = head_masks[name]
        Xtr = build_head_frame(train_feat, schema, masked)
        Xva = build_head_frame(val_feat, schema, masked)
        Xte = build_head_frame(test_feat, schema, masked)

        vtr = (train_lab[f"{name}_valid"].to_numpy() == 1)
        vva = (val_lab[f"{name}_valid"].to_numpy() == 1)
        vte = (test_lab[f"{name}_valid"].to_numpy() == 1)

        ytr = train_lab[name].to_numpy()[vtr]
        yva = val_lab[name].to_numpy()[vva]

        model = XGBClassifier(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            enable_categorical=True,
            tree_method="hist",
            eval_metric="auc",
            early_stopping_rounds=30,
            random_state=42,
        )
        model.fit(Xtr[vtr], ytr, eval_set=[(Xva[vva], yva)], verbose=False)
        model.save_model(OUT_DIR / f"{name}.json")

        w_col = COND_WEIGHT[name]
        w = test_feat.loc[vte, w_col].to_numpy(dtype=np.float64)
        y = test_lab[name].to_numpy()[vte]
        mw = w > 0  # drop any zero-weight rows, same guard as metrics.py
        y, w = y[mw], w[mw]
        Xte_valid = Xte[vte].reset_index(drop=True)
        probs = model.predict_proba(Xte_valid[mw])[:, 1]

        best_iter = getattr(model, "best_iteration", None)
        results[name] = {
            "n_train": int(vtr.sum()), "n_val": int(vva.sum()), "n_test": int(mw.sum()),
            "best_iteration": int(best_iter) if best_iter is not None else None,
            "prevalence": float(y.mean()),
            "auroc_unweighted": weighted_auroc(y, probs, np.ones_like(w)),
            "auroc_weighted": weighted_auroc(y, probs, w),
            "auprc_unweighted": weighted_auprc(y, probs, np.ones_like(w)),
            "auprc_weighted": weighted_auprc(y, probs, w),
        }
        r = results[name]
        print(f"{name:<20}{r['n_train']:>9,}{r['n_val']:>7,}{r['n_test']:>8,}"
              f"{r['best_iteration']:>10}{r['auroc_unweighted']:>10.3f}"
              f"{r['auroc_weighted']:>10.3f}{r['auprc_unweighted']:>10.3f}"
              f"{r['auprc_weighted']:>10.3f}")

    (OUT_DIR / "test_results.json").write_text(json.dumps(results, indent=2))
    write_report(results)
    print(f"\nsaved -> {OUT_DIR}/test_results.json, {OUT_MD}")
    return 0


def parse_deep_model_table(path: Path) -> dict[str, dict[str, float]]:
    """Pull the Part A per-condition AUROC/AUPRC numbers straight out of the
    already-written test_metrics.md, so this report can never drift out of
    sync with the deep model's own recorded results."""
    text = path.read_text()
    rows = re.findall(
        r"^\| (\w+) \| ([\d,]+) \| ([\d.]+)% \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$",
        text, flags=re.MULTILINE,
    )
    out = {}
    for name, n, prev, auroc_u, auroc_w, auprc_u, auprc_w in rows:
        if name in LABEL_NAMES:
            out[name] = {
                "n": int(n.replace(",", "")), "prevalence": float(prev) / 100,
                "auroc_unweighted": float(auroc_u), "auroc_weighted": float(auroc_w),
                "auprc_unweighted": float(auprc_u), "auprc_weighted": float(auprc_w),
            }
    return out


def write_report(xgb_results: dict) -> None:
    deep = parse_deep_model_table(DEEP_METRICS_PATH)
    lines = [
        "# XGBoost baseline vs. the deep per-head model — test-set comparison\n",
        "One independent `XGBClassifier` per condition, trained on the same "
        "per-head leakage-masked features and the same PSU-cluster splits as "
        "the neural model, scored with the identical survey-weighted "
        "AUROC/AUPRC functions used in `test_metrics.md`. Both models trained "
        "**without** survey-weighted loss (the deep model's best-AUROC "
        "config), so this isolates architecture, not weighting scheme.\n",
        "## Survey-weighted AUROC (primary comparison)\n",
        "| condition | n (test) | prevalence | XGBoost | Deep model | delta (deep − xgb) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in LABEL_NAMES:
        x, d = xgb_results[name], deep.get(name)
        if d is None:
            continue
        delta = d["auroc_weighted"] - x["auroc_weighted"]
        lines.append(f"| {name} | {x['n_test']:,} | {x['prevalence']:.1%} "
                     f"| {x['auroc_weighted']:.3f} | {d['auroc_weighted']:.3f} "
                     f"| {delta:+.3f} |")
    mean_x = sum(xgb_results[n]["auroc_weighted"] for n in LABEL_NAMES) / len(LABEL_NAMES)
    mean_d = sum(deep[n]["auroc_weighted"] for n in LABEL_NAMES if n in deep) / len(deep)
    lines.append(f"| **mean** | | | **{mean_x:.3f}** | **{mean_d:.3f}** "
                f"| **{mean_d - mean_x:+.3f}** |")

    lines += ["\n## Survey-weighted AUPRC\n",
             "| condition | XGBoost | Deep model | delta (deep − xgb) |",
             "|---|---:|---:|---:|"]
    for name in LABEL_NAMES:
        x, d = xgb_results[name], deep.get(name)
        if d is None:
            continue
        delta = d["auprc_weighted"] - x["auprc_weighted"]
        lines.append(f"| {name} | {x['auprc_weighted']:.3f} | {d['auprc_weighted']:.3f} "
                     f"| {delta:+.3f} |")

    lines += ["\n## Unweighted AUROC/AUPRC (full detail)\n",
             "| condition | XGBoost AUROC(u) | Deep AUROC(u) | XGBoost AUPRC(u) | Deep AUPRC(u) |",
             "|---|---:|---:|---:|---:|"]
    for name in LABEL_NAMES:
        x, d = xgb_results[name], deep.get(name)
        if d is None:
            continue
        lines.append(f"| {name} | {x['auroc_unweighted']:.3f} | {d['auroc_unweighted']:.3f} "
                     f"| {x['auprc_unweighted']:.3f} | {d['auprc_unweighted']:.3f} |")

    n_deep_wins = sum(1 for n in LABEL_NAMES if n in deep
                      and deep[n]["auroc_weighted"] > xgb_results[n]["auroc_weighted"])
    lines += ["\n## Verdict\n",
             f"The deep model beats the XGBoost baseline on survey-weighted AUROC in "
             f"**{n_deep_wins} of {len(LABEL_NAMES)}** conditions "
             f"(mean AUROC {mean_d:.3f} vs {mean_x:.3f})."]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
