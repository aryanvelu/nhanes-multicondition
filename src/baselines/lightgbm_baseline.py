"""Second classical baseline: one LightGBM classifier per condition.

Identical setup to `xgboost_baseline.py` — same per-head leakage-masked
features (via the shared `build_head_frame`), same PSU-cluster splits, same
label validity masks, same survey-weighted AUROC/AUPRC functions — so this is
a like-for-like second opinion on "does a different gradient-boosting
implementation change the classical-model picture," not a new experimental
design. Both trained without survey-weighted loss, matching the deep model's
best-AUROC config (the same convention `xgboost_baseline.py` uses).

Output: `experiments/lightgbm/<condition>.txt` (model),
`experiments/lightgbm/test_results.json`, and `results/lightgbm_baseline.md`
— a three-way table (deep / XGBoost / LightGBM) per condition, reading the
other two models' numbers from their own already-written result files so
there's no risk of transcription drift.

Usage
-----
    python -m src.baselines.lightgbm_baseline
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.baselines.xgboost_baseline import build_head_frame
from src.data.labels import LABEL_NAMES
from src.evaluation.metrics import COND_WEIGHT, weighted_auprc, weighted_auroc

PROCESSED_ROOT = Path("data/processed")
FEATURES_PATH = PROCESSED_ROOT / "features.parquet"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_perhead.json"
SPLITS_ROOT = Path("data/splits")
OUT_DIR = Path("experiments/lightgbm")
XGB_RESULTS_PATH = Path("experiments/xgboost/test_results.json")
DEEP_METRICS_PATH = Path("results/test_metrics.md")
OUT_MD = Path("results/lightgbm_baseline.md")


def load_split(name: str, features: pd.DataFrame, labels: pd.DataFrame):
    ids = pd.read_csv(SPLITS_ROOT / f"{name}.csv")
    feat = ids.merge(features, on=["SEQN", "cycle"], how="left", validate="one_to_one")
    lab = ids.merge(labels, on=["SEQN", "cycle"], how="left", validate="one_to_one")
    return feat, lab


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

        model = lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
        )
        model.fit(
            Xtr[vtr], ytr,
            eval_set=[(Xva[vva], yva)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                      lgb.log_evaluation(period=0)],
        )
        model.booster_.save_model(str(OUT_DIR / f"{name}.txt"))

        w_col = COND_WEIGHT[name]
        w = test_feat.loc[vte, w_col].to_numpy(dtype=np.float64)
        y = test_lab[name].to_numpy()[vte]
        mw = w > 0
        y, w = y[mw], w[mw]
        Xte_valid = Xte[vte].reset_index(drop=True)
        probs = model.predict_proba(Xte_valid[mw])[:, 1]

        best_iter = getattr(model, "best_iteration_", None)
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
    text = path.read_text()
    rows = re.findall(
        r"^\| (\w+) \| ([\d,]+) \| ([\d.]+)% \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$",
        text, flags=re.MULTILINE,
    )
    out = {}
    for name, n, prev, auroc_u, auroc_w, auprc_u, auprc_w in rows:
        if name in LABEL_NAMES:
            out[name] = {
                "auroc_weighted": float(auroc_w), "auprc_weighted": float(auprc_w),
            }
    return out


def write_report(lgbm_results: dict) -> None:
    deep = parse_deep_model_table(DEEP_METRICS_PATH)
    xgb = json.loads(XGB_RESULTS_PATH.read_text())

    lines = [
        "# LightGBM baseline — three-way comparison (deep / XGBoost / LightGBM)\n",
        "Same per-head leakage-masked features, splits, and survey-weighted "
        "AUROC/AUPRC as the deep model and the XGBoost baseline. All three "
        "trained without survey-weighted loss (the deep model's best-AUROC "
        "config), so this is architecture vs. architecture, not weighting "
        "scheme vs. weighting scheme.\n",
        "## Survey-weighted test AUROC\n",
        "| condition | n (test) | prevalence | deep | XGBoost | LightGBM | best |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for name in LABEL_NAMES:
        l, x, d = lgbm_results[name], xgb[name], deep.get(name)
        if d is None:
            continue
        scores = [("deep", d["auroc_weighted"]), ("xgb", x["auroc_weighted"]),
                 ("lgbm", l["auroc_weighted"])]
        best = max(scores, key=lambda t: t[1])[0]
        lines.append(f"| {name} | {l['n_test']:,} | {l['prevalence']:.1%} "
                     f"| {d['auroc_weighted']:.3f} | {x['auroc_weighted']:.3f} "
                     f"| {l['auroc_weighted']:.3f} | {best} |")
    mean_deep = np.mean([deep[n]["auroc_weighted"] for n in LABEL_NAMES if n in deep])
    mean_xgb = np.mean([xgb[n]["auroc_weighted"] for n in LABEL_NAMES])
    mean_lgbm = np.mean([lgbm_results[n]["auroc_weighted"] for n in LABEL_NAMES])
    lines.append(f"| **mean** | | | **{mean_deep:.3f}** | **{mean_xgb:.3f}** "
                f"| **{mean_lgbm:.3f}** | |")

    lines += ["\n## Survey-weighted test AUPRC\n",
             "| condition | deep | XGBoost | LightGBM | best |",
             "|---|---:|---:|---:|:---:|"]
    for name in LABEL_NAMES:
        l, x, d = lgbm_results[name], xgb[name], deep.get(name)
        if d is None:
            continue
        scores = [("deep", d["auprc_weighted"]), ("xgb", x["auprc_weighted"]),
                 ("lgbm", l["auprc_weighted"])]
        best = max(scores, key=lambda t: t[1])[0]
        lines.append(f"| {name} | {d['auprc_weighted']:.3f} | {x['auprc_weighted']:.3f} "
                     f"| {l['auprc_weighted']:.3f} | {best} |")

    n_wins = {"deep": 0, "xgb": 0, "lgbm": 0}
    for name in LABEL_NAMES:
        if name not in deep:
            continue
        scores = [("deep", deep[name]["auroc_weighted"]), ("xgb", xgb[name]["auroc_weighted"]),
                 ("lgbm", lgbm_results[name]["auroc_weighted"])]
        n_wins[max(scores, key=lambda t: t[1])[0]] += 1
    lines += ["\n## Verdict\n",
             f"AUROC wins by condition: deep {n_wins['deep']}/5, XGBoost {n_wins['xgb']}/5, "
             f"LightGBM {n_wins['lgbm']}/5 (mean: deep {mean_deep:.3f}, XGBoost "
             f"{mean_xgb:.3f}, LightGBM {mean_lgbm:.3f})."]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
