"""Stacking ensemble: blend the FT-Transformer, XGBoost, and LightGBM with a
per-condition logistic-regression meta-model.

Each base model already outputs a predicted probability per person per
condition. The meta-model for a condition is a 3-input logistic regression:
`P(condition) = sigmoid(a*ft_prob + b*xgb_prob + c*lgbm_prob + d)`. It is fit
on the **validation split's** predictions from all three base models — never
on the training split any base model actually learned its weights from —
then scored on the untouched test split with the same survey-weighted
AUROC/AUPRC used everywhere else in this project.

All three base models are trained on the current, fully-enriched per-head
feature set (3 cycles + lifestyle/family_history/weight_history/renal/
hepatic/medication_use, with the WHD020/WHD050/RXDCOUNT proxy-leak masks
applied).

Two 4th-base-learner candidates were tried and rejected, kept here as a
record rather than repeated blind re-tries:
* grouped-MLP retrained on the current schema (`train_mlp_final.py`) —
  individually the weakest of the four models on every condition; folding
  it in left the stack mean flat-to-negative (0.842 -> 0.841). See
  memory `grouped-mlp-4th-stack-negative`.
* FT-Transformer with masked-token pretraining redone on the current
  schema (`pretrain_ft.py`) — a 10-seed sweep (`ft_pretrain_sweep.py`)
  found it net *worse* than the random-init FT-Transformer (mean val
  0.842 vs 0.844, mean test 0.835 vs 0.837), with real (noise-floor-
  clearing) losses on hypertension and metabolic_syndrome outweighing a
  real gain on diabetes alone. Not adopted.

Caveat worth stating plainly: all three base models used the validation
split for *model selection* (FT-Transformer/XGBoost/LightGBM's respective
early-stopping), just not for fitting weights. That is the standard
"held-out blending" setup for stacking, not a fully nested cross-validation —
good enough to answer "does blending help," not rigorous enough to claim a
bias-free performance estimate at the third decimal place.

Usage
-----
    python -m src.evaluation.stacking
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.baselines.xgboost_baseline import build_head_frame
from src.data.labels import LABEL_NAMES
from src.evaluation.metrics import COND_WEIGHT, weighted_auprc, weighted_auroc
from src.models.ft_transformer import FTTransformerBackbone
from src.models.structured_layer import assemble_model_inputs
from src.training.finetune import MultiHead, build_head_masks, compute_logits

PROCESSED_ROOT = Path("data/processed")
FEATURES_PATH = PROCESSED_ROOT / "features.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_perhead.json"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"
SPLITS_ROOT = Path("data/splits")
FT_MODEL_PATH = Path("experiments/finetune/ft_transformer/model.pt")
XGB_DIR = Path("experiments/xgboost")
LGBM_DIR = Path("experiments/lightgbm")
OUT_PATH = Path("results/stacking.md")

BASE_MODELS = ["ft", "xgb", "lgbm"]


def load_split_arrays(name: str, schema: dict) -> dict:
    ids = pd.read_csv(SPLITS_ROOT / f"{name}.csv")
    feat = ids.merge(pd.read_parquet(FEATURES_PATH), on=["SEQN", "cycle"],
                     how="left", validate="one_to_one")
    lab = ids.merge(pd.read_parquet(LABELS_PATH), on=["SEQN", "cycle"],
                    how="left", validate="one_to_one")
    xc, xm, xk, _, _ = assemble_model_inputs(feat, schema)
    y = lab[LABEL_NAMES].to_numpy(dtype=np.float32)
    valid = lab[[f"{n}_valid" for n in LABEL_NAMES]].to_numpy(dtype=np.float32)
    weights = {c: np.nan_to_num(feat[c].to_numpy(dtype=np.float64), nan=0.0)
              for c in ("WTMEC2YR", "WTSAF2YR")}
    return {"feat": feat, "xc": xc, "xm": xm, "xk": xk, "y": y,
            "valid": valid, "weights": weights}


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


def xgb_predict_all(schema: dict, feat: pd.DataFrame) -> dict[str, np.ndarray]:
    """{label: prob array} from the saved per-label XGBoost models."""
    head_masks = schema["head_masks"]
    probs = {}
    for name in LABEL_NAMES:
        model = XGBClassifier()
        model.load_model(XGB_DIR / f"{name}.json")
        X = build_head_frame(feat, schema, head_masks[name])
        probs[name] = model.predict_proba(X)[:, 1]
    return probs


def lgbm_predict_all(schema: dict, feat: pd.DataFrame) -> dict[str, np.ndarray]:
    """{label: prob array} from the saved per-label LightGBM models."""
    head_masks = schema["head_masks"]
    probs = {}
    for name in LABEL_NAMES:
        model = lgb.Booster(model_file=str(LGBM_DIR / f"{name}.txt"))
        X = build_head_frame(feat, schema, head_masks[name])
        probs[name] = model.predict(X)  # binary objective -> P(positive) directly
    return probs


def main() -> int:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    schema = json.loads(SCHEMA_PATH.read_text())
    head_masks_t = build_head_masks(schema, device)

    val = load_split_arrays("val", schema)
    test = load_split_arrays("test", schema)

    ft_val = predict_ft(FT_MODEL_PATH, schema, device, val, head_masks_t)
    ft_test = predict_ft(FT_MODEL_PATH, schema, device, test, head_masks_t)
    xgb_val = xgb_predict_all(schema, val["feat"])
    xgb_test = xgb_predict_all(schema, test["feat"])
    lgbm_val = lgbm_predict_all(schema, val["feat"])
    lgbm_test = lgbm_predict_all(schema, test["feat"])

    rows = []
    for k, name in enumerate(LABEL_NAMES):
        v_valid = val["valid"][:, k] == 1
        w_col = COND_WEIGHT[name]
        t_valid = test["valid"][:, k] == 1
        t_w = test["weights"][w_col]
        t_mask = t_valid & (t_w > 0)

        # Fit the meta-model on validation-split predictions only.
        Xv = np.column_stack([ft_val[v_valid, k], xgb_val[name][v_valid], lgbm_val[name][v_valid]])
        yv = val["y"][v_valid, k]
        meta = LogisticRegression(max_iter=1000)
        meta.fit(Xv, yv)

        # Score all four (each base model alone, plus stacked) on the
        # identical test subset so the comparison is apples-to-apples.
        yt = test["y"][t_mask, k]
        wt = t_w[t_mask]
        f_test = ft_test[t_mask, k]
        x_test = xgb_test[name][t_mask]
        l_test = lgbm_test[name][t_mask]
        Xt = np.column_stack([f_test, x_test, l_test])
        s_test = meta.predict_proba(Xt)[:, 1]

        row = {
            "name": name, "n": int(t_mask.sum()), "prev": float(yt.mean()),
            "ft_auroc": weighted_auroc(yt, f_test, wt),
            "xgb_auroc": weighted_auroc(yt, x_test, wt),
            "lgbm_auroc": weighted_auroc(yt, l_test, wt),
            "stack_auroc": weighted_auroc(yt, s_test, wt),
            "ft_auprc": weighted_auprc(yt, f_test, wt),
            "xgb_auprc": weighted_auprc(yt, x_test, wt),
            "lgbm_auprc": weighted_auprc(yt, l_test, wt),
            "stack_auprc": weighted_auprc(yt, s_test, wt),
            "coef_ft": float(meta.coef_[0, 0]),
            "coef_xgb": float(meta.coef_[0, 1]),
            "coef_lgbm": float(meta.coef_[0, 2]),
            "intercept": float(meta.intercept_[0]),
        }
        rows.append(row)
        print(f"{name:<20} ft={row['ft_auroc']:.3f} xgb={row['xgb_auroc']:.3f} "
              f"lgbm={row['lgbm_auroc']:.3f} stack={row['stack_auroc']:.3f}  "
              f"(coef ft={row['coef_ft']:.2f} xgb={row['coef_xgb']:.2f} "
              f"lgbm={row['coef_lgbm']:.2f})")

    write_report(rows)
    return 0


def write_report(rows: list[dict]) -> None:
    lines = [
        "# Stacking ensemble — logistic-regression blend of FT-Transformer + "
        "XGBoost + LightGBM\n",
        "Per-condition logistic regression on `[ft_prob, xgb_prob, lgbm_prob]`, "
        "fit on **validation-split** predictions from all three base models "
        "(never on data any base model was trained on), scored on the "
        "held-out **test** split with the same survey-weighted AUROC/AUPRC "
        "used throughout this project. All three base models trained on the "
        "current, fully-enriched feature set. Two 4th-base-learner candidates "
        "(grouped-MLP, pretrained FT-Transformer) were tried and rejected — "
        "see module docstring.\n",
        "Caveat: all three base models used the validation split for "
        "early-stopping model selection, just not for fitting weights — "
        "standard practice for blending, not a fully nested cross-validation.\n",
        "## Survey-weighted test AUROC\n",
        "| condition | n | prevalence | FT-Transformer | XGBoost | LightGBM | **stacked** | best |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    n_stack_wins_auroc = 0
    for r in rows:
        scores = [("ft", r["ft_auroc"]), ("xgb", r["xgb_auroc"]),
                 ("lgbm", r["lgbm_auroc"]), ("stack", r["stack_auroc"])]
        best = max(scores, key=lambda t: t[1])[0]
        if best == "stack":
            n_stack_wins_auroc += 1
        lines.append(f"| {r['name']} | {r['n']:,} | {r['prev']:.1%} "
                     f"| {r['ft_auroc']:.3f} | {r['xgb_auroc']:.3f} "
                     f"| {r['lgbm_auroc']:.3f} | **{r['stack_auroc']:.3f}** | {best} |")
    mean_ft = np.mean([r["ft_auroc"] for r in rows])
    mean_xgb = np.mean([r["xgb_auroc"] for r in rows])
    mean_lgbm = np.mean([r["lgbm_auroc"] for r in rows])
    mean_stack = np.mean([r["stack_auroc"] for r in rows])
    lines.append(f"| **mean** | | | **{mean_ft:.3f}** | **{mean_xgb:.3f}** "
                f"| **{mean_lgbm:.3f}** | **{mean_stack:.3f}** | |")

    lines += ["\n## Survey-weighted test AUPRC\n",
             "| condition | FT-Transformer | XGBoost | LightGBM | **stacked** | best |",
             "|---|---:|---:|---:|---:|:---:|"]
    n_stack_wins_auprc = 0
    for r in rows:
        scores = [("ft", r["ft_auprc"]), ("xgb", r["xgb_auprc"]),
                 ("lgbm", r["lgbm_auprc"]), ("stack", r["stack_auprc"])]
        best = max(scores, key=lambda t: t[1])[0]
        if best == "stack":
            n_stack_wins_auprc += 1
        lines.append(f"| {r['name']} | {r['ft_auprc']:.3f} | {r['xgb_auprc']:.3f} "
                     f"| {r['lgbm_auprc']:.3f} | **{r['stack_auprc']:.3f}** | {best} |")

    lines += ["\n## Meta-model coefficients (what does the blend lean on?)\n",
             "`P = sigmoid(coef_ft*ft_prob + coef_xgb*xgb_prob + "
             "coef_lgbm*lgbm_prob + intercept)` — all coefficients positive "
             "and none dominating means the blend is genuinely using more "
             "than one model, not just reproducing the best single one.\n",
             "| condition | coef (FT) | coef (xgb) | coef (lgbm) | intercept |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['coef_ft']:.2f} | {r['coef_xgb']:.2f} "
                     f"| {r['coef_lgbm']:.2f} | {r['intercept']:.2f} |")

    lines += ["\n## Verdict\n",
             f"Stacking wins on AUROC in **{n_stack_wins_auroc}/{len(rows)}** conditions "
             f"(mean AUROC {mean_stack:.3f} vs FT-Transformer {mean_ft:.3f}, "
             f"XGBoost {mean_xgb:.3f}, LightGBM {mean_lgbm:.3f}) "
             f"and on AUPRC in **{n_stack_wins_auprc}/{len(rows)}**."]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
