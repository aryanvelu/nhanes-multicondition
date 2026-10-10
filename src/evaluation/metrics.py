"""Test-set evaluation of the per-head (middle-ground) model.

Two things happen here, both on the held-out **test** split (never seen in
training or model selection):

Part A — discrimination
    For the single best per-head configuration (chosen by mean validation
    AUROC), report per-condition **AUROC** and **AUPRC**, each computed two
    ways: **unweighted** (how well it ranks people in this sample) and
    **survey-weighted** (how well it would rank in the US adult population).
    The four general conditions use the exam weight ``WTMEC2YR``; metabolic
    syndrome uses the fasting-subsample weight ``WTSAF2YR``, because it is only
    defined on the fasting subsample.

    AUROC (area under the ROC curve) — probability the model scores a random
        positive above a random negative; 0.5 = chance, 1.0 = perfect.
    AUPRC (area under the precision-recall curve, a.k.a. average precision) —
        like AUROC but focused on the positive class; more informative when a
        condition is rare. Its chance level equals the positive prevalence.

Part B — the real test of survey weighting: subgroup calibration
    Calibration asks: when the model says "40% risk", do ~40% of such people
    actually have the condition? We check this *within demographic subgroups*
    (race/ethnicity, and income category), comparing, per subgroup, the
    (survey-weighted) mean predicted probability against the (survey-weighted)
    observed prevalence. The gap between them is the calibration error.

    We do this for two models — one trained WITH survey weighting, one WITHOUT —
    and ask which has *more consistent* calibration across subgroups (smaller
    and less variable gaps). Survey weighting is "worth it" precisely if the
    weighted model is better calibrated across subgroups, even though (as the
    ablation showed) it may look slightly worse on raw unweighted AUROC.

Outputs: ``results/test_metrics.md``.

Usage
-----
    python -m src.evaluation.metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.labels import LABEL_NAMES
from src.models.backbone import Backbone
from src.models.structured_layer import assemble_model_inputs
from src.training.finetune import MultiHead, build_head_masks, compute_logits

PROCESSED_ROOT = Path("data/processed")
FEATURES_PATH = PROCESSED_ROOT / "features.parquet"
SCHEMA_PATH = PROCESSED_ROOT / "feature_schema_perhead.json"
LABELS_PATH = PROCESSED_ROOT / "labels.parquet"
MERGED_PATH = PROCESSED_ROOT / "merged.parquet"
SPLITS_ROOT = Path("data/splits")
FINETUNE_ROOT = Path("experiments/finetune/perhead")
OUT_PATH = Path("results/test_metrics.md")

# Per-condition survey weight: exam weight for the four general heads; the
# fasting-subsample weight for metabolic syndrome (only defined there).
COND_WEIGHT = {
    "diabetes": "WTMEC2YR", "prediabetes": "WTMEC2YR", "obesity": "WTMEC2YR",
    "hypertension": "WTMEC2YR", "metabolic_syndrome": "WTSAF2YR",
}


# --------------------------------------------------------------------------- #
# Weighted discrimination metrics (dependency-free)
# --------------------------------------------------------------------------- #

def weighted_auroc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    """Weighted AUROC = weighted P(score_pos > score_neg), ties counted as 0.5."""
    Wpos = w[y == 1].sum(); Wneg = w[y == 0].sum()
    if Wpos == 0 or Wneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    auc = 0.0
    cum_neg = 0.0
    i, n = 0, len(s)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        grp_y, grp_w = y[i:j + 1], w[i:j + 1]
        w_pos = grp_w[grp_y == 1].sum()
        w_neg = grp_w[grp_y == 0].sum()
        auc += w_pos * (cum_neg + 0.5 * w_neg)   # negs ranked below + half ties
        cum_neg += w_neg
        i = j + 1
    return auc / (Wpos * Wneg)


def weighted_auprc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    """Weighted average precision (area under the PR curve)."""
    Wpos = w[y == 1].sum()
    if Wpos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, w = y[order], w[order]
    cum_tp = cum_fp = ap = prev_recall = 0.0
    for yi, wi in zip(y, w):
        if yi == 1:
            cum_tp += wi
        else:
            cum_fp += wi
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / Wpos
        if yi == 1:
            ap += (recall - prev_recall) * precision
            prev_recall = recall
    return ap


# --------------------------------------------------------------------------- #
# Data / model loading
# --------------------------------------------------------------------------- #

def load_test_arrays(schema):
    """Return dict of test-set arrays: features, labels, valid, weights, subgroups."""
    ids = pd.read_csv(SPLITS_ROOT / "test.csv")
    feat = ids.merge(pd.read_parquet(FEATURES_PATH), on=["SEQN", "cycle"],
                     how="left", validate="one_to_one")
    lab = ids.merge(pd.read_parquet(LABELS_PATH), on=["SEQN", "cycle"],
                    how="left", validate="one_to_one")
    raw = ids.merge(pd.read_parquet(MERGED_PATH, columns=["SEQN", "cycle", "RIDRETH1", "INDFMPIR"]),
                    on=["SEQN", "cycle"], how="left", validate="one_to_one")

    xc, xm, xk, _, _ = assemble_model_inputs(feat, schema)
    y = lab[LABEL_NAMES].to_numpy(dtype=np.float32)
    valid = lab[[f"{n}_valid" for n in LABEL_NAMES]].to_numpy(dtype=np.float32)

    weights = {c: np.nan_to_num(feat[c].to_numpy(dtype=np.float64), nan=0.0)
               for c in ("WTMEC2YR", "WTSAF2YR")}

    # Interpretable subgroups.
    reth_names = {1: "Mexican American", 2: "Other Hispanic", 3: "NH White",
                  4: "NH Black", 5: "Other/Multi"}
    reth = raw["RIDRETH1"].map(reth_names).fillna("Unknown").to_numpy()
    pir = raw["INDFMPIR"].to_numpy(dtype=float)
    income = np.where(np.isnan(pir), "Unknown",
                      np.where(pir < 1.3, "Low (<1.3)",
                               np.where(pir < 3.5, "Middle (1.3-3.5)", "High (>=3.5)")))

    return {"xc": xc, "xm": xm, "xk": xk, "y": y, "valid": valid,
            "weights": weights, "race": reth, "income": income}


def predict(model_path: Path, schema, device, data, head_masks) -> np.ndarray:
    """Load a saved model and return test probabilities [N, 5]."""
    backbone = Backbone(schema).to(device)
    heads = MultiHead(backbone.output_dim).to(device)
    ckpt = torch.load(model_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone"]); heads.load_state_dict(ckpt["heads"])
    backbone.eval(); heads.eval()
    xc, xm, xk = [torch.from_numpy(data[k]).to(device) for k in ("xc", "xm", "xk")]
    with torch.no_grad():
        logits = compute_logits(backbone, heads, xc, xm, xk, head_masks)
        return torch.sigmoid(logits).cpu().numpy()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def discrimination_table(data, probs) -> list[str]:
    lines = ["| condition | n valid | prev. | AUROC (unw.) | AUROC (wt.) | AUPRC (unw.) | AUPRC (wt.) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for k, name in enumerate(LABEL_NAMES):
        m = data["valid"][:, k] == 1
        w = data["weights"][COND_WEIGHT[name]]
        mw = m & (w > 0)
        y, s = data["y"][mw, k], probs[mw, k]
        wt = w[mw]
        prev = y.mean() if len(y) else float("nan")
        lines.append(
            f"| {name} | {int(mw.sum()):,} | {prev:.1%} "
            f"| {weighted_auroc(y, s, np.ones_like(wt)):.3f} "
            f"| {weighted_auroc(y, s, wt):.3f} "
            f"| {weighted_auprc(y, s, np.ones_like(wt)):.3f} "
            f"| {weighted_auprc(y, s, wt):.3f} |")
    return lines


def subgroup_calibration(data, probs, subgroup_key):
    """Return {condition: {subgroup: (weighted_pred, weighted_obs, gap, n)}}."""
    groups = data[subgroup_key]
    out = {}
    for k, name in enumerate(LABEL_NAMES):
        w = data["weights"][COND_WEIGHT[name]]
        valid = (data["valid"][:, k] == 1) & (w > 0)
        cond = {}
        for g in sorted(set(groups[valid])):
            m = valid & (groups == g)
            if m.sum() < 30:  # skip tiny/unstable cells
                continue
            ww = w[m]; Wsum = ww.sum()
            pred = float((ww * probs[m, k]).sum() / Wsum)
            obs = float((ww * data["y"][m, k]).sum() / Wsum)
            cond[g] = (pred, obs, abs(pred - obs), int(m.sum()))
        out[name] = cond
    return out


def consistency_summary(calib) -> tuple[float, float]:
    """Mean and max absolute calibration gap across all (condition, subgroup)."""
    gaps = [v[2] for cond in calib.values() for v in cond.values()]
    return (float(np.mean(gaps)), float(np.max(gaps))) if gaps else (float("nan"), float("nan"))


def main() -> int:
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    schema = json.loads(SCHEMA_PATH.read_text())
    head_masks = build_head_masks(schema, device)
    data = load_test_arrays(schema)

    # Best per-head config (by mean val AUROC) for Part A; its weighted sibling
    # for the Part B calibration comparison.
    def mean_val(d):
        a = json.loads((FINETUNE_ROOT / d / "finetune_config.json").read_text())["final_val_auroc"]
        return sum(a.values()) / len(a)
    variants = ["pretrain-on_weight-on", "pretrain-on_weight-off",
                "pretrain-off_weight-on", "pretrain-off_weight-off"]
    best = max(variants, key=mean_val)
    weighted_run = "pretrain-on_weight-on"     # trained WITH survey weighting
    unweighted_run = "pretrain-on_weight-off"  # trained WITHOUT

    probs_best = predict(FINETUNE_ROOT / best / "model.pt", schema, device, data, head_masks)
    probs_w = predict(FINETUNE_ROOT / weighted_run / "model.pt", schema, device, data, head_masks)
    probs_u = predict(FINETUNE_ROOT / unweighted_run / "model.pt", schema, device, data, head_masks)

    md = [f"# Test-set metrics — per-head (middle-ground) model\n",
          f"Held-out test split: **{len(data['y']):,} adults**. "
          f"Best config by mean val AUROC: **{best}**.\n",
          "## Part A — discrimination (best config)\n",
          "Per condition, unweighted vs survey-weighted AUROC/AUPRC. Weight: "
          "`WTMEC2YR` for the four general heads, `WTSAF2YR` for metabolic syndrome.\n"]
    md += discrimination_table(data, probs_best)

    # Part B
    md += ["\n## Part B — subgroup calibration: weighted vs unweighted training\n",
           "For each subgroup, survey-weighted **mean predicted probability** vs "
           "**observed prevalence**; the gap is the calibration error. Lower and "
           "more uniform gaps = better calibration. Two models compared: trained "
           f"**with** survey weighting (`{weighted_run}`) vs **without** "
           f"(`{unweighted_run}`).\n"]

    verdicts = []
    for key, title in [("race", "Race / ethnicity"), ("income", "Income category")]:
        calib_w = subgroup_calibration(data, probs_w, key)
        calib_u = subgroup_calibration(data, probs_u, key)
        mean_w, max_w = consistency_summary(calib_w)
        mean_u, max_u = consistency_summary(calib_u)
        md += [f"\n### {title}\n",
               "| model | mean |gap| across subgroups | max |gap| |",
               "|---|---:|---:|",
               f"| trained WITH survey weight | {mean_w:.3f} | {max_w:.3f} |",
               f"| trained WITHOUT survey weight | {mean_u:.3f} | {max_u:.3f} |"]
        better = "WITH" if mean_w < mean_u else "WITHOUT"
        verdicts.append((title, better, mean_w, mean_u, max_w, max_u))

        # Detailed table for metabolic_syndrome as a worked example.
        md += [f"\nDetail — metabolic syndrome by {title.lower()} "
               "(pred / observed / gap), weighted model | unweighted model:\n",
               "| subgroup | n | W pred | W obs | W gap | U pred | U obs | U gap |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
        cw, cu = calib_w["metabolic_syndrome"], calib_u["metabolic_syndrome"]
        for g in sorted(set(cw) | set(cu)):
            pw, ow, gw, nw = cw.get(g, (float("nan"),) * 4)
            pu, ou, gu, _ = cu.get(g, (float("nan"),) * 4)
            md.append(f"| {g} | {nw} | {pw:.3f} | {ow:.3f} | {gw:.3f} "
                      f"| {pu:.3f} | {ou:.3f} | {gu:.3f} |")

    md += ["\n## Verdict — was survey weighting worth it?\n"]
    for title, better, mw, mu, xw, xu in verdicts:
        md.append(f"- **{title}:** more consistent calibration when trained "
                  f"**{better}** survey weighting "
                  f"(mean |gap| {mw:.3f} weighted vs {mu:.3f} unweighted).")
    overall = sum(1 for _, b, *_ in verdicts if b == "WITH")
    md.append(f"\n{overall} of {len(verdicts)} subgroup analyses favor survey-weighted "
              "training on calibration consistency — the metric that actually reflects "
              "population fairness, unlike raw unweighted AUROC.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nsaved -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
