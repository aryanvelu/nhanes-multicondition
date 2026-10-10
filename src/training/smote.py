"""SMOTE oversampling for specific condition heads — training split only.

SMOTE (Synthetic Minority Over-sampling TEchnique) manufactures new minority-
class rows by interpolating between real minority-class neighbors in feature
space, rather than just duplicating existing rows. Applied here to
**prediabetes** and **metabolic syndrome** specifically, since those are the
two heads named in the request — not diabetes/obesity/hypertension.

This never touches validation or test data (leaking synthetic *or* real
information about those splits would invalidate the whole evaluation) — it
only augments the tensors already built from the training split, after the
leakage-safe/per-head feature representation exists (i.e. it runs on the same
``(x_cont, x_miss, x_cat, y, valid, weight)`` tensors ``finetune.py`` already
trains on, downstream of every leakage decision).

Because this project trains all 5 heads jointly on one shared backbone, a
synthetic row can't just carry one label — every batch needs values for all 5
heads. The fix: give a synthetic row a real (interpolated) label only for the
condition it was generated for, and mark **every other head's `_valid` flag
0** for that row, so the joint loss simply ignores it for the other 4
conditions (exactly the same mechanism that already lets real participants
skip labels they don't have data for). The synthetic row's survey weight is
set to 1.0 (the population-average weight after normalization), since a
synthetic person doesn't have a real one.

Mixed continuous + categorical features need ``SMOTENC`` (not plain SMOTE):
plain SMOTE would linearly interpolate categorical codes and missing-flags
into meaningless fractional values (e.g. "1.4" for a 3-category variable).
SMOTENC instead mode-selects categorical/flag columns from the nearest real
neighbors, so synthetic rows always get valid category codes and clean 0/1
missing flags.

Usage
-----
    from src.training.smote import augment_with_smote
    train_t = augment_with_smote(train_t, label_idx, seed=seed)
"""

from __future__ import annotations

import numpy as np
import torch
from imblearn.over_sampling import SMOTENC


def augment_with_smote(train_tensors: tuple[torch.Tensor, ...], label_idx: int,
                       seed: int, min_minority: int = 6) -> tuple[torch.Tensor, ...]:
    """Return new (x_cont, x_miss, x_cat, y, valid, weight) tensors with
    SMOTE-synthesized minority-class rows appended for one label's head."""
    x_cont, x_miss, x_cat, y, valid, weight = [t.numpy() for t in train_tensors]
    n_cont, n_miss, n_cat = x_cont.shape[1], x_miss.shape[1], x_cat.shape[1]

    mask = valid[:, label_idx] == 1
    X = np.concatenate([x_cont[mask], x_miss[mask], x_cat[mask]], axis=1)
    y_label = y[mask, label_idx]

    n_pos, n_neg = int(y_label.sum()), int((1 - y_label).sum())
    if min(n_pos, n_neg) < min_minority:
        raise ValueError(f"too few minority-class rows to SMOTE "
                         f"(pos={n_pos}, neg={n_neg}, need >= {min_minority})")

    # Missing-flag + categorical-code columns are discrete -> mode-selected,
    # not interpolated.
    categorical_idx = list(range(n_cont, n_cont + n_miss + n_cat))
    smote = SMOTENC(categorical_features=categorical_idx, random_state=seed)
    X_res, y_res = smote.fit_resample(X, y_label)

    # SMOTENC (oversampling only) preserves the original rows first, then
    # appends synthetic ones — only the appended tail is new.
    n_orig = X.shape[0]
    n_synth = X_res.shape[0] - n_orig
    if n_synth <= 0:
        return train_tensors  # already balanced (e.g. re-run at parity)

    X_synth, y_synth = X_res[n_orig:], y_res[n_orig:]
    xc_s = X_synth[:, :n_cont].astype(np.float32)
    xm_s = X_synth[:, n_cont:n_cont + n_miss].astype(np.float32)
    xk_s = X_synth[:, n_cont + n_miss:].astype(np.int64)

    y_full = np.zeros((n_synth, y.shape[1]), dtype=np.float32)
    valid_full = np.zeros((n_synth, valid.shape[1]), dtype=np.float32)
    y_full[:, label_idx] = y_synth
    valid_full[:, label_idx] = 1.0
    weight_full = np.ones(n_synth, dtype=np.float32)

    return (
        torch.from_numpy(np.concatenate([x_cont, xc_s], axis=0)),
        torch.from_numpy(np.concatenate([x_miss, xm_s], axis=0)),
        torch.from_numpy(np.concatenate([x_cat, xk_s], axis=0)),
        torch.from_numpy(np.concatenate([y, y_full], axis=0)),
        torch.from_numpy(np.concatenate([valid, valid_full], axis=0)),
        torch.from_numpy(np.concatenate([weight, weight_full], axis=0)),
    )
