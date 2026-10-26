"""
weights.py - class weights for the imbalanced fine-tuning tasks (numpy only).

Only ~12-14 % of the (window, car) pairs pit within 5 laps, so an unweighted loss is happy with probabilities that stay below
0.5 for most real stops (precision high, recall tiny at the threshold 0.5).  ``pos_weight`` multiplies the loss of the positive
class in the binary cross-entropy.

* ``--pit-pos-weight 1``      no weighting (default, the runs so far)
* ``--pit-pos-weight sqrt``   sqrt((1 - p) / p): about 2.5 for p = 14 %, a middle way
* ``--pit-pos-weight auto``   (1 - p) / p: about 6 for p = 14 %, positives and negatives weigh the same in total
* ``--pit-pos-weight 3.0``    any number

Effects to expect: recall at the threshold 0.5 rises, precision falls, the probabilities are no longer calibrated (the Brier score
gets worse); the ranking (AUC, AP) changes little.  The weight is, to first order, a shift of the decision threshold: a threshold
fitted on the validation season is the fair way to compare F1 across weights.

``--cmp-weights balanced`` / ``sqrt`` weight the compound classes by their (inverse) training frequency, ``none`` leaves them.
"""

from __future__ import annotations

import numpy as np

from f1strategy.finetune.labels import N_CMP


def _power(spec: str) -> float | None:
    s = str(spec).strip().lower()
    if s in ("auto", "balanced"):
        return 1.0
    if s == "sqrt":
        return 0.5
    return None


def pit_pos_weight(tr_t: dict, horizons: tuple[int, ...], spec: str = "1") -> list[float]:
    """Positive-class weight of the pit-within-H logits, one per horizon, from the training targets ``tr_t``
    (``valid`` / ``laps`` as returned by ``dataset.window_targets``)."""
    v, laps = tr_t["valid"], tr_t["laps"]
    has = v & np.isfinite(laps)
    pairs = max(int(v.sum()), 1)
    pw = _power(spec)
    out = []
    for h in horizons:
        if pw is None:
            out.append(float(spec))
            continue
        p = float((has & (laps <= h)).sum()) / pairs
        p = min(max(p, 1e-4), 1 - 1e-4)
        out.append(float(((1 - p) / p) ** pw))
    return out


def cmp_class_weight(tr_t: dict, spec: str = "none") -> list[float] | None:
    """Per-class weights of the compound cross-entropy (mean weight 1 over the classes that occur), or None."""
    s = str(spec).strip().lower()
    if s in ("none", "1", ""):
        return None
    pw = _power(s)
    if pw is None:
        raise ValueError("--cmp-weights: none | balanced | sqrt")
    has = tr_t["valid"] & np.isfinite(tr_t["laps"]) & (tr_t["cmp"] >= 0)
    cnt = np.bincount(tr_t["cmp"][has].astype(int), minlength=N_CMP).astype(np.float64)
    w = np.zeros(N_CMP)
    present = cnt > 0
    w[present] = (cnt[present].sum() / (present.sum() * cnt[present])) ** pw
    w[present] /= w[present].mean()
    return [float(x) for x in w]
