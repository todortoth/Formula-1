"""metrics.py - numpy-only metrics for the three fine-tuning tasks (shared by the model and the baselines)."""

from __future__ import annotations

import numpy as np

from f1strategy.finetune.labels import CMP_NAMES, N_CMP


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(ss):  # average rank for ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float(prec[yy].sum() / yy.sum())


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """``p`` is a probability-like score in [0, 1]."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)
    pred = p >= threshold
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    # best F1 over thresholds (an upper bound that does not depend on calibration)
    best = 0.0
    if y.any():
        order = np.argsort(-p, kind="mergesort")
        yy = y[order]
        tps = np.cumsum(yy)
        k = np.arange(1, len(yy) + 1)
        best = float(np.max(2 * tps / (k + yy.sum())))
    return {
        "n": int(len(y)), "base_rate": float(y.mean()) if len(y) else float("nan"),
        "auc": roc_auc(y, p), "ap": average_precision(y, p), "f1": float(f1), "f1_best": best,
        "precision": float(prec), "recall": float(rec), "acc": float((pred == y).mean()) if len(y) else float("nan"),
        "brier": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
    }  # fmt: skip


def regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    e = pred - true
    ae = np.abs(e)
    out = {"n": int(len(true))}
    if len(true) == 0:
        return out | {k: float("nan") for k in ("mae", "medae", "rmse", "bias", "within1", "within2", "within3")}
    out |= {
        "mae": float(ae.mean()), "medae": float(np.median(ae)), "rmse": float(np.sqrt((e**2).mean())), "bias": float(e.mean()),
        "within1": float((np.round(pred) - true).__abs__().__le__(1).mean()),
        "within2": float((np.round(pred) - true).__abs__().__le__(2).mean()),
        "within3": float((np.round(pred) - true).__abs__().__le__(3).mean()),
    }  # fmt: skip
    near = true <= 10
    out["mae_true_le10"] = float(ae[near].mean()) if near.any() else float("nan")
    return out


def classification_metrics(true: np.ndarray, pred: np.ndarray, n_classes: int = N_CMP) -> dict:
    true = np.asarray(true).astype(int)
    pred = np.asarray(pred).astype(int)
    if len(true) == 0:
        return {"n": 0, "acc": float("nan"), "macro_f1": float("nan")}
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (true, pred), 1)
    f1s = []
    for k in range(n_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        if cm[k, :].sum() == 0:  # class absent from the truth: not part of the macro average
            continue
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {
        "n": int(len(true)), "acc": float((true == pred).mean()), "macro_f1": float(np.mean(f1s)),
        "majority_acc": float(np.bincount(true, minlength=n_classes).max() / len(true)),
        "confusion": cm.tolist(), "classes": CMP_NAMES[:n_classes],
    }  # fmt: skip


def summarize(pred: dict, tgt: dict, horizons: tuple[int, ...] = (5,)) -> dict:
    """All metrics of one model on one sample set.

    ``tgt``:  ``valid`` [N, D] bool, ``laps`` [N, D] (NaN = no next stop), ``cmp`` [N, D] int (-1 none), ``nomore`` [N, D].
    ``pred``: ``pit`` [N, D, len(horizons)] probabilities, ``laps`` [N, D] (laps), ``cmp`` [N, D] class,
              ``nomore`` [N, D] probability; any key may be missing (that task is then skipped)."""
    v = tgt["valid"]
    out: dict = {"n_samples": int(v.sum())}
    laps = tgt["laps"]
    has = v & np.isfinite(laps)
    if "pit" in pred:
        for i, h in enumerate(horizons):
            y = has & (laps <= h)
            out[f"pit{h}"] = binary_metrics(y[v], pred["pit"][..., i][v])
    if "laps" in pred:
        out["laps"] = regression_metrics(laps[has], pred["laps"][has])
    if "cmp" in pred:
        m = has & (tgt["cmp"] >= 0)
        out["cmp"] = classification_metrics(tgt["cmp"][m], pred["cmp"][m])
    if "nomore" in pred:
        out["nomore"] = binary_metrics(tgt["nomore"][v], pred["nomore"][v])
    return out


def flatten_report(rep: dict) -> dict[str, float]:
    """The headline numbers of a ``summarize`` result as one flat dict (for tables / logs)."""
    f: dict[str, float] = {}
    for k, v in rep.items():
        if k.startswith("pit") and isinstance(v, dict):
            f[f"{k}/auc"], f[f"{k}/ap"], f[f"{k}/f1_best"], f[f"{k}/brier"] = (
                v["auc"],
                v["ap"],
                v["f1_best"],
                v["brier"],
            )
    if "laps" in rep:
        for m in ("mae", "medae", "within1", "within2"):
            f[f"laps/{m}"] = rep["laps"][m]
    if "cmp" in rep:
        f["cmp/acc"], f["cmp/macro_f1"] = rep["cmp"]["acc"], rep["cmp"]["macro_f1"]
    if "nomore" in rep:
        f["nomore/auc"] = rep["nomore"]["auc"]
    return f
