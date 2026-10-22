"""metrics.py - numpy-only metrics for the three fine-tuning tasks."""

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
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0

    while i < len(ss):
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
    precision = tp / np.arange(1, len(yy) + 1)

    return float(precision[yy].sum() / yy.sum())


def binary_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Binary metrics. Positive class is y=True."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)

    if len(y) != len(p):
        raise ValueError("y and p must have the same length")

    n = len(y)
    positive = int(y.sum())
    negative = n - positive

    pred = p >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # Diagnostic only: optimized on this evaluation set.
    # Do not use this as the primary test-set comparison metric.
    best_f1 = 0.0
    if positive:
        order = np.argsort(-p, kind="mergesort")
        yy = y[order]
        tp_curve = np.cumsum(yy)
        k = np.arange(1, n + 1)
        best_f1 = float(np.max(2 * tp_curve / (k + positive)))

    return {
        "n": n,
        "positive_count": positive,
        "negative_count": negative,
        "positive_rate": float(positive / n) if n else float("nan"),
        "negative_rate": float(negative / n) if n else float("nan"),
        "threshold": float(threshold),
        "auc": roc_auc(y, p),
        "ap": average_precision(y, p),
        "f1": float(f1),
        "f1_best": best_f1,
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float((pred == y).mean()) if n else float("nan"),
        "brier": float(np.mean((p - y) ** 2)) if n else float("nan"),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def regression_metrics(
    true: np.ndarray,
    pred: np.ndarray,
) -> dict:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    e = pred - true
    ae = np.abs(e)
    out = {"n": int(len(true))}

    if len(true) == 0:
        return out | {
            k: float("nan")
            for k in (
                "mae",
                "medae",
                "rmse",
                "bias",
                "within1",
                "within2",
                "within3",
                "mae_true_le10",
            )
        }

    out |= {
        "mae": float(ae.mean()),
        "medae": float(np.median(ae)),
        "rmse": float(np.sqrt((e**2).mean())),
        "bias": float(e.mean()),
        "within1": float((np.abs(np.round(pred) - true) <= 1).mean()),
        "within2": float((np.abs(np.round(pred) - true) <= 2).mean()),
        "within3": float((np.abs(np.round(pred) - true) <= 3).mean()),
    }

    near = true <= 10
    out["mae_true_le10"] = float(ae[near].mean()) if near.any() else float("nan")
    return out


def classification_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    n_classes: int = N_CMP,
) -> dict:
    true = np.asarray(true).astype(int)
    pred = np.asarray(pred).astype(int)

    if len(true) != len(pred):
        raise ValueError("true and pred must have the same length")

    if len(true) == 0:
        return {
            "n": 0,
            "acc": float("nan"),
            "macro_f1": float("nan"),
            "class_counts": [0] * n_classes,
            "class_rates": [float("nan")] * n_classes,
            "classes": CMP_NAMES[:n_classes],
        }

    if np.any((true < 0) | (true >= n_classes)) or np.any((pred < 0) | (pred >= n_classes)):
        raise ValueError("Class labels must be in [0, n_classes)")

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (true, pred), 1)

    f1s = []
    for k in range(n_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp

        # Include every class present in the ground truth.
        if cm[k, :].sum() == 0:
            continue

        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))

    counts = np.bincount(true, minlength=n_classes)

    return {
        "n": int(len(true)),
        "acc": float((true == pred).mean()),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "majority_acc": float(counts.max() / len(true)),
        "class_counts": counts.tolist(),
        "class_rates": (counts / len(true)).tolist(),
        "confusion": cm.tolist(),
        "classes": CMP_NAMES[:n_classes],
    }


def summarize(
    pred: dict,
    tgt: dict,
    horizons: tuple[int, ...] = (5,),
) -> dict:
    """Summarize task metrics and ground-truth label distributions."""
    v = tgt["valid"].astype(bool)
    out = {"n_samples": int(v.sum())}
    laps = tgt["laps"]
    has = v & np.isfinite(laps)

    if "pit" in pred:
        for i, h in enumerate(horizons):
            # Samples without a next pit stop count as negatives.
            y = has & (laps <= h)
            out[f"pit{h}"] = binary_metrics(
                y[v],
                pred["pit"][..., i][v],
            )

    if "laps" in pred:
        out["laps"] = regression_metrics(
            laps[has],
            pred["laps"][has],
        )

    if "cmp" in pred:
        m = has & (tgt["cmp"] >= 0)
        out["cmp"] = classification_metrics(
            tgt["cmp"][m],
            pred["cmp"][m],
        )

    if "nomore" in pred:
        out["nomore"] = binary_metrics(
            tgt["nomore"][v],
            pred["nomore"][v],
        )

    return out


def flatten_report(rep: dict) -> dict[str, float]:
    """Flatten headline metrics for tables and logs."""
    f: dict[str, float] = {}

    for key, values in rep.items():
        if key.startswith("pit") and isinstance(values, dict):
            for metric in (
                "auc",
                "ap",
                "f1",
                "f1_best",
                "precision",
                "recall",
                "accuracy",
                "brier",
                "positive_count",
                "negative_count",
                "positive_rate",
                "negative_rate",
            ):
                f[f"{key}/{metric}"] = values[metric]

    if "laps" in rep:
        for metric in ("mae", "medae", "within1", "within2"):
            f[f"laps/{metric}"] = rep["laps"][metric]

    if "cmp" in rep:
        f["cmp/acc"] = rep["cmp"]["acc"]
        f["cmp/macro_f1"] = rep["cmp"]["macro_f1"]

    if "nomore" in rep:
        for metric in (
            "auc",
            "ap",
            "f1",
            "f1_best",
            "precision",
            "recall",
            "accuracy",
            "brier",
            "positive_count",
            "negative_count",
            "positive_rate",
            "negative_rate",
        ):
            f[f"nomore/{metric}"] = rep["nomore"][metric]

    return f
