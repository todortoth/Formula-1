"""metrics.py - numpy-only metrics for the three fine-tuning tasks (shared by the model and the baselines).

The positive classes are rare (about 14 % of the (window, car) pairs pit within 5 laps), so threshold-free scores such as the ROC-AUC
can look comfortable while the actual decisions are poor.  Every binary task therefore reports, next to AUC and average precision,
F1 / precision / recall at a threshold that is fitted on a *validation* season (``fit_thresholds``) and applied unchanged to the test
season; ``f1_best`` is the same F1 with the threshold chosen on the evaluated set itself (an upper bound, not a result).
``label_stats`` / ``label_table`` describe how many positives and negatives every task has.
"""

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


def _curve(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tie-aware operating points: for every distinct score (descending) the threshold and the cumulative tp / fp
    when everything with ``score >= threshold`` is called positive.  Tied scores are never split."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]  # last index of every tie group
    return ss[last], np.cumsum(yy)[last], np.cumsum(~yy)[last]


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    _, tp, fp = _curve(y, s)
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Threshold (``score >= thr`` is positive) that maximises F1 on (y, s), and that F1.  No positives -> (0.5, 0.0)."""
    y = np.asarray(y).astype(bool)
    if len(y) == 0 or not y.any():
        return 0.5, 0.0
    thr, tp, fp = _curve(y, s)
    f1 = 2 * tp / (2 * tp + fp + (y.sum() - tp))
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """``p`` is a probability-like score; ``f1`` / ``precision`` / ``recall`` / ``mcc`` are at ``threshold``."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=np.float64)
    pred = p >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    prec, rec, spec = tp / max(tp + fp, 1), tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    den = float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    thr_best, f1_best = best_f1_threshold(y, p)
    p05 = p >= 0.5
    tp5, fp5, fn5 = int((p05 & y).sum()), int((p05 & ~y).sum()), int((~p05 & y).sum())
    return {
        "n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()) if len(y) else float("nan"),
        "auc": roc_auc(y, p), "ap": average_precision(y, p), "threshold": float(threshold),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)), "precision": float(prec), "recall": float(rec),
        "specificity": float(spec), "balanced_acc": float((rec + spec) / 2), "mcc": float((tp * tn - fp * fn) / den) if den > 0 else 0.0,
        "acc": float((pred == y).mean()) if len(y) else float("nan"),
        "f1_best": f1_best, "thr_best": thr_best,
        "f1_05": float(2 * tp5 / max(2 * tp5 + fp5 + fn5, 1)), "precision_05": float(tp5 / max(tp5 + fp5, 1)),
        "recall_05": float(tp5 / max(tp5 + fn5, 1)),
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
    f1s, per_class, recalls = [], [], []
    for k in range(n_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        support = int(cm[k, :].sum())
        if support == 0:  # class absent from the truth: not part of the macro average
            continue
        prec, rec = tp / max(tp + fp, 1), tp / support
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        f1s.append(f1)
        recalls.append(rec)
        per_class.append(
            {"class": CMP_NAMES[k], "support": support, "precision": float(prec), "recall": float(rec), "f1": float(f1)}
        )
    return {
        "n": int(len(true)), "acc": float((true == pred).mean()), "macro_f1": float(np.mean(f1s)),
        "balanced_acc": float(np.mean(recalls)), "majority_acc": float(np.bincount(true, minlength=n_classes).max() / len(true)),
        "per_class": per_class, "confusion": cm.tolist(), "classes": CMP_NAMES[:n_classes],
    }  # fmt: skip


def fit_thresholds(pred: dict, tgt: dict, horizons: tuple[int, ...] = (5,)) -> dict[str, float]:
    """F1-optimal thresholds of the binary tasks on one (validation) sample set: ``{"pit5": thr, ..., "nomore": thr}``."""
    v = tgt["valid"]
    laps = tgt["laps"]
    has = v & np.isfinite(laps)
    out: dict[str, float] = {}
    if "pit" in pred:
        for i, h in enumerate(horizons):
            out[f"pit{h}"] = best_f1_threshold((has & (laps <= h))[v], pred["pit"][..., i][v])[0]
    if "nomore" in pred:
        out["nomore"] = best_f1_threshold(tgt["nomore"][v], pred["nomore"][v])[0]
    return out


def summarize(
    pred: dict, tgt: dict, horizons: tuple[int, ...] = (5,), thresholds: dict[str, float] | None = None
) -> dict:
    """All metrics of one model on one sample set.

    ``tgt``:  ``valid`` [N, D] bool, ``laps`` [N, D] (NaN = no next stop), ``cmp`` [N, D] int (-1 none), ``nomore`` [N, D].
    ``pred``: ``pit`` [N, D, len(horizons)] probabilities, ``laps`` [N, D] (laps), ``cmp`` [N, D] class,
              ``nomore`` [N, D] probability; any key may be missing (that task is then skipped).
    ``thresholds``: decision thresholds of the binary tasks (``fit_thresholds`` on a validation season); default 0.5."""
    thr = thresholds or {}
    v = tgt["valid"]
    out: dict = {"n_samples": int(v.sum())}
    laps = tgt["laps"]
    has = v & np.isfinite(laps)
    if "pit" in pred:
        for i, h in enumerate(horizons):
            y = has & (laps <= h)
            out[f"pit{h}"] = binary_metrics(y[v], pred["pit"][..., i][v], thr.get(f"pit{h}", 0.5))
    if "laps" in pred:
        out["laps"] = regression_metrics(laps[has], pred["laps"][has])
    if "cmp" in pred:
        m = has & (tgt["cmp"] >= 0)
        out["cmp"] = classification_metrics(tgt["cmp"][m], pred["cmp"][m])
    if "nomore" in pred:
        out["nomore"] = binary_metrics(tgt["nomore"][v], pred["nomore"][v], thr.get("nomore", 0.5))
    return out


def flatten_report(rep: dict) -> dict[str, float]:
    """The headline numbers of a ``summarize`` result as one flat dict (for tables / logs)."""
    f: dict[str, float] = {}
    for k, v in rep.items():
        if k.startswith("pit") and isinstance(v, dict):
            for m in ("f1", "precision", "recall", "auc", "ap", "f1_best", "brier", "threshold"):
                f[f"{k}/{m}"] = v[m]
    if "laps" in rep:
        for m in ("mae", "medae", "within1", "within2"):
            f[f"laps/{m}"] = rep["laps"][m]
    if "cmp" in rep:
        f["cmp/acc"], f["cmp/macro_f1"], f["cmp/balanced_acc"] = (
            rep["cmp"]["acc"],
            rep["cmp"]["macro_f1"],
            rep["cmp"]["balanced_acc"],
        )
    if "nomore" in rep:
        for m in ("f1", "precision", "recall", "auc", "ap", "f1_best", "threshold"):
            f[f"nomore/{m}"] = rep["nomore"][m]
    return f


# ------------------------------------------------------------------------------------------------ label distribution
def label_stats(tgt: dict, horizons: tuple[int, ...] = (5,)) -> dict:
    """How many positives / negatives every task has in one sample set (same ``tgt`` as ``summarize``)."""
    v = tgt["valid"]
    laps = tgt["laps"]
    has = v & np.isfinite(laps)
    pairs = int(v.sum())
    out: dict = {"windows": int(len(v)), "windows_with_labels": int(v.any(1).sum()), "pairs": pairs}
    for h in horizons:
        pos = int((has & (laps <= h)).sum())
        out[f"pit{h}"] = {"positives": pos, "negatives": pairs - pos, "rate": pos / pairs if pairs else float("nan")}
    pos = int((tgt["nomore"] & v).sum())
    out["nomore"] = {"positives": pos, "negatives": pairs - pos, "rate": pos / pairs if pairs else float("nan")}
    x = laps[has]
    out["laps"] = {"n": int(has.sum()), "share_of_pairs": float(has.sum() / pairs) if pairs else float("nan")}
    if len(x):
        q = np.percentile(x, [0, 25, 50, 75, 100])
        out["laps"] |= {
            "min": float(q[0]),
            "q25": float(q[1]),
            "median": float(q[2]),
            "q75": float(q[3]),
            "max": float(q[4]),
            "mean": float(x.mean()),
        }
    m = has & (tgt["cmp"] >= 0)
    cnt = np.bincount(tgt["cmp"][m].astype(int), minlength=N_CMP)
    out["cmp"] = {"n": int(m.sum()), "counts": {CMP_NAMES[k]: int(cnt[k]) for k in range(N_CMP)}}
    return out


def label_table(stats: dict[str, dict], horizons: tuple[int, ...] = (5,)) -> str:
    """Markdown tables (one per task family) of ``label_stats`` results, ``stats = {"train 2019-2024": ..., "test 2026": ...}``."""
    L = []
    L += ["| set | windows | (window, car) pairs |" + "".join(f" pit{h} positives | pit{h} negatives | pit{h} rate |" for h in horizons)
          + " no-more positives | no-more negatives | no-more rate |", "|---|---|---|" + "---|---|---|" * len(horizons) + "---|---|---|"]  # fmt: skip
    for n, s in stats.items():
        row = f"| {n} | {s['windows']:,} | {s['pairs']:,} |"
        for h in horizons:
            p = s[f"pit{h}"]
            row += f" {p['positives']:,} | {p['negatives']:,} | {100 * p['rate']:.1f} % |"
        p = s["nomore"]
        L.append(row + f" {p['positives']:,} | {p['negatives']:,} | {100 * p['rate']:.1f} % |")
    L += ["", "| set | pairs with a next stop (laps) | laps to stop: min / q25 / median / q75 / max | stops with a known compound | " + " | ".join(CMP_NAMES) + " |",
          "|---|---|---|---|" + "---|" * len(CMP_NAMES)]  # fmt: skip
    for n, s in stats.items():
        lp, c = s["laps"], s["cmp"]
        dist = " / ".join(f"{lp[k]:.0f}" for k in ("min", "q25", "median", "q75", "max")) if "median" in lp else "-"
        cells = " | ".join(f"{c['counts'][k]:,} ({100 * c['counts'][k] / max(c['n'], 1):.1f} %)" for k in CMP_NAMES)
        L.append(f"| {n} | {lp['n']:,} ({100 * lp['share_of_pairs']:.0f} % of pairs) | {dist} | {c['n']:,} | {cells} |")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ plain-text report
def _grid(header: list[str], rows: list[list[str]]) -> str:
    w = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]

    def line(r):
        return "  ".join(c.ljust(w[0]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r)).rstrip()

    return "\n".join([line(header), "  ".join("-" * x for x in w)] + [line(r) for r in rows])


def _block(title: str, names: list[str], cols: list[tuple[str, str, int]], get, fmt: str = "{:.3f}") -> str:
    """One table: a row per method, ``cols = (header, key, direction)`` with direction +1 higher / -1 lower is better / 0 none.
    The best method of every column (the ``prior`` row excluded) gets a ``*``."""
    vals = [[get(n, k) for _, k, _ in cols] for n in names]
    mark = [None] * len(cols)
    cand = [i for i, n in enumerate(names) if not n.startswith("prior")] or list(range(len(names)))
    for j, (_, _, d) in enumerate(cols):
        ok = [i for i in cand if vals[i][j] is not None and np.isfinite(vals[i][j])]
        if d and ok:
            mark[j] = (max if d > 0 else min)(ok, key=lambda i: vals[i][j])
    rows = []
    for i, n in enumerate(names):
        cells = [n]
        for j, v in enumerate(vals[i]):
            txt = "-" if v is None or not np.isfinite(v) else fmt.format(v)
            cells.append(txt + ("*" if mark[j] == i else " "))
        rows.append(cells)
    return title + "\n" + _grid(["method"] + [h for h, _, _ in cols], rows)


def report_text(reports: dict[str, dict], horizons: tuple[int, ...] = (5,), val_label: str = "") -> str:
    """The reports of ``summarize`` (one per method) as aligned plain-text tables, one block per task."""
    names = list(reports)
    first = reports[names[0]]
    out = [f"{first['n_samples']:,} (window, car) pairs.   * = best of the column (prior excluded).   "
           f"F1 / P / R at the threshold of the 'thr' column"
           + (f" (fitted on {val_label})" if val_label else " (fixed)") + "; F1best = threshold chosen on this set (upper bound); F1@.5 = threshold 0.5."]  # fmt: skip
    bin_cols = [("AUC", "auc", 1), ("AP", "ap", 1), ("thr", "threshold", 0), ("F1", "f1", 1), ("P", "precision", 1),
                ("R", "recall", 1), ("F1best", "f1_best", 1), ("F1@.5", "f1_05", 1)]  # fmt: skip
    for h in horizons:
        k = f"pit{h}"
        r0 = first[k]
        p = r0["base_rate"]
        out.append("\n" + _block(f"[pit within {h} laps]  positives {r0['positives']:,} of {r0['n']:,} ({100 * p:.1f} %), "
                                  f"F1 of 'everything is a stop' = {2 * p / (1 + p):.3f}", names, bin_cols,
                                  lambda n, c, k=k: reports[n][k].get(c)))  # fmt: skip
    if "laps" in first:
        out.append("\n" + _block(f"[laps to the next stop]  {first['laps']['n']:,} pairs with a next stop", names,
                                  [("MAE", "mae", -1), ("medAE", "medae", -1), ("+-1 lap", "within1", 1), ("+-2 laps", "within2", 1)],
                                  lambda n, c: reports[n]["laps"].get(c)))  # fmt: skip
    if "cmp" in first:
        out.append("\n" + _block(f"[next compound]  {first['cmp']['n']:,} stops, most frequent class {100 * first['cmp']['majority_acc']:.1f} %",
                                  names, [("acc", "acc", 1), ("macroF1", "macro_f1", 1), ("balAcc", "balanced_acc", 1)],
                                  lambda n, c: reports[n]["cmp"].get(c)))  # fmt: skip
        pc = first["cmp"].get("per_class")
        if pc:
            rows = [
                [d["class"], f"{d['support']:,}", f"{d['precision']:.3f}", f"{d['recall']:.3f}", f"{d['f1']:.3f}"]
                for d in pc
            ]
            out.append(
                f"  per class, {names[0]}:\n"
                + "\n".join("  " + ln for ln in _grid(["class", "support", "P", "R", "F1"], rows).split("\n"))
            )
    if "nomore" in first:
        r0 = first["nomore"]
        p = r0["base_rate"]
        out.append("\n" + _block(f"[no further stop]  positives {r0['positives']:,} of {r0['n']:,} ({100 * p:.1f} %)", names, bin_cols,
                                  lambda n, c: reports[n]["nomore"].get(c)))  # fmt: skip
    return "\n".join(out)


def label_text(stats: dict[str, dict], horizons: tuple[int, ...] = (5,)) -> str:
    """``label_stats`` results as aligned plain text (``stats = {"test 2026": ...}``)."""
    L = []
    for name, s in stats.items():
        L.append(f"label distribution, {name}: {s['windows']:,} windows, {s['pairs']:,} (window, car) pairs")
        rows = []
        for key, lab in [(f"pit{h}", f"pit within {h} laps") for h in horizons] + [("nomore", "no further stop")]:
            t = s[key]
            rows.append(
                [
                    lab,
                    f"{t['positives']:,}",
                    f"{100 * t['rate']:.1f} %",
                    f"{t['negatives']:,}",
                    f"{100 * (1 - t['rate']):.1f} %",
                ]
            )
        L.append(_grid(["task", "positive", "", "negative", ""], rows))
        lp, c = s["laps"], s["cmp"]
        if "median" in lp:
            L.append(
                f"laps to the next stop: {lp['n']:,} pairs; min {lp['min']:.0f}, q25 {lp['q25']:.0f}, median {lp['median']:.0f}, q75 {lp['q75']:.0f}, max {lp['max']:.0f}"
            )
        L.append(
            f"next compound: {c['n']:,} stops; "
            + ", ".join(f"{k} {v:,} ({100 * v / max(c['n'], 1):.1f} %)" for k, v in c["counts"].items())
        )
    return "\n".join(L)
