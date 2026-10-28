"""Confusion matrix of the next-compound task from an ``evaluate`` report (needs ``--report``).

    python scripts/confusion_matrix.py runs/eval/test2026_groups_long.json                      # text, every method but the prior
    python scripts/confusion_matrix.py runs/eval/test2026_groups_long.json --methods ens_BE,gbdt --png compound_cm.png

Rows are the true compound of the next stop, columns the predicted one.  Printed: the counts with the recall of every true class and the precision
of every predicted class, then the same matrix in percent of each row (where the cars of a true class end up).  Classes that never occur in the
truth nor in the predictions (wet tyres in dry seasons) are left out unless ``--all-classes``.  ``--png`` draws the row-normalised matrices
side by side (needs matplotlib).
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def trim(cm: np.ndarray, classes: list[str], keep_all: bool = False) -> tuple[np.ndarray, list[str]]:
    cm = np.asarray(cm, dtype=np.int64)
    if keep_all:
        return cm, list(classes)
    keep = [k for k in range(len(classes)) if cm[k].sum() > 0 or cm[:, k].sum() > 0]
    return cm[np.ix_(keep, keep)], [classes[k] for k in keep]


def format_confusion(cm: np.ndarray, classes: list[str], title: str = "") -> str:
    cm = np.asarray(cm, dtype=np.int64)
    rows, cols = cm.sum(1), cm.sum(0)
    w = max(7, max(len(c) for c in classes) + 1)
    label = r"true \ predicted"
    head = f"{label:<18}" + "".join(f"{c:>{w}}" for c in classes) + f" |{'total':>8}{'recall':>8}"
    lines = [title, head, "-" * len(head)] if title else [head, "-" * len(head)]
    for i, c in enumerate(classes):
        rec = cm[i, i] / rows[i] if rows[i] else float("nan")
        lines.append(f"{c:<18}" + "".join(f"{v:>{w},}" for v in cm[i]) + f" |{rows[i]:>8,}{rec:>8.3f}")
    prec = [cm[j, j] / cols[j] if cols[j] else float("nan") for j in range(len(classes))]
    lines.append("-" * len(head))
    lines.append(f"{'predicted':<18}" + "".join(f"{v:>{w},}" for v in cols) + f" |{cm.sum():>8,}")
    lines.append(f"{'precision':<18}" + "".join(f"{p:>{w}.3f}" for p in prec))
    lines.append("")
    lines.append("share of each true class (row, %):")
    lines.append(f"{label:<18}" + "".join(f"{c:>{w}}" for c in classes))
    for i, c in enumerate(classes):
        share = 100 * cm[i] / rows[i] if rows[i] else np.full(len(classes), np.nan)
        lines.append(f"{c:<18}" + "".join(f"{v:>{w}.1f}" for v in share))
    return "\n".join(lines)


def plot(panels: list[tuple[str, np.ndarray, list[str]]], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4), squeeze=False)
    for ax, (name, cm, classes) in zip(axes[0], panels):
        share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(share, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}  (accuracy {np.trace(cm) / max(cm.sum(), 1):.3f})", fontsize=10)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{100 * share[i, j]:.0f} %\n{cm[i, j]:,}", ha="center", va="center", fontsize=8,
                        color="white" if share[i, j] > 0.55 else "black")  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    ap.add_argument("--methods", default="", help="comma list of rows of the report (default: all but the prior)")
    ap.add_argument("--png", default="")
    ap.add_argument("--all-classes", action="store_true")
    a = ap.parse_args(argv)
    rep = json.load(open(a.report))["reports"]
    names = [m for m in a.methods.split(",") if m] or [m for m in rep if not m.startswith("prior")]
    panels = []
    for n in names:
        if n not in rep or "cmp" not in rep[n] or "confusion" not in rep[n]["cmp"]:
            raise SystemExit(f"{n!r}: no compound confusion matrix in the report")
        c = rep[n]["cmp"]
        cm, classes = trim(np.array(c["confusion"]), c["classes"], a.all_classes)
        print(
            format_confusion(
                cm, classes, f"[{n}]  {int(cm.sum()):,} stops, accuracy {c['acc']:.3f}, macro-F1 {c['macro_f1']:.3f}"
            )
            + "\n"
        )
        panels.append((n, cm, classes))
    if a.png:
        plot(panels, a.png)
        print(f"figure -> {a.png}")


if __name__ == "__main__":
    main()
