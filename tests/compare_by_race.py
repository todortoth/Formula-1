"""Paired comparison of two methods over the races of an ``evaluate`` report (needs ``--report``).

    python scripts/compare_by_race.py runs/eval/test2026_medium.json final_tab_s0 gbdt

Per metric: mean difference (A - B) over the races, in how many races A is better, and a 95 % bootstrap interval of the mean
difference (races are resampled; with 16 races an interval that excludes 0 is the minimum for calling a difference real).
A race is one sample: the (window, car) pairs inside it are strongly correlated and must not be counted as independent.
"""

from __future__ import annotations

import json
import sys

import numpy as np

KEYS = [  # (flat metric key, True if higher is better)
    ("pit5/ap", True), ("pit5/auc", True), ("pit5/f1", True), ("pit5/f1_best", True),
    ("laps/mae", False), ("cmp/acc", True), ("nomore/auc", True), ("nomore/f1", True),
]  # fmt: skip


def metric_keys(by_race: dict, a: str) -> list[tuple[str, bool]]:
    """``KEYS`` plus AP and best-threshold F1 of every other pit horizon in the report (pit1, pit3, pit10, ...)."""
    first = next(iter(by_race.values()))[a]
    extra = []
    for k in sorted(first):
        if k.startswith("pit") and k.endswith("/ap") and not k.startswith("pit5/"):
            extra += [(k, True), (k[:-3] + "/f1_best", True)]
    return KEYS + extra


def compare(by_race: dict, a: str, b: str, n_boot: int = 10000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for key, higher in metric_keys(by_race, a):
        d = np.array(
            [r[a].get(key, np.nan) - r[b].get(key, np.nan) for r in by_race.values() if a in r and b in r], dtype=float
        )
        d = d[np.isfinite(d)]
        if len(d) < 2:
            continue
        sign = 1.0 if higher else -1.0  # positive = A better
        boots = rng.choice(d, (n_boot, len(d))).mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"metric": key, "mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
                     "races": int(len(d)), "a_better": int((sign * d > 0).sum()), "significant": bool(lo > 0 or hi < 0)})  # fmt: skip
    return rows


def main(argv: list[str]) -> None:
    if len(argv) != 4:
        raise SystemExit(__doc__)
    rep = json.load(open(argv[1]))
    rows = compare(rep["by_race"], argv[2], argv[3])
    print(
        f"{argv[2]} - {argv[3]}  ({rows[0]['races'] if rows else 0} races; 'A better' = races where {argv[2]} is better)"
    )
    print(f"{'metric':<14}{'mean diff':>11}{'95 % interval':>24}{'A better':>10}  real?")
    for r in rows:
        print(f"{r['metric']:<14}{r['mean_diff']:>+11.3f}{'[' + format(r['ci_low'], '+.3f') + ', ' + format(r['ci_high'], '+.3f') + ']':>24}"
              f"{str(r['a_better']) + '/' + str(r['races']):>10}  {'yes' if r['significant'] else 'no'}")  # fmt: skip


if __name__ == "__main__":
    main(sys.argv)
