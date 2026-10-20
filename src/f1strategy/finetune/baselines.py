"""
baselines.py - what the fine-tuned model has to beat.  All baselines see the same (window, car) samples, scored by
``metrics.summarize``; none sees the sequence itself, only the strategy state of the car at the END of the window.

* ``prior``   base rate / median laps / most frequent compound of the training set (no input at all).
* ``rule``    the strategist's rule of thumb: the median stint length of the current compound (training set) minus the tyre
              age gives the laps to the stop; the next compound comes from a transition table (current compound, stops so far).
* ``gbdt``    gradient-boosted trees (scikit-learn ``HistGradientBoosting``) on the raw strategy features of the car and of the
              race (tyre age, compound, stint, stops, position, gaps, pace, laps remaining, flags, weather, track length).
              This is the honest competitor: if the temporal graph model does not beat it, the sequence adds nothing.

Needs ``numpy``; ``gbdt`` needs ``scikit-learn`` (skipped with a note if missing).
"""

from __future__ import annotations

import numpy as np

from f1strategy.finetune.labels import N_CMP, TAB_NAMES

IX = {n: i for i, n in enumerate(TAB_NAMES)}


def _flat(t: dict) -> tuple[np.ndarray, ...]:
    """valid (window, car) pairs -> rows."""
    m = t["valid"]
    return m, t["tab"][m], t["laps"][m], t["cmp"][m], t["nomore"][m]


def current_cmp(tab: np.ndarray) -> np.ndarray:
    oh = tab[:, [IX[n] for n in ("cmp_soft", "cmp_medium", "cmp_hard", "cmp_inter", "cmp_wet")]]
    c = np.nan_to_num(oh).argmax(1)
    c[~(np.nan_to_num(oh).max(1) > 0.5)] = -1
    return c


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pack(m, pit, laps, cmp, nomore, horizons) -> dict:
    N, D = m.shape

    def put(vals, fill, dtype):
        out = np.full((N, D) + np.shape(vals)[1:], fill, dtype=dtype)
        out[m] = vals
        return out

    return {"pit": put(pit, 0.0, np.float32), "laps": put(laps, 1.0, np.float32), "cmp": put(cmp, 0, np.int64),
            "nomore": put(nomore, 0.0, np.float32)}  # fmt: skip


class Prior:
    name = "prior"

    def fit(self, t: dict, horizons=(5,)):
        m, tab, laps, cmp, nomore = _flat(t)
        has = np.isfinite(laps)
        self.rate = np.array([(has & (laps <= h)).mean() for h in horizons], dtype=np.float32)
        self.laps = float(np.median(laps[has]))
        k = cmp[has & (cmp >= 0)]
        self.cmp = int(np.bincount(k, minlength=N_CMP).argmax())
        self.nomore = float(nomore.mean())
        return self

    def predict(self, t: dict, horizons=(5,)) -> dict:
        m, tab, *_ = _flat(t)
        n = len(tab)
        return _pack(m, np.tile(self.rate, (n, 1)), np.full(n, self.laps), np.full(n, self.cmp), np.full(n, self.nomore), horizons)


class Rule:
    name = "rule"

    def fit(self, t: dict, horizons=(5,)):
        m, tab, laps, cmp, nomore = _flat(t)
        c = current_cmp(tab)
        has = np.isfinite(laps)
        stint_len = np.nan_to_num(tab[:, IX["tyre_age"]]) + laps  # tyre age now + laps to go = length of the stint
        glob = float(np.median(stint_len[has]))
        self.L = np.full(N_CMP, glob)
        for k in range(N_CMP):
            s = has & (c == k)
            if s.sum() >= 30:
                self.L[k] = float(np.median(stint_len[s]))
        self.table = np.ones((N_CMP, 2, N_CMP))  # (compound, stops so far > 0) -> next compound counts (+1 smoothing)
        stops = np.nan_to_num(tab[:, IX["pit_stops"]]) > 0
        ok = has & (cmp >= 0) & (c >= 0)
        np.add.at(self.table, (c[ok], stops[ok].astype(int), cmp[ok]), 1)
        self.fallback = int(np.bincount(cmp[has & (cmp >= 0)], minlength=N_CMP).argmax())
        return self

    def predict(self, t: dict, horizons=(5,), scale_s: float = 2.0) -> dict:
        m, tab, *_ = _flat(t)
        c = current_cmp(tab)
        age = np.nan_to_num(tab[:, IX["tyre_age"]])
        pred = np.clip(self.L[np.where(c >= 0, c, 0)] - age, 1.0, 60.0)  # at least the current lap
        pit = np.stack([_sigmoid((h + 0.5 - pred) / scale_s) for h in horizons], -1)
        stops = (np.nan_to_num(tab[:, IX["pit_stops"]]) > 0).astype(int)
        nxt = np.where(c >= 0, self.table[np.maximum(c, 0), stops].argmax(-1), self.fallback)
        remaining = np.nan_to_num(tab[:, IX["laps_remaining"]], nan=1e3)
        nomore = _sigmoid((pred - remaining) / scale_s)  # the predicted stop lies behind the chequered flag
        return _pack(m, pit, pred, nxt, nomore, horizons)


class Gbdt:
    name = "gbdt"

    def __init__(self, max_rows: int = 400_000, seed: int = 0, **kw):
        self.max_rows, self.seed, self.kw = max_rows, seed, kw

    def fit(self, t: dict, horizons=(5,)):
        from sklearn.ensemble import HistGradientBoostingClassifier as C
        from sklearn.ensemble import HistGradientBoostingRegressor as R

        m, tab, laps, cmp, nomore = _flat(t)
        rng = np.random.default_rng(self.seed)
        if len(tab) > self.max_rows:
            s = rng.choice(len(tab), self.max_rows, replace=False)
            tab, laps, cmp, nomore = tab[s], laps[s], cmp[s], nomore[s]
        has = np.isfinite(laps)
        kw = {"max_iter": 300, "learning_rate": 0.06, "max_leaf_nodes": 31, "l2_regularization": 1.0, "random_state": self.seed} | self.kw
        self.pit = [C(**kw).fit(tab, has & (laps <= h)) for h in horizons]
        self.laps = R(loss="absolute_error", **kw).fit(tab[has], np.minimum(laps[has], 60.0))
        k = has & (cmp >= 0)
        self.cmp = C(**kw).fit(tab[k], cmp[k])
        self.nomore = C(**kw).fit(tab, nomore)
        return self

    def predict(self, t: dict, horizons=(5,)) -> dict:
        m, tab, *_ = _flat(t)
        pit = np.stack([mdl.predict_proba(tab)[:, 1] for mdl in self.pit], -1)
        proba = self.cmp.predict_proba(tab)
        nxt = self.cmp.classes_[proba.argmax(1)]
        return _pack(m, pit, np.clip(self.laps.predict(tab), 1.0, 60.0), nxt, self.nomore.predict_proba(tab)[:, 1], horizons)


def run_baselines(train_t: dict, test_t: dict, horizons=(5,), names=("prior", "rule", "gbdt")) -> dict[str, dict]:
    """Fit on the training targets, predict the test targets -> ``{name: prediction dict}`` (shape of ``metrics.summarize``)."""
    out = {}
    for n in names:
        if n == "gbdt":
            try:
                import sklearn  # noqa: F401
            except ImportError:
                print("gbdt baseline skipped: pip install scikit-learn")
                continue
            mdl = Gbdt()
        else:
            mdl = {"prior": Prior, "rule": Rule}[n]()
        out[n] = mdl.fit(train_t, horizons).predict(test_t, horizons)
    return out
