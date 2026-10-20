"""Metrics, scrub levels and the numpy baselines."""

import numpy as np
import pytest

from f1strategy.finetune import baselines as B
from f1strategy.finetune import metrics as M
from f1strategy.finetune import scrub as S
from f1strategy.finetune.labels import TAB_NAMES

NAMES = ["x_m", "in_pit", "last_lap_time_s", "lateral_offset_m", "heading_rel_track_sin", "heading_rel_track_cos",
         "on_track", "stint", "pit_stops", "tyre_age"]  # fmt: skip


def test_auc_and_ap_known_values():
    y = np.array([0, 0, 1, 1])
    assert M.roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert M.roc_auc(y, np.zeros(4)) == pytest.approx(0.5)  # all ties
    assert M.average_precision(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.8333, abs=1e-3)
    assert np.isnan(M.roc_auc(np.zeros(4), np.arange(4)))


def test_regression_and_classification_metrics():
    r = M.regression_metrics(np.array([5, 10, 3.0]), np.array([6, 10, 1.0]))
    assert (
        r["mae"] == pytest.approx(1.0) and r["within1"] == pytest.approx(2 / 3) and r["bias"] == pytest.approx(-1 / 3)
    )
    c = M.classification_metrics(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 1]))
    assert c["acc"] == 0.75 and c["majority_acc"] == 0.5


def test_scrub_levels_and_fill():
    assert S.scrub_columns(NAMES, "none") == []
    assert S.scrub_columns(NAMES, "basic") == [1, 2]
    assert set(S.scrub_columns(NAMES, "strict")) == {1, 2, 3, 4, 5, 6, 7, 8}
    assert 9 not in S.scrub_columns(NAMES, "strict")  # tyre age is the strategy state: never scrubbed
    with pytest.raises(ValueError):
        S.scrub_columns(NAMES, "bogus")
    drv = np.ones((2, 3, 4, len(NAMES)), dtype=np.float32)
    S.scrub_raw({"drv": drv}, [1, 2], "zero")
    assert drv[..., 1].sum() == 0 and drv[..., 0].min() == 1
    S.scrub_raw({"drv": drv}, [0], "nan")
    assert np.isnan(drv[..., 0]).all()


def _targets(n=400, seed=0, years_tab=None):
    """Synthetic world: stint length per compound, the stop happens when tyre age reaches it."""
    rng = np.random.default_rng(seed)
    D, K = 22, len(TAB_NAMES)
    ix = {k: i for i, k in enumerate(TAB_NAMES)}
    tab = np.full((n, D, K), np.nan, np.float32)
    cmp_now = rng.integers(0, 3, (n, D))
    age = rng.integers(0, 30, (n, D)).astype(np.float32)
    for k in range(3):
        tab[..., ix["cmp_soft"] + k] = cmp_now == k
    tab[..., ix["tyre_age"]] = age
    tab[..., ix["pit_stops"]] = rng.integers(0, 3, (n, D))
    tab[..., ix["laps_remaining"]] = rng.integers(5, 60, (n, D))
    L = np.array([15, 25, 35])[cmp_now]
    laps = np.maximum(L - age, 1).astype(np.float32)
    nomore = laps > tab[..., ix["laps_remaining"]]
    laps = np.where(nomore, np.nan, laps).astype(np.float32)
    nxt = np.where(nomore, -1, np.minimum(cmp_now + 1, 2)).astype(np.int64)
    return {"valid": rng.random((n, D)) < 0.9, "laps": laps, "cmp": nxt, "nomore": nomore, "tab": tab,
            "race": np.zeros(n, np.int32), "frame": np.arange(n)}  # fmt: skip


def test_rule_baseline_recovers_a_rule_world_and_beats_the_prior():
    tr, te = _targets(seed=1), _targets(seed=2)
    pred = B.run_baselines(tr, te, (5,), ("prior", "rule"))
    rule = M.summarize(pred["rule"], te, (5,))
    prior = M.summarize(pred["prior"], te, (5,))
    assert rule["pit5"]["auc"] > 0.95 and prior["pit5"]["auc"] == pytest.approx(0.5)
    assert rule["laps"]["mae"] < 1.0 < prior["laps"]["mae"]
    assert rule["cmp"]["acc"] > 0.9
    assert pred["rule"]["pit"].shape == (400, 22, 1)


def test_gbdt_baseline_runs_if_sklearn_is_present():
    pytest.importorskip("sklearn")
    tr, te = _targets(300, seed=3), _targets(100, seed=4)
    pred = B.run_baselines(tr, te, (5,), ("gbdt",))
    rep = M.summarize(pred["gbdt"], te, (5,))
    assert rep["pit5"]["auc"] > 0.9 and rep["laps"]["mae"] < 3
