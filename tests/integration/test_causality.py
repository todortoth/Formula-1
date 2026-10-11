"""
Causality: a snapshot at time t may only depend on data from before t.

Training data built from race replays must not leak the future into the past. This test builds the graph
twice - once from the complete session and once from a session in which everything after t + 2 s has been
deleted or blanked (as it would be live: unfinished laps have no end time) - and requires the same graph.

The 2 s margin covers the interpolation neighbours (position/car samples arrive every ~0.25 s, velocity uses
+-0.3 s).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from f1strategy import DriverTiming, RaceControlTimeline, RaceGraphBuilder, TelemetryStore
from tests.helpers import clone_session
from tests.synthetic.session import T0_DATE

MARGIN = 2.0
# Derived from the *final* lap timestamp of a driver; verified separately below (it is only ever True after that time).
KNOWN_FUTURE_DEPENDENT = {"finished_or_retired"}


def _td(seconds):
    return pd.to_timedelta(float(seconds), unit="s")


def truncate(session, horizon_s):
    """The session as it would look at `horizon_s`: no later samples, unfinished laps without end times."""
    h = _td(horizon_s)
    laps = session.laps[session.laps["LapStartTime"] <= h].copy()
    for col in ("Time", "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime", "PitInTime", "PitOutTime"):
        laps.loc[laps[col] > h, col] = pd.NaT
    unfinished = laps["Time"].isna()
    laps.loc[unfinished, "LapTime"] = pd.NaT
    laps.loc[unfinished, "Position"] = np.nan
    msgs = session.race_control_messages
    return clone_session(
        session,
        laps=laps.reset_index(drop=True),
        pos_data={k: v[v["SessionTime"] <= h].copy() for k, v in session.pos_data.items()},
        car_data={k: v[v["SessionTime"] <= h].copy() for k, v in session.car_data.items()},
        track_status=session.track_status[session.track_status["Time"] <= h].copy(),
        race_control_messages=msgs[msgs["Time"] <= T0_DATE + h].copy(),
        weather_data=session.weather_data[session.weather_data["Time"] <= h].copy(),
    )


def _builder_from(analyser, session):
    timing = DriverTiming(session.laps)
    store = TelemetryStore(session, analyser.track, analyser._driver_map)
    timeline = RaceControlTimeline.from_session(session)
    return RaceGraphBuilder(session, analyser.track, store, timing, timeline, analyser._driver_map)


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-6, rel=1e-9)
    return a == b


@pytest.mark.parametrize("t", [45.0, 100.0, 121.0, 150.0, 184.3, 232.0, 250.0, 300.0, 345.5, 358.0])
def test_graph_does_not_depend_on_data_after_t(analyser, syn, t):
    full = analyser.build_graph_at_time(t)
    cut = _builder_from(analyser, truncate(syn.session, t + MARGIN)).build(t)

    assert full.graph["driver_codes"] == cut.graph["driver_codes"]
    for key, value in full.graph["global"].items():
        assert _same(value, cut.graph["global"][key]), f"global[{key}]: {value} != {cut.graph['global'][key]}"
    for node, attrs in full.nodes(data=True):
        for key, value in attrs.items():
            if key in KNOWN_FUTURE_DEPENDENT:
                continue
            assert _same(value, cut.nodes[node][key]), f"{node}.{key}: {value} != {cut.nodes[node][key]}"
    assert set(full.edges) == set(cut.edges)
    for u, v, attrs in full.edges(data=True):
        for key, value in attrs.items():
            assert _same(value, cut.edges[u, v][key]), f"edge {u}->{v}.{key}"


def test_the_truncation_really_removes_the_future(syn):
    cut = truncate(syn.session, 150.0)
    assert cut.laps["Time"].max() <= _td(150.0) and cut.laps["LapStartTime"].max() <= _td(150.0)
    assert all(v["SessionTime"].max() <= _td(150.0) for v in cut.pos_data.values())
    assert cut.track_status["Time"].max() <= _td(150.0)
    assert cut.race_control_messages["Time"].max() <= T0_DATE + _td(150.0)
    assert cut.laps["Time"].isna().any()  # laps in progress have no end time, like live data


def test_the_causality_test_has_teeth(analyser, syn):
    """Sanity check of the test itself: if the builder were given *different* history, the comparison must fail."""
    t = 184.3
    full = analyser.build_graph_at_time(t)
    corrupted = truncate(syn.session, t + MARGIN)
    corrupted.laps.loc[corrupted.laps["Driver"] == "VER", "Compound"] = "WET"  # alter the past
    other = _builder_from(analyser, corrupted).build(t)
    assert full.nodes["VER"]["compound"] != other.nodes["VER"]["compound"]


def test_finished_flag_is_only_true_after_the_last_lap_timestamp(analyser):
    last = {c: analyser.timing.last_end_time(c) for c in analyser.graph_builder.driver_map.values()}
    for code, end in last.items():
        for t in (end - 60, end - 1, end + 4.9):
            assert not analyser.timing.at(code, t).finished_or_retired, (code, t)
        assert analyser.timing.at(code, end + 5.1).finished_or_retired
