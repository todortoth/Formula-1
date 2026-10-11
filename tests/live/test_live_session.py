"""
Live tests against the real FastF1 / F1 live-timing API.

Deselected by default (`-m 'not live'` in pyproject.toml); run them with:

    python -m pytest -m live tests/live

They download several hundred MB on first use (cached in $F1_CACHE_DIR or ~/fastf1_cache) and depend on the
availability of F1's servers, so they run in a separate scheduled GitHub workflow (live-data.yml), not on every push.
The reference values come from a real run on the 2025 Belgian Grand Prix (6946.6 m, 695 nodes, 20 drivers).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from f1strategy import GraphExporter, SessionAnalyser, suggest_pool_factor

pytestmark = [pytest.mark.live, pytest.mark.filterwarnings("ignore")]  # fastf1 / pandas emit their own warnings

LAP = 44


@pytest.fixture(scope="module")
def spa():
    return SessionAnalyser(2025, "Belgian Grand Prix", "R", cache_dir=os.environ.get("F1_CACHE_DIR"))


def test_track_matches_the_real_circuit(spa):
    tr = spa.track
    assert 6800 < tr.length_m < 7100 and abs(tr.spacing_m - 10) < 0.2
    assert tr.sector_end_s is not None and 0 < tr.sector_end_s[0] < tr.sector_end_s[1] < tr.length_m
    assert (tr.corner_number > 0).sum() >= 18  # Spa has 19 numbered corners
    assert tr.drs_zone.max() >= 2  # at least the Kemmel and start/finish straights
    assert (tr.marshal_sector > 0).all()  # every node belongs to a known marshal sector


def test_all_twenty_drivers_are_present_with_telemetry(spa):
    G = spa.build_session_graph(LAP, "Sector 2")
    assert len(G.graph["driver_codes"]) == 20
    for code in G.graph["driver_codes"]:
        d = G.nodes[code]
        assert 0 <= d["speed_kmh"] < 380 and abs(d["lateral_offset_m"]) < 30 and d["compound"] in (
            "SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN")  # fmt: skip


def test_the_leader_is_on_the_sector_boundary_at_the_snapshot(spa):
    G = spa.build_session_graph(LAP, "Sector 2")
    leader = G.graph["driver_nodes"][0]
    assert leader.position == 1 and abs(spa.track.circular_delta(spa.track.sector_end_s[1], leader.track_s_m)) < 40


def test_snapshots_are_consistent_and_exportable(spa):
    exporter = GraphExporter(normalise=True)
    for sector in ("Sector 1", "Sector 2", "Sector 3"):
        arrays = exporter.export(spa.build_session_graph(LAP, sector))
        factor = suggest_pool_factor(len(arrays.track_x), 20, max_tokens=512)
        pooled = arrays.pool_track(factor)
        assert factor == 2 and pooled.n_tokens <= 512
        assert pooled.track_mask.shape == pooled.track_x.shape


def test_race_control_state_over_the_race_is_plausible(spa):
    """The 2025 Belgian GP started in the wet behind the safety car: some yellow / SC period must exist."""
    tl = spa.timeline.to_frame()
    assert set(tl["track_status"]) & {"2", "4"}
    assert tl["drs_enabled"].dropna().any()


def test_pit_stop_counts_are_plausible(spa):
    G = spa.build_session_graph(LAP, "Sector 1")
    stops = [G.nodes[c]["pit_stops"] for c in G.graph["driver_codes"]]
    assert 0 <= min(stops) and max(stops) <= 5 and sum(stops) > 0


def test_graph_building_is_fast_enough_for_dataset_generation(spa):
    import time

    t0 = time.perf_counter()
    for sector in ("Sector 1", "Sector 2", "Sector 3"):
        spa.build_session_graph(LAP, sector)
    assert (time.perf_counter() - t0) / 3 < 5.0  # seconds per snapshot, after the one-off progress computation


def test_weather_is_available(spa):
    g = spa.build_session_graph(LAP, "Sector 2").graph["global"]
    assert np.isfinite([g["air_temp"], g["track_temp"], g["humidity"]]).all()


def test_continuous_sequence_over_a_real_lap_is_continuous_and_equals_the_point_by_point_graph(spa):
    """Not yet run against the real API when this was written: if the continuity thresholds (1.5 x speed + 3 m)
    turn out to be too tight for real data (pit lane, red-flag stops), adjust them in `continuity_report`."""
    seq = spa.build_sequence_for_laps(LAP, LAP, step_s=2.0)
    assert seq.n_drivers == 20 and seq.n_frames > 30 and seq.step_s == 2.0
    report = seq.continuity_report()
    assert report.ok, report.summary()
    i = seq.n_frames // 2
    ref = GraphExporter().export(spa.build_graph_at_time(float(seq.times[i])))
    got = seq.frame(i)
    for key, value in ref.to_dict().items():
        assert np.array_equal(value, got.to_dict()[key], equal_nan=True), key


def test_gif_of_a_real_lap(spa, tmp_path):
    from PIL import Image

    from f1strategy import RaceAnimator

    seq = spa.build_sequence_for_laps(LAP, LAP, step_s=4.0)
    out = RaceAnimator(seq, colors=spa.driver_colors(), title="live test").save_gif(tmp_path / "lap.gif", fps=5)
    with Image.open(out) as im:
        assert im.n_frames == seq.n_frames and im.size == (720, 450)


def test_real_telemetry_rates_are_around_four_to_five_hertz(spa):
    """The assumption behind the 10 Hz documentation (position ~220 ms, car data ~240 ms). Unverified when written."""
    rates = spa.telemetry_rates()
    assert 3.0 < rates["position_hz"] < 6.0 and 3.0 < rates["car_hz"] < 6.0
    assert 6.0 < rates["merged_hz"] < 12.0  # ~8.7 Hz: the sum of 1/0.22 s and 1/0.24 s


def test_parquet_dataset_of_a_real_lap_at_10_hz(spa, tmp_path):
    """Not yet run against the real API when this was written; thresholds may need adjusting."""
    import polars as pl

    ds = spa.write_sequence_parquet_for_laps(tmp_path / "lap.ds", LAP, hz=10, chunk_frames=300)
    assert ds.step_s == pytest.approx(0.1) and ds.n_frames > 600 and len(ds.manifest["chunks"]) >= 2
    report = ds.continuity_report()
    assert report.ok, report.summary()
    assert (
        ds.scan_drivers().select(pl.col("pos_age_s").max()).collect().item() < 1.0
    )  # every value is backed by a recent sample
    i = ds.n_frames // 3
    seq = ds.read_sequence(i, i + 1)
    ref = GraphExporter().export(spa.build_graph_at_time(float(seq.times[0])))
    got = seq.frame(0)
    for key, value in ref.to_dict().items():
        assert np.array_equal(value, got.to_dict()[key], equal_nan=True), key
