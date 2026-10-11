"""End-to-end: session -> graphs -> arrays -> files, and the properties a training dataset relies on."""

from __future__ import annotations

import numpy as np
import pytest

from f1strategy import GraphExporter, SessionAnalyser
from f1strategy.graph.exporter import DRIVER_FEATURES, GLOBAL_FEATURES, TRACK_FEATURES

TIMES = np.arange(35.0, 380.0, 12.0)


def test_every_snapshot_has_the_identical_schema_and_stacks_into_a_dataset(analyser):
    exporter = GraphExporter(normalise=True)
    snaps = [exporter.export(analyser.build_graph_at_time(float(t))) for t in TIMES]
    for field, width in (
        ("track_x", len(TRACK_FEATURES)),
        ("driver_x", len(DRIVER_FEATURES)),
        ("global_x", len(GLOBAL_FEATURES)),
    ):
        stacked = np.stack([getattr(s, field) for s in snaps])
        assert stacked.shape[0] == len(TIMES) and stacked.shape[-1] == width and np.isfinite(stacked).all()
    assert all(s.driver_codes[0] == "VER" for s in snaps)  # the synthetic leader never loses the lead


def test_exports_are_deterministic(analyser):
    exporter = GraphExporter()
    for t in (100.0, 184.3, 300.0):
        a, b = exporter.export(analyser.build_graph_at_time(t)), exporter.export(analyser.build_graph_at_time(t))
        for k, v in a.to_dict().items():
            assert np.array_equal(v, b.to_dict()[k]), k


def test_two_independent_analysers_agree(syn, analyser):
    other = SessionAnalyser.from_session(syn.session, drs_detection_offset_m=200.0)
    a, b = (
        GraphExporter().export(analyser.build_graph_at_time(150.0)),
        GraphExporter().export(other.build_graph_at_time(150.0)),
    )
    for k, v in a.to_dict().items():
        assert np.array_equal(v, b.to_dict()[k]), k


def test_race_control_timeline_is_visible_in_the_exported_features(analyser):
    expect = {  # time -> (is_green, is_yellow, is_sc, is_vsc, is_vsc_ending, drs_enabled)
        50.0: (1, 0, 0, 0, 0, 0),
        110.0: (0, 1, 0, 0, 0, 1),
        150.0: (1, 0, 0, 0, 0, 1),
        200.0: (0, 0, 1, 0, 0, 0),
        260.0: (1, 0, 0, 0, 0, 1),
        340.0: (0, 0, 0, 1, 0, 1),
        356.0: (0, 0, 0, 0, 1, 1),
    }
    names = ["is_green", "is_yellow", "is_sc", "is_vsc", "is_vsc_ending", "drs_enabled"]
    for t, flags in expect.items():
        a = GraphExporter().export(analyser.build_graph_at_time(t))
        got = tuple(int(a.global_x[0, a.global_names.index(n)]) for n in names)
        assert got == flags, (t, got)


def test_dataset_roundtrip_through_npz(analyser, tmp_path):
    exporter = GraphExporter(normalise=True)
    snaps = [exporter.export(analyser.build_graph_at_time(float(t))).pool_track(5) for t in TIMES[:8]]
    path = tmp_path / "ds.npz"
    np.savez_compressed(path, track_x=np.stack([s.track_x for s in snaps]), driver_x=np.stack([s.driver_x for s in snaps]),
                        times=np.array([s.time_s for s in snaps]))  # fmt: skip
    back = np.load(path)
    assert back["track_x"].shape == (8, 100, len(TRACK_FEATURES)) and np.allclose(back["times"], TIMES[:8])
    assert np.array_equal(back["driver_x"][3], snaps[3].driver_x)


def test_the_leader_crosses_each_sector_where_the_snapshot_says(analyser):
    """build_session_graph(lap, sector) puts the leader on the sector boundary of the track model."""
    tr = analyser.track
    for sector, boundary in (("Sector 1", tr.sector_end_s[0]), ("Sector 2", tr.sector_end_s[1])):
        for lap in (2, 3, 4):
            G = analyser.build_session_graph(lap, sector)
            leader = G.graph["driver_nodes"][0]
            assert abs(tr.circular_delta(boundary, leader.track_s_m)) < 20.0, (sector, lap)
            assert leader.position == 1


@pytest.mark.parametrize("spacing", [5.0, 10.0, 25.0])
def test_pipeline_runs_at_other_resolutions(syn, spacing):
    a = SessionAnalyser.from_session(syn.session, track_spacing_m=spacing)
    arr = GraphExporter().export(a.build_graph_at_time(150.0))
    assert abs(arr.track_spacing_m - spacing) < spacing * 0.02 and len(arr.track_x) == a.track.n
