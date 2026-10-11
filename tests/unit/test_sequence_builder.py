"""SequenceBuilder and the SessionAnalyser entry points for continuous data."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from f1strategy import SequenceBuilder, SessionAnalyser, time_grid
from f1strategy.animation import team_colors


class TestSequenceBuilder:
    def test_builds_exactly_the_requested_frames(self, analyser):
        times = time_grid(100.0, 110.0, 2.5)
        seq = analyser.sequence_builder.build(times)
        assert seq.times.tolist() == times.tolist() and seq.step_s == 2.5

    def test_progress_callback_reports_every_frame(self, analyser):
        calls = []
        analyser.build_sequence(100.0, 104.0, 1.0, progress=lambda done, total: calls.append((done, total)))
        assert calls == [(k, 5) for k in range(1, 6)]

    @pytest.mark.parametrize("bad", [[], [[1.0, 2.0]], [5.0, 4.0], [1.0, 1.0]])
    def test_invalid_times_are_rejected(self, analyser, bad):
        with pytest.raises(ValueError):
            analyser.sequence_builder.build(bad)

    def test_irregular_times_are_accepted_but_have_no_nominal_step(self, analyser):
        seq = analyser.sequence_builder.build([100.0, 101.0, 103.5])
        assert seq.step_s is None and seq.n_frames == 3

    def test_single_frame_sequence(self, analyser):
        seq = analyser.build_sequence(150.0, 150.0)
        assert seq.n_frames == 1 and seq.step_s is None and seq.continuity_report().ok

    def test_building_is_stateless_a_slice_equals_a_separate_build(self, analyser, sequence):
        part = analyser.build_sequence(float(sequence.times[10]), float(sequence.times[29]), 1.0)
        mine = sequence.slice(10, 30)
        for name in ("times", "track_flags", "driver_x", "driver_mask", "driver_to_track", "global_x", "edge_attr"):
            assert np.array_equal(getattr(part, name), getattr(mine, name), equal_nan=True), name

    def test_deterministic(self, analyser):
        a, b = analyser.build_sequence(100.0, 120.0), analyser.build_sequence(100.0, 120.0)
        assert np.array_equal(a.driver_x, b.driver_x, equal_nan=True) and np.array_equal(a.track_flags, b.track_flags)

    def test_builder_exports_raw_values_regardless_of_normalisation_elsewhere(self, analyser):
        seq = analyser.build_sequence(100.0, 101.0)
        j = seq.driver_names.index("speed_kmh")
        assert seq.driver_x[:, :, j].max() > 100  # km/h, not scaled to 0..1


class TestAnalyserEntryPoints:
    def test_build_sequence_for_laps_covers_the_lap_window(self, analyser):
        start, end = analyser.graph_builder.lap_window(2, 3)
        seq = analyser.build_sequence_for_laps(2, 3, step_s=0.5)
        assert seq.times[0] == pytest.approx(start) and end - 0.5 < seq.times[-1] <= end + 1e-9 and seq.step_s == 0.5

    def test_default_last_lap_is_the_first_lap(self, analyser):
        seq = analyser.build_sequence_for_laps(3)
        assert seq.times[-1] <= analyser.graph_builder.lap_window(3)[1] + 1e-9 and len(seq) in (57, 58, 59)

    def test_sequences_of_consecutive_laps_join_without_a_gap(self, analyser):
        a, b = analyser.build_sequence_for_laps(3), analyser.build_sequence_for_laps(4)
        assert b.times[0] - a.times[-1] < 1.0 + 1e-9

    def test_unknown_lap_raises_clearly(self, analyser):
        with pytest.raises(ValueError, match="No data for lap"):
            analyser.build_sequence_for_laps(99)

    def test_reversed_range_raises(self, analyser):
        with pytest.raises(ValueError, match="before"):
            analyser.build_sequence(200.0, 100.0)

    def test_iter_graphs_is_lazy_and_yields_networkx_graphs(self, analyser):
        gen = analyser.iter_graphs(time_grid(100.0, 103.0, 1.0))
        assert iter(gen) is gen  # a generator: nothing is built until it is consumed
        graphs = list(gen)
        assert len(graphs) == 4 and all(isinstance(g, nx.DiGraph) for g in graphs)
        assert [g.graph["time_s"] for g in graphs] == [100.0, 101.0, 102.0, 103.0]

    def test_iter_graphs_equals_the_point_by_point_api(self, analyser):
        (g,) = list(analyser.iter_graphs([184.3]))
        ref = analyser.build_graph_at_time(184.3)
        assert set(g.nodes) == set(ref.nodes) and set(g.edges) == set(ref.edges)
        for n, attrs in ref.nodes(data=True):
            for k, v in attrs.items():
                assert v == g.nodes[n][k] or (isinstance(v, float) and np.isnan(v) and np.isnan(g.nodes[n][k])), (n, k)

    def test_iter_graphs_validates_times_when_consumed(self, analyser):
        with pytest.raises(ValueError):
            list(analyser.iter_graphs([5.0, 4.0]))

    def test_point_by_point_snapshots_still_work_next_to_sequences(self, analyser):
        G = analyser.build_session_graph(3, "Sector 2")
        seq = analyser.build_sequence(G.graph["time_s"], G.graph["time_s"] + 3.0, 1.0)
        first = seq.frame(0)
        assert first.time_s == G.graph["time_s"] and first.driver_codes == G.graph["driver_codes"]


class TestDriverColors:
    def test_fallback_palette_is_stable_and_distinct(self, analyser):
        colors = analyser.driver_colors()
        assert sorted(colors) == sorted(analyser._driver_map.values())
        assert len(set(colors.values())) == len(colors) and analyser.driver_colors() == colors

    def test_team_colors_are_used_when_the_results_table_has_them(self):
        res = pd.DataFrame({"Abbreviation": ["VER", "NOR", "XXX"], "TeamColor": ["3671C6", "#FF8000", "zzzzzz"]})
        colors = team_colors(SimpleNamespace(results=res), ["VER", "NOR", "XXX", "LEC"])
        assert colors["VER"] == "#3671C6" and colors["NOR"] == "#FF8000"
        assert colors["XXX"] != "#zzzzzz" and colors["XXX"].startswith("#") and len(colors["LEC"]) == 7

    def test_unknown_abbreviations_in_results_are_ignored(self):
        res = pd.DataFrame({"Abbreviation": ["ZZZ"], "TeamColor": ["112233"]})
        assert "ZZZ" not in team_colors(SimpleNamespace(results=res), ["AAA"])

    @pytest.mark.parametrize(
        "session", [SimpleNamespace(), SimpleNamespace(results=pd.DataFrame({"Abbreviation": ["A"]}))]
    )
    def test_missing_results_or_column_fall_back(self, session):
        assert team_colors(session, ["A", "B"]).keys() == {"A", "B"}


def test_sequence_builder_is_exposed_on_the_analyser(analyser):
    assert (
        isinstance(analyser.sequence_builder, SequenceBuilder)
        and analyser.sequence_builder.graph_builder is analyser.graph_builder
    )
    assert isinstance(analyser, SessionAnalyser)


def test_every_chunk_size_default_is_the_single_constant():
    """One constant, referenced everywhere, so the defaults cannot drift apart."""
    import inspect

    from f1strategy import SessionAnalyser
    from f1strategy.cli import build_parser, run_continuous
    from f1strategy.sequence import ParquetSequence, RaceSequence, SequenceBuilder, write_dataset
    from f1strategy.sequence.sampling import DEFAULT_CHUNK_FRAMES

    assert DEFAULT_CHUNK_FRAMES == 2000
    targets = {
        SequenceBuilder.build_chunks: "chunk_frames",
        write_dataset: "chunk_frames",
        RaceSequence.to_parquet: "chunk_frames",
        ParquetSequence.iter_chunks: "frames_per_chunk",
        SessionAnalyser.write_sequence_parquet: "chunk_frames",
        SessionAnalyser.write_sequence_parquet_for_laps: "chunk_frames",
        run_continuous: "chunk_frames",
    }
    for fn, param in targets.items():
        assert inspect.signature(fn).parameters[param].default == DEFAULT_CHUNK_FRAMES, fn.__qualname__
    assert build_parser().parse_args([]).chunk_frames == DEFAULT_CHUNK_FRAMES
