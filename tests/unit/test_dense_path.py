"""The dense (graph-free) path, telemetry ages / native rates, 10 Hz sampling and the new sequence operations."""

from __future__ import annotations

import numpy as np
import pytest

from f1strategy import GraphExporter, RaceSequence, SessionAnalyser, time_grid
from f1strategy.domain.driver import DriverNode
from f1strategy.graph.exporter import (
    COMPOUNDS,
    DRIVER_FEATURES,
    EDGE_FEATURES,
    GLOBAL_FEATURES,
    TRACK_DYNAMIC,
    TRACK_FEATURES,
    pack_flags,
)
from f1strategy.sequence.sampling import resolve_step
from tests.synthetic.session import make_synthetic_session

TIMES = [40.0, 96.0, 121.0, 184.3, 235.0, 300.0, 340.0, 356.0, 375.0]


class TestSharedRowBuilders:
    """Both paths call the same row functions; pin them to the schema by *name*, not by position."""

    def _node(self):
        return DriverNode(
            "AAA", 3, "hard", 7, 1, 91.5, x_m=11.0, y_m=22.0, z_m=3.0, speed_kmh=250.0, vx_mps=1.0, vy_mps=2.0,
            heading_rad=0.5, heading_valid=True, heading_rel_track_rad=0.25, throttle=80.0, brake=0.0, gear=6.0,
            rpm=11000.0, drs_open=True, drs_eligible=False, track_node=5, track_s_m=1234.0, lateral_offset_m=-1.5,
            progress_m=9876.0, lap_number=4, lap_fraction=0.25, on_track=True, pos_age_s=0.07, car_age_s=0.13,
            gap_to_leader_s=4.5, gap_ahead_s=1.5, gap_behind_s=0.5, official_position=3.0, last_lap_time_s=90.1,
            fresh_tyre=True, stint=2, in_pit=False, laps_completed=3, finished_or_retired=False,
        )  # fmt: skip

    def test_driver_row_values_land_in_the_columns_named_by_the_schema(self):
        row = dict(zip(DRIVER_FEATURES, GraphExporter.driver_row(self._node().to_attributes()), strict=True))
        expected = {
            "x_m": 11.0, "y_m": 22.0, "speed_kmh": 250.0, "throttle": 80.0, "gear": 6.0, "rpm": 11000.0, "drs_open": 1.0,
            "drs_eligible": 0.0, "track_s_m": 1234.0, "lap_fraction": 0.25, "lap_fraction_sin": 1.0, "lap_fraction_cos": 0.0,
            "lateral_offset_m": -1.5, "progress_m": 9876.0, "lap_number": 4.0, "position": 3.0, "pos_age_s": 0.07,
            "car_age_s": 0.13, "gap_to_leader_s": 4.5, "gap_ahead_s": 1.5, "gap_behind_s": 0.5, "tyre_age": 7.0,
            "fresh_tyre": 1.0, "stint": 2.0, "pit_stops": 1.0, "in_pit": 0.0, "on_track": 1.0, "pace_last3_s": 91.5,
            "last_lap_time_s": 90.1, "finished_or_retired": 0.0, "compound_hard": 1.0, "compound_soft": 0.0,
            "compound_unknown": 0.0, "official_position": 3.0,
        }  # fmt: skip
        for name, value in expected.items():
            assert row[name] == pytest.approx(value, abs=1e-6), name
        assert row["heading_sin"] == pytest.approx(np.sin(0.5)) and row["heading_rel_track_cos"] == pytest.approx(
            np.cos(0.25)
        )
        assert row["vx_mps"] == 1.0 and row["vy_mps"] == 2.0

    def test_row_lengths_equal_the_schema_lengths(self, G_sc):
        node = self._node().to_attributes()
        assert len(GraphExporter.driver_row(node)) == len(DRIVER_FEATURES)
        assert len(GraphExporter.global_row(G_sc.graph["global"])) == len(GLOBAL_FEATURES)
        edge = next(e for *_, e in G_sc.edges(data=True) if e["edge_type"] == "proximity")
        assert len(GraphExporter.edge_row(edge)) == len(EDGE_FEATURES)

    def test_every_compound_has_exactly_one_flag_in_the_row(self):
        for c in [*COMPOUNDS, "mystery"]:
            node = self._node().to_attributes()
            node["compound"] = c
            row = dict(zip(DRIVER_FEATURES, GraphExporter.driver_row(node), strict=True))
            flags = [row[f"compound_{k.lower()}"] for k in COMPOUNDS] + [row["compound_unknown"]]
            assert sum(flags) == 1

    def test_to_attributes_covers_every_key_the_row_reads(self):
        attrs = self._node().to_attributes()
        GraphExporter.driver_row(attrs)  # would raise KeyError if DriverNode lost an attribute the exporter needs


class TestLightPathEqualsGraphPath:
    @pytest.mark.parametrize("t", TIMES)
    def test_snapshot_describes_the_same_race_as_the_graph(self, analyser, t):
        snap, G = analyser.graph_builder.snapshot(t), analyser.graph_builder.build(t)
        assert [d.driver_code for d in snap.drivers] == G.graph["driver_codes"]
        for key, value in snap.global_features.items():
            other = G.graph["global"][key]
            assert value == other or (isinstance(value, float) and np.isnan(value) and np.isnan(other)), key
        edges = {(snap.drivers[i].driver_code, snap.drivers[j].driver_code) for i, j, _ in snap.edges}
        assert edges == {(u, v) for u, v, e in G.edges(data=True) if e["edge_type"] == "proximity"}

    @pytest.mark.parametrize("t", TIMES)
    def test_records_equal_the_exported_graph_arrays(self, analyser, t):
        rec = analyser.sequence_builder.record(t)
        ref = GraphExporter().export(analyser.build_graph_at_time(t))
        assert rec.driver_codes == ref.driver_codes and rec.time_s == ref.time_s
        assert np.array_equal(rec.driver_x, ref.driver_x) and np.array_equal(rec.driver_mask, ref.driver_mask)
        assert np.array_equal(rec.global_x, ref.global_x[0]) and np.array_equal(rec.global_mask, ref.global_mask[0])
        assert np.array_equal(rec.driver_to_track, ref.driver_to_track)
        assert rec.race_distance_m == ref.race_distance_m
        order = np.lexsort((rec.edge_dst, rec.edge_src))  # the graph lists edges per source; compare as sets of rows
        ref_order = np.lexsort((ref.driver_edge_index[1], ref.driver_edge_index[0]))
        assert np.array_equal(rec.edge_src[order], ref.driver_edge_index[0][ref_order])
        assert np.array_equal(rec.edge_attr[order], ref.driver_edge_attr[ref_order])

    def test_flags_from_the_light_path_equal_the_graph_exporter_columns(self, analyser):
        for t in TIMES:
            rec = analyser.sequence_builder.record(t)
            ref = GraphExporter().export(analyser.build_graph_at_time(t))
            for k, name in enumerate(
                ["flag_yellow", "flag_double_yellow", "flag_sc", "flag_vsc", "flag_vsc_ending", "flag_red"]
            ):
                assert np.array_equal(
                    (rec.track_flags >> k) & 1, ref.track_x[:, ref.track_names.index(name)].astype(int)
                ), (t, name)

    def test_static_track_equals_the_graph_exporters_static_columns(self, analyser):
        static = analyser.sequence_builder.static_track
        ref = GraphExporter().export(analyser.build_graph_at_time(100.0))
        cols = [ref.track_names.index(n) for n in static.names]
        assert np.array_equal(static.values, ref.track_x[:, cols]) and np.array_equal(
            static.mask, ref.track_mask[:, cols]
        )
        assert static.xy_bbox == ref.xy_bbox and static.length_m == ref.track_length_m
        assert static.names == [n for n in TRACK_FEATURES if n not in TRACK_DYNAMIC]

    def test_pack_flags(self):
        class RC:
            is_safety_car, is_vsc, is_vsc_ending, is_red = True, False, True, False

        y, d = np.array([True, False, False]), np.array([False, True, False])
        assert pack_flags(y, d, RC).tolist() == [0b010101, 0b010110, 0b010100]

    def test_build_chunks_concatenate_to_build(self, analyser):
        times = time_grid(100.0, 130.0, hz=4)
        whole = analyser.sequence_builder.build(times)
        chunks = list(analyser.sequence_builder.build_chunks(times, chunk_frames=17))
        assert len(times) == 121 and [c.n_frames for c in chunks] == [17] * 7 + [2]
        joined = RaceSequence.concat(chunks)
        for name in (
            "times",
            "track_flags",
            "driver_x",
            "driver_mask",
            "driver_to_track",
            "global_x",
            "edge_attr",
            "edge_exists",
        ):
            assert np.array_equal(getattr(joined, name), getattr(whole, name), equal_nan=True), name
        assert joined.step_s == whole.step_s == 0.25

    def test_chunks_share_driver_slots_and_the_nominal_step(self, analyser):
        chunks = list(analyser.sequence_builder.build_chunks(time_grid(100.0, 110.0, 1.0), chunk_frames=3))
        assert [c.n_frames for c in chunks] == [3, 3, 3, 2]
        assert all(c.driver_codes == chunks[0].driver_codes and c.step_s == 1.0 for c in chunks)  # even a 2-frame chunk

    def test_chunk_size_is_validated(self, analyser):
        with pytest.raises(ValueError, match="chunk_frames"):
            list(analyser.sequence_builder.build_chunks([1.0, 2.0], chunk_frames=0))

    def test_driver_slots_are_fixed_up_front(self, analyser):
        assert analyser.sequence_builder.driver_slots() == sorted(analyser._driver_map.values())

    def test_the_graph_path_is_still_independent_and_consistent(self, analyser):
        G = analyser.build_graph_at_time(150.0)
        assert (
            G.graph["driver_nodes"][0].nearby_drivers
            == analyser.graph_builder.snapshot(150.0).drivers[0].nearby_drivers
        )


class TestSequenceOperations:
    def test_decimate_keeps_every_nth_frame_and_scales_the_step(self, sequence):
        d = sequence.decimate(5)
        assert d.n_frames == -(-sequence.n_frames // 5) and d.step_s == 5.0
        assert np.array_equal(d.times, sequence.times[::5]) and np.shares_memory(d.driver_x, sequence.driver_x)
        with pytest.raises(ValueError):
            sequence.decimate(0)

    def test_strided_slice(self, sequence):
        s = sequence.slice(10, 50, 4)
        assert s.n_frames == 10 and s.times.tolist() == sequence.times[10:50:4].tolist() and s.step_s == 4.0

    def test_slice_of_an_irregular_sequence_keeps_step_unknown(self, sequence):
        irregular = sequence.slice(0, 20)
        irregular.step_s = None
        assert irregular.slice(0, 10, 2).step_s is None

    def test_concat_restores_the_original_and_the_step(self, sequence):
        joined = RaceSequence.concat([sequence.slice(0, 100), sequence.slice(100, 250), sequence.slice(250, 341)])
        assert joined.step_s == 1.0 and np.array_equal(joined.driver_x, sequence.driver_x, equal_nan=True)

    def test_concat_of_a_gap_has_no_nominal_step(self, sequence):
        assert RaceSequence.concat([sequence.slice(0, 10), sequence.slice(20, 30)]).step_s is None

    def test_concat_rejects_mixed_sessions(self, sequence):
        a, b = sequence.slice(0, 5), sequence.slice(10, 15)
        b.driver_codes = b.driver_codes[::-1]
        with pytest.raises(ValueError, match="different drivers"):
            RaceSequence.concat([a, b])
        c = sequence.slice(20, 25)
        c.track_static = c.track_static + 1
        with pytest.raises(ValueError, match="different tracks"):
            RaceSequence.concat([a, c])
        with pytest.raises(ValueError, match="nothing"):
            RaceSequence.concat([])

    def test_from_records_needs_at_least_one_record(self, analyser):
        with pytest.raises(ValueError, match="at least one frame"):
            RaceSequence.from_records([], analyser.sequence_builder.static_track)


class TestTelemetryAges:
    def test_age_is_zero_exactly_at_a_real_sample_and_grows_until_the_next(self, analyser, syn):
        tp = analyser.telemetry._pos["VER"]["t"]
        k = 500
        at = analyser.telemetry.sample("VER", float(tp[k]))
        assert at.pos_age_s == pytest.approx(0.0, abs=1e-9)
        mid = analyser.telemetry.sample("VER", float((tp[k] + tp[k + 1]) / 2))
        assert mid.pos_age_s == pytest.approx((tp[k + 1] - tp[k]) / 2)
        just_before_next = analyser.telemetry.sample("VER", float(tp[k + 1] - 1e-6))
        assert just_before_next.pos_age_s == pytest.approx(tp[k + 1] - tp[k], abs=1e-4)

    def test_car_age_follows_the_car_data_clock(self, analyser):
        tc = analyser.telemetry._car["NOR"]["t"]
        s = analyser.telemetry.sample("NOR", float(tc[300]) + 0.05)
        assert s.car_age_s == pytest.approx(0.05, abs=1e-9)

    def test_ages_are_nan_before_the_first_sample_and_grow_after_the_last(self, analyser):
        first = analyser.telemetry._pos["VER"]["t"][0]
        assert np.isnan(analyser.telemetry.sample("VER", float(first) - 1.0).pos_age_s)
        last = analyser.telemetry._pos["VER"]["t"][-1]
        assert analyser.telemetry.sample("VER", float(last) + 5.0).pos_age_s == pytest.approx(
            5.0
        )  # stale, and it says so

    def test_age_never_exceeds_one_sampling_interval_while_data_flows(self, analyser):
        ages = [analyser.telemetry.sample("LEC", float(t)) for t in np.arange(60.0, 300.0, 0.1)]
        assert max(a.pos_age_s for a in ages) < 0.25 and max(a.car_age_s for a in ages) < 0.26

    def test_at_10_hz_the_mean_age_is_about_half_an_interval(self, analyser):
        ages = np.array([analyser.telemetry.sample("HAM", float(t)).pos_age_s for t in np.arange(60.0, 300.0, 0.1)])
        assert 0.09 < ages.mean() < 0.13

    def test_missing_car_data_gives_nan_car_age(self, syn):
        from tests.helpers import clone_session

        a = SessionAnalyser.from_session(clone_session(syn.session, car_data={}))
        assert np.isnan(a.telemetry.sample("VER", 150.0).car_age_s) and np.isfinite(
            a.telemetry.sample("VER", 150.0).pos_age_s
        )

    def test_ages_are_causal(self, analyser):
        """The age only looks backwards: it is identical when all data after t is removed."""
        t = 150.0
        full = analyser.telemetry.sample("VER", t)
        tp = analyser.telemetry._pos["VER"]["t"]
        newest = tp[np.searchsorted(tp, t, side="right") - 1]
        assert full.pos_age_s == pytest.approx(t - newest)

    def test_ages_reach_the_graph_nodes_the_exporter_and_the_sequence(self, analyser, sequence):
        G = analyser.build_graph_at_time(150.123)
        node = G.nodes["VER"]
        assert 0 <= node["pos_age_s"] < 0.25 and 0 <= node["car_age_s"] < 0.26
        a = GraphExporter().export(G)
        assert a.driver_x[a.driver_codes.index("VER"), a.driver_names.index("pos_age_s")] == pytest.approx(
            node["pos_age_s"], abs=1e-6
        )
        j = sequence.driver_names.index("pos_age_s")
        assert (sequence.driver_x[:, :, j] >= 0).all() and sequence.driver_x[:, :, j].max() < 0.25

    def test_at_integer_seconds_the_ages_cycle_so_a_model_can_tell_real_from_interpolated(self, analyser):
        seq = analyser.build_sequence(100.0, 110.0, hz=10)
        j = seq.driver_names.index("pos_age_s")
        col = seq.driver_x[:, seq.driver_codes.index("VER"), j]
        assert col.std() > 0.05 and len(np.unique(np.round(col, 3))) > 20  # varies from frame to frame


class TestNativeRates:
    def test_rates_match_the_fastf1_style_clocks(self, analyser):
        rates = analyser.telemetry.native_rates()
        assert set(rates) == set(analyser._driver_map.values())
        for pos, car in rates.values():
            assert pos == pytest.approx(1 / 0.22, rel=0.03) and car == pytest.approx(1 / 0.24, rel=0.03)

    def test_summary_over_all_drivers(self, analyser):
        r = analyser.telemetry_rates()
        assert r["position_hz"] == pytest.approx(4.55, abs=0.15) and r["car_hz"] == pytest.approx(4.17, abs=0.1)
        assert r["merged_hz"] == pytest.approx(r["position_hz"] + r["car_hz"])

    def test_missing_car_stream_gives_nan_rate(self, syn):
        from tests.helpers import clone_session

        a = SessionAnalyser.from_session(clone_session(syn.session, car_data={}))
        assert np.isnan(a.telemetry_rates()["car_hz"]) and np.isfinite(a.telemetry_rates()["position_hz"])


@pytest.fixture(scope="module")
def lap10(analyser):
    return analyser.build_sequence_for_laps(3, hz=10)


class TestTenHertz:
    def test_ten_frames_per_second(self, lap10, analyser):
        start, end = analyser.graph_builder.lap_window(3)
        assert lap10.step_s == pytest.approx(0.1) and abs(lap10.n_frames - (end - start) * 10) <= 1.5
        assert np.allclose(np.diff(lap10.times), 0.1)

    def test_ten_hz_data_is_continuous(self, lap10):
        rep = lap10.continuity_report()
        assert rep.ok and 0.9 < rep.max_position_step_ratio < 1.2, rep.summary()

    def test_positions_move_smoothly_between_ten_hz_frames(self, lap10):
        names = lap10.driver_names
        x, y, v = (lap10.driver_x[:, 0, names.index(n)] for n in ("x_m", "y_m", "speed_kmh"))
        step = np.hypot(np.diff(x), np.diff(y))
        assert step.max() < (v.max() / 3.6) * 0.1 * 1.3 + 1.0  # never further than speed x dt
        assert np.abs(np.diff(v)).max() < 25.0  # speed changes by at most a few percent per 0.1 s

    def test_most_frames_between_real_samples_are_interpolated(self, lap10):
        j = lap10.driver_names.index("pos_age_s")
        age = lap10.driver_x[:, :, j]
        assert (age > 0.01).mean() > 0.9  # a 10 Hz frame rarely coincides with a real 4.5 Hz sample

    def test_hz_and_step_are_equivalent(self, analyser):
        a = analyser.build_sequence(100.0, 105.0, hz=4)
        b = analyser.build_sequence(100.0, 105.0, step_s=0.25)
        assert np.array_equal(a.driver_x, b.driver_x, equal_nan=True) and a.step_s == b.step_s == 0.25

    def test_frame_at_10_hz_still_equals_the_graph_exporter(self, analyser, lap10):
        for i in (0, 123, 400, lap10.n_frames - 1):
            ref = GraphExporter().export(analyser.build_graph_at_time(float(lap10.times[i])))
            got = lap10.frame(i)
            for k, v in ref.to_dict().items():
                assert np.array_equal(v, got.to_dict()[k], equal_nan=True), (i, k)


class TestResolveStep:
    @pytest.mark.parametrize(
        "step, hz, expected", [(None, None, 1.0), (0.5, None, 0.5), (None, 10, 0.1), (None, 4, 0.25), (None, 0.5, 2.0)]
    )
    def test_values(self, step, hz, expected):
        assert resolve_step(step, hz) == pytest.approx(expected)

    def test_default_can_be_changed(self):
        assert resolve_step(default_s=5.0) == 5.0

    def test_both_is_an_error_and_hz_must_be_positive(self):
        with pytest.raises(ValueError, match="either"):
            resolve_step(1.0, 10)
        for bad in (0, -3):
            with pytest.raises(ValueError, match="hz"):
                resolve_step(None, bad)

    def test_time_grid_takes_hz(self):
        g = time_grid(0.0, 1.0, hz=10)
        assert len(g) == 11 and g[1] == pytest.approx(0.1)
        with pytest.raises(ValueError):
            time_grid(0.0, 1.0, step_s=0.1, hz=10)


class TestRealisticScale:
    """20 cars like a real grid; slower, so just one test."""

    def test_twenty_car_grid_runs_on_the_dense_path_and_matches_the_graph_path(self):
        a = SessionAnalyser.from_session(make_synthetic_session(n_drivers=20).session)
        seq = a.build_sequence(100.0, 110.0, hz=10)
        assert seq.n_drivers == 20 and seq.continuity_report().ok
        ref = GraphExporter().export(a.build_graph_at_time(float(seq.times[50])))
        got = seq.frame(50)
        assert ref.driver_codes == got.driver_codes and len(got.driver_codes) == 20
        for k, v in ref.to_dict().items():
            assert np.array_equal(v, got.to_dict()[k], equal_nan=True), k
