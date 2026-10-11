"""RaceSequence: packing, exact frame reconstruction, windows, persistence and the continuity checks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from f1strategy import DriverTiming, GraphExporter, RaceGraphBuilder, RaceSequence
from f1strategy.sequence.dataset import FLAG_BITS, FORMAT_VERSION

CHECK_TIMES = [40.0, 96.0, 110.0, 121.0, 130.0, 184.0, 200.0, 235.0, 250.0, 300.0, 340.0, 356.0, 370.0]


def _equal(a, b, *, close=False):
    da, db = a.to_dict(), b.to_dict()
    assert da.keys() == db.keys()
    for k in da:
        same = (
            np.allclose(da[k], db[k], atol=1e-6, equal_nan=True)
            if close
            else np.array_equal(da[k], db[k], equal_nan=True)
        )
        assert same, k
    assert a.driver_codes == b.driver_codes and a.track_names == b.track_names and a.driver_names == b.driver_names


class TestFrameReconstruction:
    """frame(i) must be *exactly* what the exporter produces for the same time (single source of truth)."""

    @pytest.mark.parametrize("t", CHECK_TIMES)
    def test_raw_frame_equals_exporter_output(self, analyser, sequence, t):
        i = int(np.argmin(abs(sequence.times - t)))
        ref = GraphExporter().export(analyser.build_graph_at_time(float(sequence.times[i])))
        _equal(sequence.frame(i), ref)

    @pytest.mark.parametrize("t", [110.0, 190.0, 300.0])
    def test_normalised_frame_equals_normalising_exporter(self, analyser, sequence, t):
        i = int(np.argmin(abs(sequence.times - t)))
        ref = GraphExporter(normalise=True).export(analyser.build_graph_at_time(float(sequence.times[i])))
        _equal(sequence.frame(i, normalise=True), ref, close=True)

    def test_pooled_frame_equals_pooled_exporter_output(self, analyser, sequence):
        i = 100
        ref = GraphExporter(normalise=True).export(analyser.build_graph_at_time(float(sequence.times[i]))).pool_track(5)
        _equal(sequence.frame(i, normalise=True, pool=5), ref, close=True)

    def test_every_frame_of_the_race_is_exact(self, analyser, sequence):
        ex = GraphExporter()
        for i in range(0, sequence.n_frames, 7):
            _equal(sequence.frame(i), ex.export(analyser.build_graph_at_time(float(sequence.times[i]))))

    def test_negative_index_and_bounds(self, sequence):
        _equal(sequence.frame(-1), sequence.frame(sequence.n_frames - 1))
        for bad in (sequence.n_frames, -sequence.n_frames - 1):
            with pytest.raises(IndexError):
                sequence.frame(bad)

    def test_frames_iterator(self, sequence):
        got = list(sequence.slice(0, 3).frames())
        assert [g.time_s for g in got] == sequence.times[:3].tolist()

    def test_a_driver_missing_from_some_frames_is_marked_absent(self, analyser, syn, sequence):
        timing = DriverTiming(syn.session.laps[syn.session.laps["Driver"] != "ALO"])
        partial = RaceGraphBuilder(analyser.session, analyser.track, analyser.telemetry, timing, analyser.timeline,
                                   analyser._driver_map)  # fmt: skip
        ex = GraphExporter()
        times = [100.0, 101.0, 102.0]
        frames = [ex.export(analyser.build_graph_at_time(100.0)), ex.export(partial.build(101.0)),
                  ex.export(analyser.build_graph_at_time(102.0))]  # fmt: skip
        seq = RaceSequence.from_frames(frames)
        alo = seq.driver_codes.index("ALO")
        assert seq.driver_present[:, alo].tolist() == [True, False, True] and seq.times.tolist() == times
        _equal(seq.frame(1), frames[1])
        assert "ALO" not in seq.frame(1).driver_codes and len(seq.frame(0).driver_codes) == 6


class TestLayout:
    def test_shapes_and_dtypes(self, sequence, analyser):
        T, D, N = sequence.n_frames, 6, analyser.track.n
        assert (sequence.n_drivers, sequence.n_track, len(sequence)) == (D, N, T)
        assert sequence.track_flags.shape == (T, N) and sequence.track_flags.dtype == np.uint8
        assert sequence.driver_x.shape[:2] == (T, D) and sequence.edge_attr.shape[:3] == (T, D, D)
        assert sequence.edge_exists.dtype == bool and sequence.driver_to_track.dtype == np.int32
        assert sequence.step_s == 1.0 and sequence.validate() is sequence

    def test_drivers_keep_their_slot_for_the_whole_race(self, sequence):
        assert sequence.driver_codes == sorted(sequence.driver_codes)
        j = sequence.driver_names.index("position")
        for slot, code in enumerate(sequence.driver_codes):
            assert sequence.driver_present[:, slot].all()
            assert sequence.driver_x[:, slot, j].min() >= 1 and sequence.driver_x[:, slot, j].max() <= 6
        # the leader's slot is always VER here, so its position is 1 throughout
        assert (sequence.driver_x[:, sequence.driver_codes.index("VER"), j] == 1).all()

    def test_driver_order_is_leader_first(self, sequence):
        for i in (0, 120, 250):
            order = sequence.driver_order(i)
            pos = sequence.driver_x[i, order, sequence.driver_names.index("position")]
            assert pos.tolist() == sorted(pos.tolist()) and len(order) == 6

    @pytest.mark.parametrize("bit, name", list(enumerate(FLAG_BITS)))
    def test_flag_bits_round_trip(self, analyser, sequence, bit, name):
        ex = GraphExporter()
        for t in (121.0, 190.0, 340.0, 356.0, 60.0):
            i = int(np.argmin(abs(sequence.times - t)))
            ref = ex.export(analyser.build_graph_at_time(float(sequence.times[i])))
            unpacked = (sequence.track_flags[i] >> bit) & 1
            assert np.array_equal(unpacked, ref.track_x[:, ref.track_names.index(name)].astype(int))

    def test_flags_are_actually_set_at_the_right_moments(self, sequence):
        def at(t):
            return sequence.track_flags[int(np.argmin(abs(sequence.times - t)))]

        assert (at(60.0) == 0).all()  # green
        assert (((at(190.0) >> FLAG_BITS.index("flag_sc")) & 1) == 1).all()  # SC everywhere
        assert (((at(340.0) >> FLAG_BITS.index("flag_vsc")) & 1) == 1).all()
        yellow = ((at(121.0) >> FLAG_BITS.index("flag_yellow")) & 1).sum()
        assert 0 < yellow < sequence.n_track  # only one marshal sector

    def test_n_drivers_is_derived_not_stored(self, sequence):
        assert "n_drivers" not in sequence.track_static_names
        fr = sequence.frame(50)
        assert fr.track_x[:, fr.track_names.index("n_drivers")].sum() == 6

    def test_static_track_features_are_stored_once(self, sequence):
        assert sequence.track_static.ndim == 2 and "x_m" in sequence.track_static_names
        assert not set(FLAG_BITS) & set(sequence.track_static_names)

    def test_the_compact_layout_is_much_smaller_than_stacking_frames(self, sequence):
        stacked = sum(f.track_x.nbytes + f.driver_x.nbytes + f.driver_edge_attr.nbytes for f in sequence.frames())
        assert sequence.nbytes < stacked / 4

    def test_changing_static_track_data_is_rejected(self, analyser):
        ex = GraphExporter()
        a, b = ex.export(analyser.build_graph_at_time(100.0)), ex.export(analyser.build_graph_at_time(101.0))
        b.track_x[5, b.track_names.index("x_m")] += 1.0  # a different circuit / a corrupted frame
        with pytest.raises(ValueError, match="static track features changed"):
            RaceSequence.from_frames([a, b])

    def test_inputs_are_validated(self, analyser):
        ex = GraphExporter()
        a = ex.export(analyser.build_graph_at_time(100.0))
        with pytest.raises(ValueError, match="at least one frame"):
            RaceSequence.from_frames([])
        with pytest.raises(ValueError, match="raw"):
            RaceSequence.from_frames([GraphExporter(normalise=True).export(analyser.build_graph_at_time(100.0))])
        b = ex.export(analyser.build_graph_at_time(101.0))
        b.driver_names.append("extra")
        with pytest.raises(ValueError, match="schemas"):
            RaceSequence.from_frames([a, b])

    def test_non_increasing_times_are_rejected(self, analyser):
        f = GraphExporter().export(analyser.build_graph_at_time(100.0))
        with pytest.raises(ValueError, match="strictly increasing"):
            RaceSequence.from_frames([f, f])

    def test_irregular_grid_has_no_nominal_step(self, analyser):
        ex = GraphExporter()
        frames = [ex.export(analyser.build_graph_at_time(t)) for t in (100.0, 101.0, 103.5)]
        assert RaceSequence.from_frames(frames).step_s is None
        assert RaceSequence.from_frames(frames, step_s=1.0).step_s == 1.0

    def test_repr_and_nbytes(self, sequence):
        r = repr(sequence)
        assert "341 frames" in r and "6 drivers" in r and "step 1 s" in r and sequence.nbytes > 0

    def test_validate_rejects_corrupted_arrays(self, sequence):
        from dataclasses import replace

        with pytest.raises(ValueError, match="inconsistent RaceSequence"):
            replace(sequence, track_flags=sequence.track_flags[:-1]).validate()
        with pytest.raises(ValueError, match="inconsistent RaceSequence"):
            replace(sequence, driver_to_track=sequence.driver_to_track + 10_000).validate()


class TestWindows:
    def test_slice_is_a_view(self, sequence):
        part = sequence.slice(10, 20)
        assert part.n_frames == 10 and np.shares_memory(part.driver_x, sequence.driver_x)
        assert part.times.tolist() == sequence.times[10:20].tolist()
        _equal(part.frame(0), sequence.frame(10))

    @pytest.mark.parametrize(
        "length, stride, expected", [(10, 1, 332), (10, 10, 34), (341, 1, 1), (342, 1, 0), (1, 100, 4)]
    )
    def test_window_counts(self, sequence, length, stride, expected):
        assert sum(1 for _ in sequence.windows(length, stride)) == expected

    def test_windows_are_consecutive_and_overlap_by_stride(self, sequence):
        w = list(sequence.windows(8, 4))
        assert w[0].times[4] == w[1].times[0] and all(len(x) == 8 for x in w)

    @pytest.mark.parametrize("length, stride", [(0, 1), (5, 0), (-1, 1)])
    def test_invalid_window_arguments(self, sequence, length, stride):
        with pytest.raises(ValueError):
            list(sequence.windows(length, stride))


class TestPersistence:
    def test_round_trip_is_exact(self, sequence, tmp_path):
        path = tmp_path / "s.npz"
        sequence.save(path)
        back = RaceSequence.load(path)
        assert back.driver_codes == sequence.driver_codes and back.step_s == sequence.step_s
        for name in ("times", "track_static", "track_flags", "driver_x", "driver_mask", "driver_present",
                     "driver_to_track", "global_x", "edge_attr", "edge_exists", "xy_bbox"):  # fmt: skip
            assert np.array_equal(getattr(back, name), getattr(sequence, name), equal_nan=True), name
        assert back.track_names == sequence.track_names and back.race_distance_m == sequence.race_distance_m
        _equal(back.frame(77), sequence.frame(77))

    def test_file_is_compressed(self, sequence, tmp_path):
        sequence.save(tmp_path / "s.npz")
        assert (tmp_path / "s.npz").stat().st_size < sequence.nbytes / 3

    def test_loading_never_unpickles(self, sequence, tmp_path):
        sequence.save(tmp_path / "s.npz")
        with np.load(tmp_path / "s.npz", allow_pickle=False) as z:  # would raise if any object array were stored
            assert all(z[k].dtype != object for k in z.files)

    def test_unknown_format_version_is_rejected(self, sequence, tmp_path):
        path = tmp_path / "s.npz"
        sequence.save(path)
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        meta = json.loads(str(data["meta"]))
        meta["format_version"] = FORMAT_VERSION + 1
        data["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(tmp_path / "future.npz", **data)
        with pytest.raises(ValueError, match="unsupported sequence format"):
            RaceSequence.load(tmp_path / "future.npz")

    def test_flag_bit_layout_is_pinned_because_it_is_part_of_the_file_format(self):
        assert FORMAT_VERSION == 1
        assert FLAG_BITS == ["flag_yellow", "flag_double_yellow", "flag_sc", "flag_vsc", "flag_vsc_ending", "flag_red"]

    def test_a_file_with_a_different_flag_layout_is_refused(self, sequence, tmp_path):
        sequence.save(tmp_path / "s.npz")
        with np.load(tmp_path / "s.npz") as z:
            data = {k: z[k] for k in z.files}
        meta = json.loads(str(data["meta"]))
        meta["flag_bits"] = meta["flag_bits"][::-1]
        data["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(tmp_path / "other.npz", **data)
        with pytest.raises(ValueError, match="flag bit layout"):
            RaceSequence.load(tmp_path / "other.npz")

    def test_irregular_and_unknown_race_distance_survive(self, analyser, tmp_path):
        ex = GraphExporter()
        seq = RaceSequence.from_frames([ex.export(analyser.build_graph_at_time(t)) for t in (100.0, 101.0, 103.5)])
        seq.race_distance_m = None
        seq.save(tmp_path / "i.npz")
        back = RaceSequence.load(tmp_path / "i.npz")
        assert back.step_s is None and back.race_distance_m is None


class TestContinuity:
    def test_the_synthetic_race_is_continuous(self, sequence):
        rep = sequence.continuity_report()
        assert rep.ok and rep.position_jumps == [] and rep.progress_regressions == []
        assert 0.8 < rep.max_position_step_ratio < 1.3  # cars move as far as their speed implies, no more
        assert rep.n_frames == sequence.n_frames and "OK" in rep.summary()

    @pytest.mark.parametrize("step", [0.25, 0.5, 2.0, 5.0])
    def test_continuity_holds_for_other_time_steps(self, analyser, step):
        seq = analyser.build_sequence(120.0, 200.0, step)
        assert seq.continuity_report().ok, seq.continuity_report().summary()

    def test_lap_wraps_equal_the_number_of_line_crossings(self, analyser, sequence, syn):
        t0, t1 = sequence.times[0], sequence.times[-1]
        expected = sum(1 for k in syn.truth["lap_end"].values() for tt in k.values() if t0 < tt <= t1)
        assert abs(sequence.continuity_report().lap_wraps - expected) <= 2

    def test_a_teleporting_car_is_detected_with_frame_and_driver(self, sequence):
        bad = sequence.slice(0, 80)
        bad.driver_x = bad.driver_x.copy()
        slot = bad.driver_codes.index("HAM")
        bad.driver_x[30:, slot, bad.driver_names.index("x_m")] += 400.0  # 400 m jump at frame 30
        rep = bad.continuity_report()
        assert not rep.ok and "PROBLEMS" in rep.summary()
        frame, code, step, allowed = rep.position_jumps[0]
        assert (frame, code) == (30, "HAM") and step > 300 and step > allowed

    def test_progress_running_backwards_is_detected(self, sequence):
        bad = sequence.slice(0, 80)
        bad.driver_x = bad.driver_x.copy()
        slot = bad.driver_codes.index("LEC")
        bad.driver_x[40:, slot, bad.driver_names.index("progress_m")] -= 200.0
        rep = bad.continuity_report()
        assert rep.progress_regressions and rep.progress_regressions[0][:2] == (40, "LEC")

    def test_a_single_frame_is_trivially_continuous(self, sequence):
        assert sequence.slice(5, 6).continuity_report().ok

    def test_masked_values_are_ignored(self, sequence):
        bad = sequence.slice(0, 80)
        bad.driver_x = bad.driver_x.copy()
        bad.driver_mask = bad.driver_mask.copy()
        slot, xi = bad.driver_codes.index("ALO"), bad.driver_names.index("x_m")
        bad.driver_x[30, slot, xi] += 900.0
        bad.driver_mask[30, slot, xi] = False  # flagged as missing -> not a teleport
        assert bad.continuity_report().ok

    def test_wrap_safe_lap_features_are_smooth_where_lap_fraction_jumps(self, sequence):
        j = {n: sequence.driver_names.index(n) for n in ("lap_fraction", "lap_fraction_sin", "lap_fraction_cos")}
        slot = sequence.driver_codes.index("VER")
        x = sequence.driver_x[:, slot]
        assert np.abs(np.diff(x[:, j["lap_fraction"]])).max() > 0.9  # the raw feature jumps from ~1 to ~0
        assert np.abs(np.diff(x[:, j["lap_fraction_sin"]])).max() < 0.35  # the encoded one does not
        assert np.abs(np.diff(x[:, j["lap_fraction_cos"]])).max() < 0.35
        assert np.allclose(x[:, j["lap_fraction_sin"]] ** 2 + x[:, j["lap_fraction_cos"]] ** 2, 1.0, atol=1e-4)
