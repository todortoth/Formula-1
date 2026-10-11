"""Parquet datasets: exact round trips, nulls instead of masks, streaming writes, Polars access, error handling."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from f1strategy import ParquetSequence, ParquetSequenceWriter, RaceSequence, write_dataset
from f1strategy.sequence.dataset import FLAG_BITS
from f1strategy.sequence.parquet import MANIFEST, PARQUET_FORMAT

ARRAYS = ("times", "track_static", "track_static_mask", "track_flags", "driver_x", "driver_mask", "driver_present",
          "driver_to_track", "global_x", "global_mask", "edge_attr", "edge_attr_mask", "edge_exists", "xy_bbox")  # fmt: skip


def assert_same(a: RaceSequence, b: RaceSequence):
    for name in ARRAYS:
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name
    assert a.driver_codes == b.driver_codes and a.track_names == b.track_names and a.driver_names == b.driver_names
    assert a.global_names == b.global_names and a.edge_names == b.edge_names
    assert (a.track_length_m, a.track_spacing_m, a.race_distance_m, a.step_s) == (
        b.track_length_m, b.track_spacing_m, b.race_distance_m, b.step_s,
    )  # fmt: skip


@pytest.fixture(scope="module")
def dataset(sequence, tmp_path_factory):
    """The whole synthetic race written in small chunks (so every test also exercises chunk boundaries)."""
    path = tmp_path_factory.mktemp("pq") / "race"
    write_dataset(sequence, path, chunk_frames=60)
    return ParquetSequence.open(path)


class TestRoundTrip:
    def test_the_whole_sequence_comes_back_exactly(self, dataset, sequence):
        assert_same(dataset.read_sequence(), sequence)

    def test_every_frame_is_identical_through_the_file(self, dataset, sequence):
        back = dataset.read_sequence()
        for i in range(0, sequence.n_frames, 11):
            a, b = sequence.frame(i), back.frame(i)
            for k, v in a.to_dict().items():
                assert np.array_equal(v, b.to_dict()[k], equal_nan=True), (i, k)

    @pytest.mark.parametrize(
        "start, stop, every", [(0, 10, 1), (100, 160, 1), (55, 70, 1), (0, 341, 7), (13, 300, 5), (59, 61, 1)]
    )
    def test_partial_and_decimated_reads_equal_the_same_slice_of_the_original(
        self, dataset, sequence, start, stop, every
    ):
        part = dataset.read_sequence(start, stop, every)
        ref = sequence.slice(start, min(stop, sequence.n_frames), every)
        assert part.n_frames == ref.n_frames
        for name in ARRAYS[:12]:
            assert np.array_equal(getattr(part, name), getattr(ref, name), equal_nan=True), name
        assert part.step_s == ref.step_s

    def test_chunk_boundaries_are_invisible(self, sequence, tmp_path):
        for chunk in (1, 7, 341, 10_000):
            ds = ParquetSequence.open(write_dataset(sequence.slice(0, 50), tmp_path / f"c{chunk}", chunk_frames=chunk))
            assert_same(ds.read_sequence(), sequence.slice(0, 50))

    def test_convenience_wrappers_on_the_sequence(self, sequence, tmp_path):
        part = sequence.slice(20, 80)
        part.to_parquet(tmp_path / "w")
        assert_same(RaceSequence.from_parquet(tmp_path / "w"), part)
        assert RaceSequence.from_parquet(tmp_path / "w", 10, 30, 2).n_frames == 10

    def test_iter_chunks_covers_the_dataset_once_in_order(self, dataset, sequence):
        chunks = list(dataset.iter_chunks(100))
        assert [c.n_frames for c in chunks] == [100, 100, 100, 41]
        assert_same(RaceSequence.concat(chunks), sequence)

    def test_iter_chunks_with_decimation(self, dataset):
        chunks = list(dataset.iter_chunks(30, every=5))
        assert sum(c.n_frames for c in chunks) == len(range(0, dataset.n_frames, 5))

    def test_retired_and_pit_cars_survive(self, dataset, sequence):
        back = dataset.read_sequence()
        j = sequence.driver_names.index("finished_or_retired")
        sai = sequence.driver_codes.index("SAI")
        assert np.array_equal(back.driver_x[:, sai, j], sequence.driver_x[:, sai, j]) and back.driver_x[-1, sai, j] == 1


class TestFilesAndSchema:
    def test_layout_on_disk(self, dataset):
        names = sorted(p.name for p in dataset.path.iterdir())
        assert names == ["drivers", "edges", "global", MANIFEST, "track.parquet"]
        assert len(list((dataset.path / "drivers").glob("part-*.parquet"))) == 6  # 341 frames in chunks of 60

    def test_manifest_describes_the_dataset(self, dataset, sequence):
        m = dataset.manifest
        assert m["format"] == PARQUET_FORMAT and m["n_frames"] == sequence.n_frames and m["step_s"] == 1.0
        assert m["driver_codes"] == sequence.driver_codes and m["flag_bits"] == FLAG_BITS
        assert [c["frame_start"] for c in m["chunks"]] == [0, 60, 120, 180, 240, 300]
        assert m["chunks"][-1]["frame_stop"] == sequence.n_frames
        assert m["t_start"] == sequence.times[0] and m["t_stop"] == sequence.times[-1]
        assert json.loads((dataset.path / MANIFEST).read_text())["n_track"] == sequence.n_track

    def test_driver_table_is_long_format_with_typed_columns(self, dataset, sequence):
        df = dataset.scan_drivers().collect()
        assert df.height == int(sequence.driver_present.sum())
        schema = df.schema
        assert schema["frame"] == pl.Int32 and schema["t"] == pl.Float64 and schema["driver"] == pl.String
        assert schema["slot"] == pl.UInt8 and schema["x_m"] == pl.Float32
        assert set(sequence.driver_names) <= set(df.columns) and {"frame", "t", "driver", "slot", "track_node"} <= set(
            df.columns
        )

    def test_masked_values_are_nulls_not_zeros(self, dataset, sequence):
        df = dataset.scan_drivers().collect()
        j = sequence.driver_names.index("gap_ahead_s")
        # the leader has no car ahead: the mask is False and the file holds a null (not a 0.0 that looks like a gap)
        expected_nulls = int((sequence.driver_present & ~sequence.driver_mask[:, :, j]).sum())
        assert expected_nulls > 0 and df["gap_ahead_s"].null_count() == expected_nulls
        assert df.filter(pl.col("driver") == "VER")["gap_ahead_s"].null_count() == sequence.n_frames  # VER always leads

    def test_the_track_table_has_one_row_per_node_and_the_static_features(self, dataset, sequence):
        tr = dataset.read_track()
        assert tr.height == sequence.n_track and tr["node"].to_list() == list(range(sequence.n_track))
        assert tr.columns == ["node", *sequence.track_static_names]

    def test_global_table_has_flag_sector_lists(self, dataset, sequence):
        g = dataset.scan_global().collect().sort("frame")
        assert g.height == sequence.n_frames and g["frame"].to_list() == list(range(sequence.n_frames))
        i121 = int(np.argmin(abs(sequence.times - 121.0)))
        assert g["double_yellow_sectors"][i121].to_list() == [3] and g["yellow_sectors"][i121].to_list() == [8]
        i60 = int(np.argmin(abs(sequence.times - 60.0)))
        assert g["yellow_sectors"][i60].to_list() == [] and g["is_sc"][i60] == 0

    def test_edge_table_is_sparse(self, dataset, sequence):
        e = dataset.scan_edges().collect()
        assert e.height == int(sequence.edge_exists.sum()) and e.height < sequence.n_frames * 6 * 5 + 1
        assert {"frame", "t", "src", "dst", "src_slot", "dst_slot", *sequence.edge_names} <= set(e.columns)
        assert e["src"].ne(e["dst"]).all()

    def test_files_are_zstd_compressed_and_much_smaller_than_the_dense_arrays(self, dataset, sequence):
        assert dataset.disk_bytes() < sequence.nbytes / 2

    def test_repr(self, dataset):
        r = repr(dataset)
        assert "341 frames" in r and "6 drivers" in r and "6 chunks" in r and "MB on disk" in r


class TestPolars:
    """The point of the format: analysis without densifying anything."""

    def test_lazy_queries(self, dataset):
        top = (
            dataset.scan_drivers()
            .group_by("driver")
            .agg(pl.col("speed_kmh").max().alias("vmax"))
            .sort("vmax")
            .collect()
        )
        assert top.height == 6 and top["vmax"].min() > 250

    def test_predicate_pushdown_on_the_time_range(self, dataset):
        df = dataset.scan_drivers().filter((pl.col("t") >= 180) & (pl.col("t") <= 190)).collect()
        assert df["t"].min() >= 180 and df["t"].max() <= 190 and df.height == 11 * 6

    def test_join_drivers_with_global_flags(self, dataset):
        df = dataset.scan_drivers().join(dataset.scan_global().select("frame", "is_sc"), on="frame").collect()
        sc = df.filter(pl.col("is_sc") == 1)
        assert sc["t"].min() >= 180 and sc["t"].max() <= 245

    def test_rolling_window_features_per_driver(self, dataset):
        df = (dataset.scan_drivers().sort("slot", "frame")
              .with_columns(avg=pl.col("speed_kmh").rolling_mean(10).over("driver")).collect())  # fmt: skip
        assert df["avg"].null_count() == 6 * 9 and df["avg"].max() < 400

    def test_frame_range_by_time(self, dataset, sequence):
        lo, hi = dataset.frame_range(100.0, 130.0)
        assert sequence.times[lo] == 100.0 and sequence.times[hi - 1] == 130.0
        assert dataset.frame_range() == (0, dataset.n_frames)
        assert dataset.frame_range(1e9, None)[0] == dataset.n_frames

    def test_time_range_reads_the_right_frames(self, dataset, sequence):
        lo, hi = dataset.frame_range(150.0, 160.0)
        part = dataset.read_sequence(lo, hi)
        assert part.times[0] == 150.0 and part.times[-1] == 160.0


class TestContinuityInPolars:
    def test_agrees_with_the_dense_implementation(self, dataset, sequence):
        a, b = dataset.continuity_report(), sequence.continuity_report()
        assert a.ok and b.ok and a.max_position_step_ratio == pytest.approx(b.max_position_step_ratio, rel=1e-4)
        assert a.lap_wraps == b.lap_wraps and a.n_frames == b.n_frames

    def test_detects_the_same_injected_teleport_and_regression_as_the_dense_check(self, sequence, tmp_path):
        bad = sequence.slice(0, 120)
        bad.driver_x = bad.driver_x.copy()
        bad.driver_x[30:, bad.driver_codes.index("HAM"), bad.driver_names.index("x_m")] += 400.0
        bad.driver_x[70:, bad.driver_codes.index("LEC"), bad.driver_names.index("progress_m")] -= 200.0
        ds = ParquetSequence.open(write_dataset(bad, tmp_path / "bad", chunk_frames=45))
        a, b = ds.continuity_report(), bad.continuity_report()
        assert not a.ok and not b.ok
        assert [(f, c) for f, c, *_ in a.position_jumps] == [(f, c) for f, c, *_ in b.position_jumps] == [(30, "HAM")]
        assert (
            [(f, c) for f, c, _ in a.progress_regressions]
            == [(f, c) for f, c, _ in b.progress_regressions]
            == [(70, "LEC")]
        )
        assert a.position_jumps[0][2] == pytest.approx(b.position_jumps[0][2], rel=1e-4)

    def test_masked_values_are_not_mistaken_for_jumps(self, sequence, tmp_path):
        part = sequence.slice(0, 80)
        part.driver_x, part.driver_mask = part.driver_x.copy(), part.driver_mask.copy()
        slot, xi = part.driver_codes.index("ALO"), part.driver_names.index("x_m")
        part.driver_x[30, slot, xi] += 900.0
        part.driver_mask[30, slot, xi] = False  # stored as a null -> excluded from the check
        assert ParquetSequence.open(write_dataset(part, tmp_path / "m")).continuity_report().ok

    def test_a_pair_of_frames_is_only_judged_when_both_speeds_are_known(self, sequence, tmp_path):
        """Missing data is 'not evaluated', not 'judged against the one speed we happen to have' (dense == Polars)."""
        part = sequence.slice(0, 80)
        part.driver_x, part.driver_mask = part.driver_x.copy(), part.driver_mask.copy()
        slot = part.driver_codes.index("ALO")
        part.driver_x[30:, slot, part.driver_names.index("x_m")] += 400.0  # a 400 m step between frames 29 and 30
        part.driver_mask[29, slot, part.driver_names.index("speed_kmh")] = (
            False  # ...but the speed at frame 29 is unknown
        )
        assert part.continuity_report().ok
        assert ParquetSequence.open(write_dataset(part, tmp_path / "v")).continuity_report().ok
        part.driver_mask[29, slot, part.driver_names.index("speed_kmh")] = (
            True  # with the speed known it *is* a teleport
        )
        assert not part.continuity_report().ok
        assert not ParquetSequence.open(write_dataset(part, tmp_path / "w")).continuity_report().ok

    def test_a_driver_missing_from_some_frames_is_not_compared_across_the_gap(self, sequence, tmp_path):
        part = sequence.slice(0, 60)
        part.driver_present = part.driver_present.copy()
        part.driver_present[30:35, part.driver_codes.index("NOR")] = False  # absent for 5 frames, then back
        ds = ParquetSequence.open(write_dataset(part, tmp_path / "gap"))
        assert ds.continuity_report().ok  # the 5 frame gap is a real move of ~5 frames, must not be read as a jump
        assert ds.read_sequence().driver_present[30:35, part.driver_codes.index("NOR")].sum() == 0


class TestWriter:
    def test_streaming_append_equals_one_shot_write(self, analyser, sequence, tmp_path):
        times = np.arange(100.0, 130.0, 0.5)
        with ParquetSequenceWriter(tmp_path / "s") as w:
            for chunk in analyser.sequence_builder.build_chunks(times, chunk_frames=11):
                w.append(chunk)
        streamed = ParquetSequence.open(tmp_path / "s")
        assert len(streamed.manifest["chunks"]) == 6 and streamed.step_s == 0.5
        assert_same(streamed.read_sequence(), analyser.sequence_builder.build(times))

    def test_refuses_to_overwrite_unless_asked(self, sequence, tmp_path):
        write_dataset(sequence.slice(0, 5), tmp_path / "d")
        with pytest.raises(FileExistsError):
            write_dataset(sequence.slice(0, 5), tmp_path / "d")
        write_dataset(sequence.slice(0, 7), tmp_path / "d", overwrite=True)
        assert ParquetSequence.open(tmp_path / "d").n_frames == 7

    def test_overwrite_removes_stale_parts(self, sequence, tmp_path):
        write_dataset(sequence.slice(0, 50), tmp_path / "d", chunk_frames=10)
        write_dataset(sequence.slice(0, 5), tmp_path / "d", overwrite=True, chunk_frames=10)
        assert len(list((tmp_path / "d" / "drivers").iterdir())) == 1

    def test_an_unfinished_dataset_refuses_to_open(self, sequence, tmp_path):
        w = ParquetSequenceWriter(tmp_path / "half")
        w.append(sequence.slice(0, 10))  # no close(): the manifest is missing
        with pytest.raises(FileNotFoundError, match="not finished"):
            ParquetSequence.open(tmp_path / "half")

    def test_a_failed_write_never_produces_a_manifest(self, sequence, tmp_path):
        with pytest.raises(RuntimeError), ParquetSequenceWriter(tmp_path / "boom") as w:
            w.append(sequence.slice(0, 10))
            raise RuntimeError("crash while writing")
        assert not (tmp_path / "boom" / MANIFEST).exists()

    def test_chunks_must_be_consecutive_in_time_and_from_one_session(self, sequence, tmp_path):
        with ParquetSequenceWriter(tmp_path / "t") as w:
            w.append(sequence.slice(10, 20))
            with pytest.raises(ValueError, match="consecutive in time"):
                w.append(sequence.slice(0, 5))
            with pytest.raises(ValueError, match="consecutive in time"):
                w.append(sequence.slice(15, 25))
            other = sequence.slice(30, 40)
            other.track_static = other.track_static + 1.0
            with pytest.raises(ValueError, match="different track"):
                w.append(other)
            w.append(sequence.slice(30, 40))

    def test_different_driver_slots_are_rejected(self, sequence, tmp_path):
        with ParquetSequenceWriter(tmp_path / "s") as w:
            w.append(sequence.slice(0, 5))
            other = sequence.slice(10, 15)
            other.driver_codes = other.driver_codes[::-1]
            with pytest.raises(ValueError, match="schema or driver slots"):
                w.append(other)

    def test_empty_writer_and_closed_writer(self, sequence, tmp_path):
        with pytest.raises(ValueError, match="nothing was written"):
            ParquetSequenceWriter(tmp_path / "e").close()
        w = ParquetSequenceWriter(tmp_path / "c")
        w.append(sequence.slice(0, 3))
        w.close()
        assert w.close() == w.path  # idempotent
        with pytest.raises(ValueError, match="closed"):
            w.append(sequence.slice(5, 8))

    def test_empty_chunks_are_ignored(self, sequence, tmp_path):
        with ParquetSequenceWriter(tmp_path / "z") as w:
            w.append(sequence.slice(0, 0))
            w.append(sequence.slice(0, 4))
        assert ParquetSequence.open(tmp_path / "z").n_frames == 4

    def test_irregular_chunk_steps_clear_the_nominal_step(self, sequence, tmp_path):
        a, b = sequence.slice(0, 10), sequence.slice(20, 30, 2)  # step 1 s then 2 s
        with ParquetSequenceWriter(tmp_path / "i") as w:
            w.append(a)
            w.append(b)
        assert ParquetSequence.open(tmp_path / "i").step_s is None


class TestReaderErrors:
    def test_not_a_dataset(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ParquetSequence.open(tmp_path)

    def test_unknown_format_and_version(self, dataset, tmp_path):
        import shutil

        for key, value, msg in (
            ("format", "other", "unsupported dataset format"),
            ("format_version", 99, "unsupported dataset format"),
        ):
            copy = tmp_path / key
            shutil.copytree(dataset.path, copy)
            m = json.loads((copy / MANIFEST).read_text())
            m[key] = value
            (copy / MANIFEST).write_text(json.dumps(m))
            with pytest.raises(ValueError, match=msg):
                ParquetSequence.open(copy)

    def test_a_different_flag_layout_is_refused(self, dataset, tmp_path):
        import shutil

        shutil.copytree(dataset.path, tmp_path / "f")
        m = json.loads((tmp_path / "f" / MANIFEST).read_text())
        m["flag_bits"] = m["flag_bits"][::-1]
        (tmp_path / "f" / MANIFEST).write_text(json.dumps(m))
        with pytest.raises(ValueError, match="flag bit layout"):
            ParquetSequence.open(tmp_path / "f")

    @pytest.mark.parametrize("args", [(5, 5, 1), (10, 5, 1), (-1, None, 1), (400, None, 1), (0, 10, 0)])
    def test_invalid_frame_ranges(self, dataset, args):
        with pytest.raises(ValueError, match="invalid frame range"):
            dataset.read_sequence(*args)

    def test_iter_chunks_validates(self, dataset):
        with pytest.raises(ValueError):
            list(dataset.iter_chunks(0))

    def test_stop_beyond_the_end_is_clamped(self, dataset):
        assert dataset.read_sequence(300, 10_000).n_frames == dataset.n_frames - 300
