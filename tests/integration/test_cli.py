"""Command line interface: argument parsing, the `run` pipeline on a stand-in session, process-level entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from f1strategy import cli
from f1strategy.graph.exporter import TRACK_FEATURES
from f1strategy.sequence.sampling import DEFAULT_CHUNK_FRAMES

ROOT = Path(__file__).resolve().parents[2]


def _run_proc(*args, cwd=ROOT):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=180, cwd=cwd)


class TestParser:
    def test_defaults_reproduce_the_original_example(self):
        a = cli.build_parser().parse_args([])
        assert (a.year, a.event, a.session, a.lap) == (2025, "Belgian Grand Prix", "R", 44)
        assert a.sectors == list(cli.SECTORS) and a.out_dir == "graphs" and a.spacing == 10.0 and a.max_tokens == 512
        assert not (a.no_tables or a.no_plots or a.no_export or a.verbose) and a.cache_dir is None
        assert not (a.continuous or a.no_gif or a.no_snapshots)  # the point-by-point behaviour stays the default
        assert (a.from_lap, a.to_lap, a.step, a.hz, a.gif_every, a.fps) == (None, None, None, None, 0, 10.0)
        assert (a.dataset_format, a.chunk_frames) == ("parquet", DEFAULT_CHUNK_FRAMES)

    def test_overrides(self):
        a = cli.build_parser().parse_args(
            [
                "--year",
                "2024",
                "--event",
                "Monza",
                "--session",
                "Q",
                "--lap",
                "3",
                "--sectors",
                "Sector 2",
                "Sector 3",
                "--out-dir",
                "o",
                "--spacing",
                "5",
                "--max-tokens",
                "256",
                "--drs-detection-offset",
                "150",
                "--cache-dir",
                "c",
                "--no-tables",
                "--no-plots",
                "--no-export",
                "-v",
            ]  # fmt: skip
        )
        assert (a.year, a.event, a.session, a.lap, a.sectors) == (2024, "Monza", "Q", 3, ["Sector 2", "Sector 3"])
        assert (a.spacing, a.max_tokens, a.drs_detection_offset, a.cache_dir) == (5.0, 256, 150.0, "c")
        assert a.no_tables and a.no_plots and a.no_export and a.verbose

    @pytest.mark.parametrize(
        "raw, parsed", [("5", 5), ("12", 12), ("Monza", "Monza"), ("Sao Paulo", "Sao Paulo"), ("2024-5", "2024-5")]
    )
    def test_event_digits_become_a_round_number_names_stay_strings(self, raw, parsed):
        got = cli.build_parser().parse_args(["--event", raw]).event
        assert got == parsed and type(got) is type(parsed)

    @pytest.mark.parametrize("argv", [["--sectors", "Sector 4"], ["--lap", "x"], ["--year", "abc"], ["--bogus"]])
    def test_invalid_arguments_exit_with_code_2(self, argv):
        with pytest.raises(SystemExit) as exc:
            cli.build_parser().parse_args(argv)
        assert exc.value.code == 2


class TestContinuousMode:
    def test_run_continuous_streams_a_parquet_dataset_and_a_gif_by_default(self, analyser, tmp_path, capsys):
        res = cli.run_continuous(analyser, "Synthetic GP", 2, 3, step_s=1.0, out_dir=str(tmp_path))
        assert Path(res.parquet_dir).name == "sequence_lap2-3" and Path(res.gif_file).name == "sequence_lap2-3.gif"
        assert res.sequence_file is None and res.continuity_ok and res.n_frames > 100
        out = capsys.readouterr().out
        assert "Continuous mode: laps 2-3" in out and "continuity OK" in out and "100%" in out and "x real time" in out
        assert "real telemetry: position 4.5 Hz, car data 4.2 Hz" in out and "Saved Parquet dataset" in out
        from f1strategy import ParquetSequence

        ds = ParquetSequence.open(res.parquet_dir)
        assert ds.n_frames == res.n_frames and ds.step_s == 1.0

    def test_npz_format_writes_the_single_file_only(self, analyser, tmp_path):
        res = cli.run_continuous(analyser, "x", 3, out_dir=str(tmp_path), gif=False, dataset_format="npz")
        assert res.parquet_dir is None and Path(res.sequence_file).name == "sequence_lap3.npz"
        assert [p.name for p in tmp_path.iterdir()] == ["sequence_lap3.npz"]
        from f1strategy import RaceSequence

        assert RaceSequence.load(res.sequence_file).n_frames == res.n_frames

    def test_both_formats_hold_identical_data(self, analyser, tmp_path):
        res = cli.run_continuous(
            analyser, "x", 3, out_dir=str(tmp_path), gif=False, dataset_format="both", chunk_frames=20
        )
        from f1strategy import RaceSequence

        a, b = RaceSequence.load(res.sequence_file), RaceSequence.from_parquet(res.parquet_dir)
        assert np.array_equal(a.driver_x, b.driver_x, equal_nan=True) and np.array_equal(a.track_flags, b.track_flags)

    def test_hz_sets_the_data_rate_independently_of_the_gif_rate(self, analyser, tmp_path, capsys):
        res = cli.run_continuous(analyser, "x", 3, hz=10, out_dir=str(tmp_path), fps=5, chunk_frames=200, gif_every=10)
        from PIL import Image

        from f1strategy import ParquetSequence

        ds = ParquetSequence.open(res.parquet_dir)
        assert ds.step_s == pytest.approx(0.1) and 560 < ds.n_frames < 600 and len(ds.manifest["chunks"]) == 3
        with Image.open(res.gif_file) as im:
            assert im.n_frames == -(-ds.n_frames // 10) and im.info["duration"] == 200
        out = capsys.readouterr().out
        assert "(10 Hz)" in out and "one frame every 0.1 s" in out and "interpolates between real samples" in out

    def test_gif_speed_is_reported_for_the_decimated_dataset(self, analyser, tmp_path, capsys):
        cli.run_continuous(analyser, "x", 3, hz=10, out_dir=str(tmp_path), gif_every=10, fps=5)
        assert "5x real time" in capsys.readouterr().out  # 10 frames x 0.1 s x 5 fps

    def test_a_single_lap_is_named_after_that_lap(self, analyser, tmp_path):
        res = cli.run_continuous(analyser, "x", 3, out_dir=str(tmp_path), gif=False)
        assert Path(res.parquet_dir).name == "sequence_lap3" and res.gif_file is None
        assert [p.name for p in tmp_path.iterdir()] == ["sequence_lap3"]

    def test_rerunning_replaces_the_previous_dataset(self, analyser, tmp_path):
        cli.run_continuous(analyser, "x", 3, out_dir=str(tmp_path), gif=False, chunk_frames=20)
        res = cli.run_continuous(analyser, "x", 3, out_dir=str(tmp_path), gif=False, chunk_frames=1000)
        from f1strategy import ParquetSequence

        assert len(ParquetSequence.open(res.parquet_dir).manifest["chunks"]) == 1

    def test_gif_frame_count_is_automatically_capped(self, analyser, tmp_path):
        from PIL import Image

        res = cli.run_continuous(
            analyser, "x", 2, 5, step_s=0.5, out_dir=str(tmp_path), dataset_format="npz"
        )  # ~460 frames
        with Image.open(res.gif_file) as im:
            assert 150 <= im.n_frames <= 300 < res.n_frames

    def test_the_gif_is_drawn_from_a_decimated_read_not_the_whole_dataset(self, analyser, tmp_path, monkeypatch):
        from f1strategy.sequence import ParquetSequence

        reads = []
        original = ParquetSequence.read_sequence

        def spy(self, start_frame=0, stop_frame=None, every=1):
            reads.append(every)
            return original(self, start_frame, stop_frame, every)

        monkeypatch.setattr(ParquetSequence, "read_sequence", spy)
        cli.run_continuous(analyser, "x", 2, 4, step_s=0.5, out_dir=str(tmp_path))
        assert reads and reads[-1] > 1  # automatic decimation: never densifies the full dataset for the animation

    def test_main_runs_both_modes_with_continuous(self, monkeypatch, analyser, tmp_path):
        monkeypatch.setattr(cli, "SessionAnalyser", lambda **kw: analyser)
        cli.main(["--lap", "3", "--out-dir", str(tmp_path), "--continuous", "--no-tables", "--no-gif"])
        names = sorted(p.name for p in tmp_path.iterdir())
        assert "sequence_lap3" in names and "graph_sector2.png" in names and "graph_lap3_sector2.npz" in names

    def test_no_snapshots_runs_only_the_continuous_part(self, monkeypatch, analyser, tmp_path):
        monkeypatch.setattr(cli, "SessionAnalyser", lambda **kw: analyser)
        cli.main(["--lap", "3", "--out-dir", str(tmp_path), "--continuous", "--no-snapshots", "--no-gif"])
        assert [p.name for p in tmp_path.iterdir()] == ["sequence_lap3"]

    def test_lap_range_and_rate_options_are_forwarded(self, monkeypatch, analyser, tmp_path):
        monkeypatch.setattr(cli, "SessionAnalyser", lambda **kw: analyser)
        cli.main(["--lap", "9", "--from-lap", "2", "--to-lap", "3", "--hz", "0.5", "--format", "npz",
                  "--out-dir", str(tmp_path), "--continuous", "--no-snapshots", "--no-gif"])  # fmt: skip
        from f1strategy import RaceSequence

        seq = RaceSequence.load(tmp_path / "sequence_lap2-3.npz")
        assert seq.step_s == 2.0 and seq.times[0] == pytest.approx(analyser.graph_builder.lap_window(2)[0])

    @pytest.mark.parametrize("argv", [["--no-snapshots"], ["--continuous", "--step", "1", "--hz", "10"],
                                      ["--continuous", "--hz", "0"], ["--continuous", "--hz", "-5"]])  # fmt: skip
    def test_invalid_combinations_fail_before_any_download(self, monkeypatch, argv):
        def boom(**kw):
            raise AssertionError("the session must not be loaded")

        monkeypatch.setattr(cli, "SessionAnalyser", boom)
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2

    def test_continuous_is_off_by_default(self, monkeypatch, analyser, tmp_path):
        monkeypatch.setattr(cli, "SessionAnalyser", lambda **kw: analyser)
        cli.main(["--lap", "3", "--out-dir", str(tmp_path), "--no-tables", "--no-plots", "--no-export"])
        assert list(tmp_path.iterdir()) == []


class TestRun:
    def test_writes_one_npz_and_png_per_sector(self, analyser, tmp_path, capsys):
        res = cli.run(analyser, "Synthetic GP", 3, out_dir=str(tmp_path / "out"))
        names = sorted(Path(p).name for p in res.npz_files + res.png_files)
        assert names == sorted(
            [f"graph_lap3_sector{i}.npz" for i in (1, 2, 3)] + [f"graph_sector{i}.png" for i in (1, 2, 3)]
        )
        out = capsys.readouterr().out
        assert (
            "Track: TrackModel(" in out
            and "Summarized Data at lap 3" in out
            and out.count("Graph successfully created") == 3
        )
        assert all(Path(p).stat().st_size > 0 for p in res.npz_files + res.png_files)

    def test_exported_npz_is_internally_consistent_even_when_pooled(self, analyser, tmp_path):
        """End-to-end regression for the bug seen on real data (pooled features but unpooled mask)."""
        res = cli.run(
            analyser, "x", 3, out_dir=str(tmp_path), sectors=["Sector 2"], max_tokens=128, tables=False, plots=False
        )
        d = np.load(res.npz_files[0])
        assert d["track_x"].shape == d["track_mask"].shape == (100, len(TRACK_FEATURES))
        assert int(d["pool_factor"]) == 5 and d["driver_to_track"].max() < 100
        assert d["driver_edge_index"].shape[1] == d["driver_edge_attr"].shape[0]

    def test_flags_switch_outputs_off(self, analyser, tmp_path, capsys):
        res = cli.run(
            analyser, "x", 3, out_dir=str(tmp_path), sectors=["Sector 1"], tables=False, plots=False, export=False
        )
        assert res.npz_files == [] and res.png_files == [] and "Summarized Data" not in capsys.readouterr().out
        assert list(tmp_path.iterdir()) == []

    def test_sector_selection(self, analyser, tmp_path):
        res = cli.run(analyser, "x", 2, out_dir=str(tmp_path), sectors=["Sector 3"], tables=False)
        assert [Path(p).name for p in res.npz_files] == ["graph_lap2_sector3.npz"]

    def test_creates_the_output_directory(self, analyser, tmp_path):
        target = tmp_path / "a" / "b"
        cli.run(analyser, "x", 2, out_dir=str(target), sectors=["Sector 1"], tables=False, plots=False)
        assert target.is_dir()

    def test_survives_a_session_without_results_table(self, analyser, tmp_path, capsys):
        cli.run(analyser, "x", 3, out_dir=str(tmp_path), sectors=["Sector 1"], plots=False, export=False)
        assert "results unavailable" in capsys.readouterr().out  # the synthetic session has no `results`


class TestMain:
    @pytest.fixture()
    def stub(self, monkeypatch, analyser):
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return analyser

        monkeypatch.setattr(cli, "SessionAnalyser", fake)
        return seen

    def test_arguments_are_forwarded_to_the_analyser(self, stub, tmp_path):
        code = cli.main(["--year", "2024", "--event", "Monza", "--session", "Q", "--lap", "3", "--spacing", "20",
                         "--drs-detection-offset", "250", "--cache-dir", "cc", "--out-dir", str(tmp_path), "--no-tables",
                         "--no-plots"])  # fmt: skip
        assert code == 0
        assert stub == dict(year=2024, location="Monza", session_type="Q", track_spacing_m=20.0,
                            drs_detection_offset_m=250.0, cache_dir="cc")  # fmt: skip
        assert len(list(tmp_path.glob("*.npz"))) == 3

    @pytest.mark.parametrize("flag, level", [("-v", "DEBUG"), (None, "INFO")])
    def test_verbosity_sets_the_package_log_level(self, stub, tmp_path, flag, level):
        import logging

        pkg = logging.getLogger("f1strategy")
        try:
            argv = ["--lap", "2", "--out-dir", str(tmp_path), "--no-tables", "--no-plots", "--no-export"]
            cli.main(argv + ([flag] if flag else []))
            assert pkg.level == getattr(logging, level)
        finally:
            pkg.setLevel(logging.NOTSET)  # do not leak into other tests


class TestProcessLevelEntryPoints:
    @pytest.mark.parametrize(
        "args", [["-m", "f1strategy", "--help"], ["main.py", "--help"], ["-m", "f1strategy.cli", "--help"]]
    )
    def test_help_works_from_every_entry_point(self, args):
        out = _run_proc(*args)
        assert out.returncode == 0, out.stderr
        assert "--lap" in out.stdout and "--sectors" in out.stdout

    def test_main_py_falls_back_to_the_src_directory_when_the_package_is_not_installed(self):
        """Simulate a fresh clone: `import f1strategy` fails until main.py has put ./src on sys.path."""
        script = """
import importlib.abc, runpy, sys

class BlockUntilSrc(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] == 'f1strategy' and not (sys.path and __import__('os').path.basename(sys.path[0]) == 'src'):
            raise ImportError('f1strategy is not installed (simulated)')

sys.meta_path.insert(0, BlockUntilSrc())
sys.argv = ['main.py', '--help']
try:
    runpy.run_path('main.py', run_name='__main__')
except SystemExit as exc:
    print('exit', exc.code)
print('loaded-from', sys.modules['f1strategy'].__file__)
"""
        out = _run_proc("-c", script)
        assert out.returncode == 0, out.stderr
        assert "exit 0" in out.stdout and "src" in out.stdout.split("loaded-from")[1]

    def test_invalid_argument_is_reported_by_the_process(self):
        out = _run_proc("-m", "f1strategy", "--sectors", "Sector 9")
        assert out.returncode == 2 and "invalid choice" in out.stderr
