"""Command line interface: argument parsing, the `run` pipeline on a stand-in session, process-level entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from f1strategy import cli
from f1strategy.graph.exporter import TRACK_FEATURES

ROOT = Path(__file__).resolve().parents[2]


def _run_proc(*args, cwd=ROOT):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=180, cwd=cwd)


class TestParser:
    def test_defaults_reproduce_the_original_example(self):
        a = cli.build_parser().parse_args([])
        assert (a.year, a.event, a.session, a.lap) == (2025, "Belgian Grand Prix", "R", 44)
        assert a.sectors == list(cli.SECTORS) and a.out_dir == "graphs" and a.spacing == 10.0 and a.max_tokens == 512
        assert not (a.no_tables or a.no_plots or a.no_export or a.verbose) and a.cache_dir is None

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
