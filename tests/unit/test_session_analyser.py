"""SessionAnalyser: construction paths, cache handling, lazy FastF1 import, wiring of the components."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from f1strategy import RaceGraphBuilder, SessionAnalyser
from f1strategy.session import DEFAULT_CACHE_DIR
from tests.helpers import clone_session


@pytest.fixture()
def fake_fastf1(monkeypatch, syn):
    """A stand-in `fastf1` module recording how SessionAnalyser talks to it."""
    calls = {"cache": [], "get_session": [], "load": []}
    session = clone_session(syn.session, load=lambda **kw: calls["load"].append(kw))
    mod = types.ModuleType("fastf1")
    mod.Cache = types.SimpleNamespace(enable_cache=lambda path: calls["cache"].append(path))
    mod.get_session = lambda *a: (calls["get_session"].append(a), session)[1]
    monkeypatch.setitem(sys.modules, "fastf1", mod)
    return calls


class TestConstruction:
    def test_needs_either_a_session_or_year_and_event(self):
        for kwargs in ({}, {"year": 2025}, {"location": "Monza"}):
            with pytest.raises(ValueError, match="year.*location.*session"):
                SessionAnalyser(**kwargs)

    def test_from_session_equals_session_keyword(self, syn):
        a = SessionAnalyser.from_session(syn.session)
        b = SessionAnalyser(session=syn.session)
        assert a.track.n == b.track.n and a._driver_map == b._driver_map

    def test_components_are_wired_together(self, analyser):
        assert isinstance(analyser.graph_builder, RaceGraphBuilder)
        assert analyser.graph_builder.track is analyser.track is analyser.telemetry.track
        assert (
            analyser.graph_builder.telemetry is analyser.telemetry and analyser.graph_builder.timing is analyser.timing
        )
        assert analyser._driver_map == {"1": "VER", "4": "NOR", "16": "LEC", "44": "HAM", "14": "ALO", "55": "SAI"}

    def test_track_spacing_option_changes_the_resolution(self, syn):
        fine = SessionAnalyser.from_session(syn.session, track_spacing_m=5.0)
        assert fine.track.n == pytest.approx(2 * 500, abs=2) and abs(fine.track.spacing_m - 5.0) < 0.1
        coarse = SessionAnalyser.from_session(syn.session, track_spacing_m=50.0)
        assert coarse.track.n == pytest.approx(100, abs=1)

    def test_drs_detection_options_are_forwarded(self, syn):
        a = SessionAnalyser.from_session(syn.session, drs_detection_points_m=[100.0, 2000.0])
        assert a.track.is_drs_detection.sum() == 2


class TestLoadingThroughFastF1:
    def test_loads_everything_the_graph_needs(self, fake_fastf1, tmp_path):
        SessionAnalyser(2025, "Belgian Grand Prix", "R", cache_dir=str(tmp_path / "c"))
        assert fake_fastf1["get_session"] == [(2025, "Belgian Grand Prix", "R")]
        assert fake_fastf1["load"] == [{"laps": True, "telemetry": True, "weather": True, "messages": True}]

    def test_cache_dir_precedence_argument_then_env_then_home(self, fake_fastf1, tmp_path, monkeypatch):
        explicit, from_env = tmp_path / "explicit", tmp_path / "env"
        monkeypatch.setenv("F1_CACHE_DIR", str(from_env))
        SessionAnalyser(2025, "x", cache_dir=str(explicit))
        SessionAnalyser(2025, "x")
        monkeypatch.delenv("F1_CACHE_DIR")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        SessionAnalyser(2025, "x")
        assert [Path(p) for p in fake_fastf1["cache"]] == [explicit, from_env, tmp_path / "fastf1_cache"]
        assert explicit.is_dir() and from_env.is_dir() and (tmp_path / "fastf1_cache").is_dir()
        assert DEFAULT_CACHE_DIR == "~/fastf1_cache"

    def test_injected_session_never_touches_fastf1_or_the_cache(self, syn, monkeypatch):
        monkeypatch.setitem(sys.modules, "fastf1", None)  # any `import fastf1` would now raise ImportError
        SessionAnalyser.from_session(syn.session)

    def test_session_type_defaults_to_race(self, fake_fastf1, tmp_path):
        SessionAnalyser(2024, "Monza", cache_dir=str(tmp_path))
        assert fake_fastf1["get_session"][0][2] == "R"


def test_importing_the_package_does_not_import_fastf1():
    code = "import sys, f1strategy; assert 'fastf1' not in sys.modules, 'fastf1 imported eagerly'"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr


def test_public_api_is_importable_and_versioned():
    import f1strategy

    for name in f1strategy.__all__:
        assert hasattr(f1strategy, name), name
    assert f1strategy.__version__.count(".") == 2
