"""Shared fixtures. The synthetic session is expensive-ish (~1 s), so the standard one is built once per run."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")  # never open a GUI window from tests

from f1strategy import GraphExporter, SessionAnalyser  # noqa: E402
from tests.synthetic.session import make_synthetic_session  # noqa: E402


@pytest.fixture(scope="session")
def syn():
    """Synthetic session (``.session``) plus its ground truth (``.truth``)."""
    return make_synthetic_session()


@pytest.fixture(scope="session")
def analyser(syn):
    """Fully built analyser; DRS detection points are estimated 200 m before each zone."""
    return SessionAnalyser.from_session(syn.session, drs_detection_offset_m=200.0)


@pytest.fixture(scope="session")
def G_sc(analyser):
    """Graph at lap 3 / sector 2 = 184.3 s: safety car, DRS disabled."""
    return analyser.build_session_graph(3, "Sector 2")


@pytest.fixture(scope="session")
def arrays(G_sc):
    return GraphExporter().export(G_sc)


@pytest.fixture(scope="session")
def track(analyser):
    """The track model of the standard synthetic session."""
    return analyser.track


@pytest.fixture(scope="session")
def sequence(analyser):
    """The whole synthetic race (lights out to flag), one frame per second: yellow, SC, VSC, pit stop, retirement."""
    return analyser.build_sequence(40.0, 380.0, 1.0)
