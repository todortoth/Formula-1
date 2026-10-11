"""
f1strategy · Formula 1 Strategic Decisions
=========================================================
Turns FastF1 session data into graph snapshots (a 10 m track ring + one node per driver, with flags, DRS
zones, telemetry, gaps and tyre state) and exports them as fixed-schema arrays for machine learning models.

Public API::

    from f1strategy import SessionAnalyser, GraphExporter
"""

from f1strategy.animation import RaceAnimator, make_gif, team_colors
from f1strategy.domain.driver import DriverNode
from f1strategy.domain.sector import SectorNode
from f1strategy.graph.builder import RaceGraphBuilder
from f1strategy.graph.exporter import GraphArrays, GraphExporter, suggest_pool_factor
from f1strategy.racecontrol.timeline import RaceControlState, RaceControlTimeline
from f1strategy.sequence import (
    ContinuityReport,
    ParquetSequence,
    ParquetSequenceWriter,
    RaceSequence,
    SequenceBuilder,
    time_grid,
    write_dataset,
)
from f1strategy.session import SessionAnalyser
from f1strategy.telemetry.store import TelemetryStore
from f1strategy.telemetry.timing import DriverTiming
from f1strategy.track.model import TrackModel

__version__ = "0.4.0"

__all__ = [
    "ContinuityReport",
    "DriverNode",
    "DriverTiming",
    "GraphArrays",
    "GraphExporter",
    "ParquetSequence",
    "ParquetSequenceWriter",
    "RaceAnimator",
    "RaceControlState",
    "RaceControlTimeline",
    "RaceGraphBuilder",
    "RaceSequence",
    "SectorNode",
    "SequenceBuilder",
    "SessionAnalyser",
    "TelemetryStore",
    "TrackModel",
    "__version__",
    "make_gif",
    "suggest_pool_factor",
    "team_colors",
    "time_grid",
    "write_dataset",
]
