"""
builder.py · Formula 1 Strategic Decisions
=========================================================
Builds :class:`~f1strategy.sequence.dataset.RaceSequence` objects by evaluating the race on a time grid.

Two levels of cost, one set of numbers:

* **dense path (default).** ``RaceGraphBuilder.snapshot`` + ``GraphExporter.export_snapshot`` produce the per-frame
  arrays directly, without constructing a networkx graph (which dominates the cost of a snapshot). The row builders are
  *the same functions* the graph exporter uses, and the tests require frames to be bit-for-bit identical to
  ``GraphExporter.export(graph)``.
* **graph path** (:meth:`iter_graphs`) yields the full networkx graphs lazily, for plotting or inspection.

For races sampled at 10 Hz (~54 000 frames) use :meth:`build_chunks`: it yields bounded-size sequences that share the
same driver slots, to be written to disk as they are produced (see ``f1strategy.sequence.parquet``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator

import networkx as nx
import numpy as np

from f1strategy.graph.builder import RaceGraphBuilder
from f1strategy.graph.exporter import FrameRecord, GraphExporter, StaticTrack, pack_flags
from f1strategy.sequence.dataset import RaceSequence
from f1strategy.sequence.sampling import DEFAULT_CHUNK_FRAMES

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]  # (frames done, frames total)


def _as_times(times: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(times) if not isinstance(times, np.ndarray) else times, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("times must be a non-empty 1-D sequence of session seconds")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("times must be strictly increasing")
    return arr


def _nominal_step(grid: np.ndarray) -> float | None:
    steps = np.diff(grid)
    if len(steps) and bool(np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9)):
        return float(steps[0])
    return None


class SequenceBuilder:
    """Evaluates the race at many times and packs the result into ``RaceSequence`` objects."""

    def __init__(self, graph_builder: RaceGraphBuilder) -> None:
        self.graph_builder = graph_builder
        self._exporter = GraphExporter(normalise=False)  # always raw; normalisation happens when frames are read
        self._static: StaticTrack | None = None

    # ------------------------------------------------------------------ pieces
    @property
    def static_track(self) -> StaticTrack:
        if self._static is None:
            self._static = self._exporter.static_track(self.graph_builder.track)
        return self._static

    def driver_slots(self) -> list[str]:
        """Fixed driver slots (alphabetical): every car with telemetry and timing data, whether or not it is present
        in every frame. Fixed up front so consecutive chunks agree."""
        gb = self.graph_builder
        return sorted(c for c in gb.telemetry.codes if gb.timing.has(c))

    def record(self, t: float) -> FrameRecord:
        """The light per-frame record at session time ``t`` (no graph)."""
        gb = self.graph_builder
        snap = gb.snapshot(float(t))
        yellow, double = gb.node_flag_arrays(snap.race_control)
        return self._exporter.export_snapshot(snap, pack_flags(yellow, double, snap.race_control), gb.track.length_m)

    def iter_graphs(self, times: Iterable[float]) -> Iterator[nx.DiGraph]:
        """Lazily yield the full networkx graph for each time (use it to plot / inspect frame by frame)."""
        for t in _as_times(times):
            yield self.graph_builder.build(float(t))

    # ------------------------------------------------------------------ building
    def build(self, times: Iterable[float], progress: ProgressCallback | None = None) -> RaceSequence:
        """Build the whole sequence in memory for ``times`` (strictly increasing session seconds).

        For long or fast-sampled stretches use :meth:`build_chunks` instead."""
        grid = _as_times(times)
        records = []
        for k, t in enumerate(grid):
            records.append(self.record(t))
            if progress:
                progress(k + 1, len(grid))
        seq = RaceSequence.from_records(
            records, self.static_track, driver_codes=self.driver_slots(), step_s=_nominal_step(grid)
        )
        log.info("Built %r (%.1f MB in memory)", seq, seq.nbytes / 1e6)
        return seq

    def build_chunks(
        self, times: Iterable[float], chunk_frames: int = DEFAULT_CHUNK_FRAMES, progress: ProgressCallback | None = None
    ) -> Iterator[RaceSequence]:
        """Yield consecutive sequences of at most ``chunk_frames`` frames (bounded memory), all with the same driver
        slots and nominal step. Concatenating them equals :meth:`build` over the same times."""
        if chunk_frames < 1:
            raise ValueError("chunk_frames must be >= 1")
        grid = _as_times(times)
        slots, step, done = self.driver_slots(), _nominal_step(grid), 0
        for start in range(0, len(grid), chunk_frames):
            part = grid[start : start + chunk_frames]
            records = []
            for t in part:
                records.append(self.record(t))
                done += 1
                if progress:
                    progress(done, len(grid))
            yield RaceSequence.from_records(records, self.static_track, driver_codes=slots, step_s=step)
