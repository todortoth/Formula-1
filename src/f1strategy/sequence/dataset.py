"""
dataset.py · Formula 1 Strategic Decisions
=========================================================
``RaceSequence``: a continuous stretch of a race as one set of arrays, ready to feed a model.

Why not simply stack the per-snapshot ``GraphArrays``? A track token is ~25 floats x 700 nodes, and almost all of it
never changes. At one frame per second over a race that would be several gigabytes. Instead:

* **static track features** are stored once (``track_static``);
* the six **flags** per track node are bit-packed into one ``uint8`` per node and frame (``track_flags``);
* ``n_drivers`` per node is derived from the driver positions when a frame is reconstructed;
* **drivers sit in fixed slots** (alphabetical by code), so a car keeps its row across time - the live race position
  is just a feature (``position``);
* **proximity edges are dense** ``[T, D, D]`` matrices (``edge_exists`` + ``edge_attr``), which is also the natural
  shape for an attention bias.

Values are stored *raw* (physical units). ``frame(i, normalise=True)`` applies the same scaling the exporter uses, so
the data that drives the animation and the data a model sees are the same numbers.

``frame(i)`` reconstructs the exact ``GraphArrays`` the exporter produces for that snapshot (a test enforces it),
which keeps the point-by-point and the continuous code paths consistent.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from f1strategy.graph.exporter import (
    DRIVER_FEATURES,
    EDGE_FEATURES,
    FLAG_BITS,
    GLOBAL_FEATURES,
    TRACK_DYNAMIC,
    TRACK_FEATURES,
    FrameRecord,
    GraphArrays,
    StaticTrack,
)
from f1strategy.sequence.sampling import DEFAULT_CHUNK_FRAMES

FORMAT_VERSION = 1

_TIME_ARRAYS = (
    "times", "track_flags", "driver_x", "driver_mask", "driver_present", "driver_to_track",
    "global_x", "global_mask", "edge_attr", "edge_attr_mask", "edge_exists",
)  # fmt: skip
_ARRAY_FIELDS = (
    "times", "track_static", "track_static_mask", "track_flags", "driver_x", "driver_mask", "driver_present",
    "driver_to_track", "global_x", "global_mask", "edge_attr", "edge_attr_mask", "edge_exists", "xy_bbox",
)  # fmt: skip


@dataclass
class ContinuityReport:
    """Result of :meth:`RaceSequence.continuity_report`."""

    n_frames: int
    max_position_step_ratio: float  # largest observed step / speed-implied step (1.0 = exactly as fast as the car goes)
    position_jumps: list[tuple[int, str, float, float]] = field(
        default_factory=list
    )  # (frame, code, step m, allowed m)
    progress_regressions: list[tuple[int, str, float]] = field(default_factory=list)  # (frame, code, delta m)
    lap_wraps: int = 0  # start/finish crossings (informational)

    @property
    def ok(self) -> bool:
        return not self.position_jumps and not self.progress_regressions

    def summary(self) -> str:
        state = "OK" if self.ok else "PROBLEMS"
        return (
            f"continuity {state}: {self.n_frames} frames, max step ratio {self.max_position_step_ratio:.2f}, "
            f"{len(self.position_jumps)} position jump(s), {len(self.progress_regressions)} progress regression(s), "
            f"{self.lap_wraps} start/finish crossings"
        )


@dataclass
class RaceSequence:
    """A continuous sequence of race snapshots (see the module docstring for the layout)."""

    times: np.ndarray  # [T] session seconds
    driver_codes: list[str]  # [D] slot order
    track_static: np.ndarray  # [N, Fs]
    track_static_mask: np.ndarray  # [N, Fs] bool
    track_static_names: list[str]
    track_flags: np.ndarray  # [T, N] uint8, bit k = FLAG_BITS[k]
    driver_x: np.ndarray  # [T, D, Fd] raw
    driver_mask: np.ndarray  # [T, D, Fd] bool, True = value present
    driver_present: np.ndarray  # [T, D] bool, driver is part of the snapshot
    driver_to_track: np.ndarray  # [T, D] int32, nearest track node (-1 = unknown)
    driver_names: list[str]
    global_x: np.ndarray  # [T, Fg]
    global_mask: np.ndarray  # [T, Fg]
    global_names: list[str]
    edge_attr: np.ndarray  # [T, D, D, Fe]
    edge_attr_mask: np.ndarray  # [T, D, D, Fe] bool
    edge_exists: np.ndarray  # [T, D, D] bool, edge slot i -> slot j exists
    edge_names: list[str]
    track_length_m: float
    track_spacing_m: float
    xy_bbox: np.ndarray  # [3] centre x, centre y, half extent (for normalisation)
    race_distance_m: float | None = None
    step_s: float | None = None  # nominal sampling step; None if the grid is irregular
    track_names: list[str] = field(default_factory=list)  # full per-frame track schema (static + dynamic)

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.times)

    @property
    def n_frames(self) -> int:
        return len(self.times)

    @property
    def n_track(self) -> int:
        return self.track_static.shape[0]

    @property
    def n_drivers(self) -> int:
        return len(self.driver_codes)

    def validate(self) -> RaceSequence:
        """Raise ``ValueError`` on inconsistent shapes; returns ``self``."""
        T, N, D = len(self.times), self.n_track, self.n_drivers

        def check(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"inconsistent RaceSequence: {msg}")

        check(self.track_flags.shape == (T, N) and self.track_flags.dtype == np.uint8, "track_flags")
        check(self.track_static.shape == self.track_static_mask.shape, "track_static_mask")
        check(self.driver_x.shape == (T, D, len(self.driver_names)) == self.driver_mask.shape, "driver_x / mask")
        check(self.driver_present.shape == (T, D) == self.driver_to_track.shape, "driver_present / driver_to_track")
        check(self.global_x.shape == (T, len(self.global_names)) == self.global_mask.shape, "global")
        fe = len(self.edge_names)
        check(self.edge_attr.shape == (T, D, D, fe) == self.edge_attr_mask.shape, "edge_attr")
        check(self.edge_exists.shape == (T, D, D), "edge_exists")
        check(bool(np.all(np.diff(self.times) > 0)), "times must be strictly increasing")
        check(self.driver_to_track.max(initial=-1) < N, "driver_to_track out of range")
        return self

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_frames(
        cls, frames: Sequence[GraphArrays], driver_codes: Sequence[str] | None = None, step_s: float | None = None
    ) -> RaceSequence:
        """Pack raw (un-normalised) ``GraphArrays`` snapshots of one session into a sequence."""
        if not frames:
            raise ValueError("need at least one frame")
        first = frames[0]
        if any(f.normalised for f in frames):
            raise ValueError("pass raw (un-normalised) frames; normalisation is applied when reading a frame")
        static_cols = [i for i, n in enumerate(first.track_names) if n not in TRACK_DYNAMIC]
        static = StaticTrack(
            names=[first.track_names[i] for i in static_cols],
            values=first.track_x[:, static_cols].copy(),
            mask=first.track_mask[:, static_cols].copy(),
            xy_bbox=tuple(first.xy_bbox),
            length_m=first.track_length_m,
            spacing_m=first.track_spacing_m,
        )
        for t, fr in enumerate(frames):
            if (fr.track_names, fr.driver_names, fr.global_names, fr.edge_names) != (
                first.track_names, first.driver_names, first.global_names, first.edge_names,
            ):  # fmt: skip
                raise ValueError("frames use different feature schemas")
            if not np.array_equal(fr.track_x[:, static_cols], static.values):
                raise ValueError(f"static track features changed at frame {t}: frames must come from one session")
        return cls.from_records(
            [FrameRecord.from_arrays(f) for f in frames], static, driver_codes=driver_codes, step_s=step_s
        )

    @classmethod
    def from_records(
        cls,
        records: Sequence[FrameRecord],
        static: StaticTrack,
        driver_codes: Sequence[str] | None = None,
        step_s: float | None = None,
    ) -> RaceSequence:
        """Pack light per-frame records (the dense sampling path, no graph involved) into a sequence.

        ``driver_codes`` fixes the driver slots; pass the same list for consecutive chunks so a car keeps its row."""
        if not records:
            raise ValueError("need at least one frame")
        slots = sorted({c for r in records for c in r.driver_codes}) if driver_codes is None else list(driver_codes)
        slot_of = {c: i for i, c in enumerate(slots)}
        T, N, D = len(records), len(static.values), len(slots)
        fd, fg, fe = len(DRIVER_FEATURES), len(GLOBAL_FEATURES), len(EDGE_FEATURES)

        times = np.array([r.time_s for r in records], dtype=np.float64)
        track_flags = np.zeros((T, N), dtype=np.uint8)
        driver_x = np.zeros((T, D, fd), dtype=np.float32)
        driver_mask = np.zeros(driver_x.shape, dtype=bool)
        present = np.zeros((T, D), dtype=bool)
        d2t = np.full((T, D), -1, dtype=np.int32)
        global_x = np.zeros((T, fg), dtype=np.float32)
        global_mask = np.zeros(global_x.shape, dtype=bool)
        edge_attr = np.zeros((T, D, D, fe), dtype=np.float32)
        edge_attr_mask = np.zeros(edge_attr.shape, dtype=bool)
        edge_exists = np.zeros((T, D, D), dtype=bool)

        for t, r in enumerate(records):
            track_flags[t] = r.track_flags
            slot_idx = np.array([slot_of[c] for c in r.driver_codes], dtype=int)
            driver_x[t, slot_idx] = r.driver_x
            driver_mask[t, slot_idx] = r.driver_mask
            present[t, slot_idx] = True
            d2t[t, slot_idx] = r.driver_to_track
            global_x[t], global_mask[t] = r.global_x, r.global_mask
            if len(r.edge_src):
                u, v = slot_idx[r.edge_src], slot_idx[r.edge_dst]
                edge_attr[t, u, v] = r.edge_attr
                edge_attr_mask[t, u, v] = r.edge_mask
                edge_exists[t, u, v] = True

        return cls(
            times=times, driver_codes=slots, track_static=static.values.copy(), track_static_mask=static.mask.copy(),
            track_static_names=list(static.names), track_flags=track_flags, driver_x=driver_x,
            driver_mask=driver_mask, driver_present=present, driver_to_track=d2t, driver_names=list(DRIVER_FEATURES),
            global_x=global_x, global_mask=global_mask, global_names=list(GLOBAL_FEATURES), edge_attr=edge_attr,
            edge_attr_mask=edge_attr_mask, edge_exists=edge_exists, edge_names=list(EDGE_FEATURES),
            track_length_m=static.length_m, track_spacing_m=static.spacing_m,
            xy_bbox=np.asarray(static.xy_bbox, dtype=float), race_distance_m=records[0].race_distance_m,
            step_s=step_s, track_names=list(TRACK_FEATURES),
        ).validate()  # fmt: skip

    @classmethod
    def concat(cls, parts: Sequence[RaceSequence]) -> RaceSequence:
        """Join consecutive sequences of the same session (same driver slots and static track) along time."""
        if not parts:
            raise ValueError("nothing to concatenate")
        first = parts[0]
        for p in parts[1:]:
            if p.driver_codes != first.driver_codes or p.track_static_names != first.track_static_names:
                raise ValueError("cannot concatenate sequences with different drivers or track schemas")
            if not np.array_equal(p.track_static, first.track_static) or p.driver_names != first.driver_names:
                raise ValueError("cannot concatenate sequences of different tracks")
        data = {f: getattr(first, f) for f in first.__dataclass_fields__}
        for name in _TIME_ARRAYS:
            data[name] = np.concatenate([getattr(p, name) for p in parts])
        steps = np.diff(data["times"])
        regular = len(steps) > 0 and bool(np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9))
        data["step_s"] = float(steps[0]) if regular else None
        return cls(**data).validate()

    # ------------------------------------------------------------------ views
    def slice(self, start: int, stop: int, step: int = 1) -> RaceSequence:
        """Sub-sequence of frames ``start:stop:step`` (arrays are views, nothing is copied)."""
        sl = slice(start, stop, step)
        data = {f: getattr(self, f) for f in self.__dataclass_fields__}
        for name in _TIME_ARRAYS:
            data[name] = getattr(self, name)[sl]
        if self.step_s is not None:
            data["step_s"] = self.step_s * step
        return RaceSequence(**data)

    def decimate(self, every: int) -> RaceSequence:
        """Keep every ``every``-th frame (e.g. 10 Hz data -> 2 Hz for an animation)."""
        if every < 1:
            raise ValueError("every must be >= 1")
        return self.slice(0, self.n_frames, every)

    def windows(self, length: int, stride: int = 1) -> Iterator[RaceSequence]:
        """Sliding windows of ``length`` consecutive frames - the typical training sample for a sequence model."""
        if length < 1 or stride < 1:
            raise ValueError("length and stride must be >= 1")
        for start in range(0, self.n_frames - length + 1, stride):
            yield self.slice(start, start + length)

    def driver_order(self, i: int) -> np.ndarray:
        """Slot indices of the drivers present in frame ``i``, leader first (by the ``position`` feature)."""
        present = np.flatnonzero(self.driver_present[i])
        pos = self.driver_x[i, present, self.driver_names.index("position")]
        return present[np.argsort(pos, kind="stable")]

    def frame(self, i: int, *, normalise: bool = False, pool: int = 1) -> GraphArrays:
        """Reconstruct snapshot ``i`` as ``GraphArrays`` (drivers ordered leader first, like the exporter)."""
        if not -self.n_frames <= i < self.n_frames:
            raise IndexError(f"frame {i} out of range for {self.n_frames} frames")
        i %= self.n_frames
        N, order = self.n_track, self.driver_order(i)
        track_names = self.track_names
        track_x = np.zeros((N, len(track_names)), dtype=np.float32)
        track_mask = np.ones(track_x.shape, dtype=bool)
        for j, name in enumerate(self.track_static_names):
            col = track_names.index(name)
            track_x[:, col] = self.track_static[:, j]
            track_mask[:, col] = self.track_static_mask[:, j]
        for k, name in enumerate(FLAG_BITS):
            track_x[:, track_names.index(name)] = (self.track_flags[i] >> k) & 1
        nodes = self.driver_to_track[i, order]
        track_x[:, track_names.index("n_drivers")] = np.bincount(nodes[nodes >= 0], minlength=N)[:N]

        idx = np.arange(N)
        ring = np.concatenate([np.stack([idx, (idx + 1) % N]), np.stack([(idx + 1) % N, idx])], axis=1)
        sel = np.ix_(order, order)
        src, dst = np.nonzero(self.edge_exists[i][sel])  # row-major, the same order the graph yields its edges
        arrays = GraphArrays(
            time_s=float(self.times[i]), track_length_m=self.track_length_m, track_spacing_m=self.track_spacing_m,
            track_x=track_x, track_mask=track_mask, track_names=list(track_names),
            driver_x=self.driver_x[i, order].copy(), driver_mask=self.driver_mask[i, order].copy(),
            driver_names=list(self.driver_names), driver_codes=[self.driver_codes[s] for s in order],
            global_x=self.global_x[i : i + 1].copy(), global_mask=self.global_mask[i : i + 1].copy(),
            global_names=list(self.global_names), track_edge_index=ring,
            driver_to_track=nodes.astype(np.int64),
            driver_edge_index=np.stack([src, dst]).astype(np.int64),
            driver_edge_attr=self.edge_attr[i][sel][src, dst].astype(np.float32),
            driver_edge_mask=self.edge_attr_mask[i][sel][src, dst],
            edge_names=list(self.edge_names), xy_bbox=tuple(float(v) for v in self.xy_bbox),
            race_distance_m=self.race_distance_m,
        ).validate()  # fmt: skip
        if normalise:
            arrays = arrays.normalised_copy()
        return arrays.pool_track(pool) if pool > 1 else arrays

    def frames(self, **kwargs: Any) -> Iterator[GraphArrays]:
        """Iterate over all frames as ``GraphArrays`` (keyword arguments as for :meth:`frame`)."""
        for i in range(self.n_frames):
            yield self.frame(i, **kwargs)

    # ------------------------------------------------------------------ quality
    def continuity_report(
        self, speed_factor: float = 1.5, slack_m: float = 3.0, regression_m: float = 5.0
    ) -> ContinuityReport:
        """Check that the sequence is physically continuous.

        A car must not move further between two frames than ``speed_factor`` x its speed x dt + ``slack_m``
        (otherwise it teleported - bad data or a wrong time step), and its race progress must not run backwards by
        more than ``regression_m``. Gaps in the time grid are accounted for through dt."""
        names = self.driver_names
        xi, yi, vi, pi = (names.index(n) for n in ("x_m", "y_m", "speed_kmh", "progress_m"))
        frac = self.driver_x[:, :, names.index("lap_fraction")]
        report = ContinuityReport(n_frames=self.n_frames, max_position_step_ratio=0.0)
        if self.n_frames < 2:
            return report
        dt = np.diff(self.times)[:, None]
        both = self.driver_present[:-1] & self.driver_present[1:]
        for col in (xi, yi, vi, pi):
            both &= self.driver_mask[:-1, :, col] & self.driver_mask[1:, :, col]
        x, y, v, prog = (self.driver_x[:, :, c] for c in (xi, yi, vi, pi))
        step = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
        vmax = np.maximum(v[:-1], v[1:]) / 3.6
        allowed = speed_factor * vmax * dt + slack_m
        moving = both & (vmax * dt > 1.0)
        if moving.any():
            report.max_position_step_ratio = float((step[moving] / (vmax * dt)[moving]).max())
        for t, d in zip(*np.nonzero(both & (step > allowed)), strict=True):
            report.position_jumps.append((int(t) + 1, self.driver_codes[d], float(step[t, d]), float(allowed[t, d])))
        for t, d in zip(*np.nonzero(both & (np.diff(prog, axis=0) < -regression_m)), strict=True):
            report.progress_regressions.append((int(t) + 1, self.driver_codes[d], float(np.diff(prog, axis=0)[t, d])))
        report.lap_wraps = int((both & (np.diff(frac, axis=0) < -0.5)).sum())
        return report

    # ------------------------------------------------------------------ persistence
    def to_parquet(self, path: str | Path, overwrite: bool = False, chunk_frames: int = DEFAULT_CHUNK_FRAMES) -> Path:
        """Write this sequence as a Parquet dataset directory (see :mod:`f1strategy.sequence.parquet`)."""
        from f1strategy.sequence.parquet import write_dataset

        return write_dataset(self, path, overwrite=overwrite, chunk_frames=chunk_frames)

    @classmethod
    def from_parquet(
        cls, path: str | Path, start_frame: int = 0, stop_frame: int | None = None, every: int = 1
    ) -> RaceSequence:
        """Read (a part of) a Parquet dataset back into a dense sequence."""
        from f1strategy.sequence.parquet import ParquetSequence

        return ParquetSequence.open(path).read_sequence(start_frame, stop_frame, every)

    def save(self, path: str | Path) -> None:
        """Write a compressed ``.npz`` (arrays + a JSON blob with names and scalars)."""
        meta = {
            "format_version": FORMAT_VERSION, "driver_codes": self.driver_codes,
            "track_static_names": self.track_static_names, "driver_names": self.driver_names,
            "global_names": self.global_names, "edge_names": self.edge_names, "track_names": self.track_names,
            "track_length_m": self.track_length_m, "track_spacing_m": self.track_spacing_m,
            "race_distance_m": self.race_distance_m, "step_s": self.step_s, "flag_bits": FLAG_BITS,
        }  # fmt: skip
        arrays = {name: getattr(self, name) for name in _ARRAY_FIELDS}
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> RaceSequence:
        """Read a file written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta["format_version"] != FORMAT_VERSION:
                raise ValueError(f"unsupported sequence format version {meta['format_version']}")
            if meta.get("flag_bits") != FLAG_BITS:  # the bit order is part of the file format
                raise ValueError(f"flag bit layout of the file {meta.get('flag_bits')} differs from {FLAG_BITS}")
            arrays = {name: z[name] for name in _ARRAY_FIELDS}
        scalars = {k: meta[k] for k in ("track_length_m", "track_spacing_m", "race_distance_m", "step_s")}
        names = {k: meta[k] for k in ("driver_codes", "track_static_names", "driver_names", "global_names",
                                      "edge_names", "track_names")}  # fmt: skip
        return cls(**arrays, **names, **scalars).validate()

    def __repr__(self) -> str:
        span = f"{self.times[0]:.1f}-{self.times[-1]:.1f} s" if self.n_frames else "empty"
        step = f", step {self.step_s:g} s" if self.step_s else ""
        return (f"RaceSequence({self.n_frames} frames, {span}{step}, {self.n_drivers} drivers, "
                f"{self.n_track} track nodes)")  # fmt: skip

    @property
    def nbytes(self) -> int:
        """In-memory size of all arrays."""
        return sum(getattr(self, name).nbytes for name in _ARRAY_FIELDS)
