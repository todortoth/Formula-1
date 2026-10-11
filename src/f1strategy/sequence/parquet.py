"""
parquet.py · Formula 1 Strategic Decisions
=========================================================
Parquet export of continuous race data, built on Polars - for sequences too big to hold in memory
(a 90 minute race at 10 Hz is ~54 000 frames x 20 drivers = ~1.1 million driver rows).

Layout of a dataset directory
-----------------------------
::

    manifest.json              schema, driver slots, chunk index, scalars (written last: no manifest = incomplete)
    track.parquet              one row per track node: the static features (written once)
    drivers/part-00000.parquet one row per (frame, driver present): frame, t, driver, slot, track_node, all features
    global/part-00000.parquet  one row per frame: frame, t, global features, flagged marshal sectors
    edges/part-00000.parquet   one row per proximity edge (frame, src, dst, edge features)

Parts are written chunk by chunk while the race is processed, so memory stays bounded, and they are time ordered, so
Parquet statistics let ``scan`` queries on ``frame`` / ``t`` skip everything outside the requested range.

Missing values (the exporter's mask) are real Parquet **nulls**; zero-filled values never leak into the files.
Flags are stored compactly: ``yellow_sectors`` / ``double_yellow_sectors`` list the flagged marshal sectors of a frame
(a per-node flag is "the node's marshal sector is in the list"), while SC / VSC / VSC ending / red are the global
features ``is_sc`` / ``is_vsc`` / ``is_vsc_ending`` / ``is_red`` (they apply to every node).

Reading
-------
* ``scan_drivers()`` / ``scan_edges()`` / ``scan_global()`` return Polars ``LazyFrame`` s for out-of-core analysis;
* ``read_sequence(start_frame, stop_frame, every)`` rebuilds a dense :class:`RaceSequence` for a frame range (decimated if
  asked) - the same object the in-memory builder returns, so ``frame()``, windows and the GIF renderer work unchanged;
* ``iter_chunks()`` walks the dataset in bounded-size sequences, e.g. for a training loop;
* ``continuity_report()`` runs the continuity check *in Polars* over the whole dataset, without densifying it.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from f1strategy.sequence.dataset import (
    FLAG_BITS,
    FORMAT_VERSION,
    ContinuityReport,
    RaceSequence,
)
from f1strategy.sequence.sampling import DEFAULT_CHUNK_FRAMES

log = logging.getLogger(__name__)

PARQUET_FORMAT = "f1strategy-parquet"
MANIFEST = "manifest.json"
_TABLES = ("drivers", "global", "edges")
_SECTOR_LISTS = ("yellow_sectors", "double_yellow_sectors")


def _part_name(i: int) -> str:
    return f"part-{i:05d}.parquet"


def _float_column(name: str, values: np.ndarray, mask: np.ndarray) -> pl.Series:
    """Float32 column where ``mask == False`` becomes null (the values behind the mask are zero-filled)."""
    arr = np.where(mask, values, np.nan).astype(np.float32)
    return pl.Series(name, arr, dtype=pl.Float32, nan_to_null=True)


def _sector_lists(seq: RaceSequence) -> tuple[list[list[int]], list[list[int]]]:
    """Per frame: the marshal sectors under yellow / double yellow (derived from the packed per-node flags)."""
    ms = seq.track_static[:, seq.track_static_names.index("marshal_sector")].astype(int)
    known = ms > 0  # -1 = unknown marshal sector: such nodes can never carry a sector flag
    out = []
    for bit in (FLAG_BITS.index("flag_yellow"), FLAG_BITS.index("flag_double_yellow")):
        per_frame = []
        for flags in seq.track_flags:
            on = ((flags >> bit) & 1).astype(bool) & known
            per_frame.append(sorted(int(s) for s in np.unique(ms[on])))
        out.append(per_frame)
    return out[0], out[1]


class ParquetSequenceWriter:
    """Streams :class:`RaceSequence` chunks into a dataset directory (bounded memory).

    ::

        with ParquetSequenceWriter("race_ds") as w:
            for chunk in builder.build_chunks(times, chunk_frames=2000):
                w.append(chunk)
    """

    def __init__(
        self, path: str | Path, overwrite: bool = False, compression: str = "zstd", compression_level: int = 3
    ) -> None:
        self.path = Path(path)
        if self.path.exists() and any(self.path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{self.path} is not empty (pass overwrite=True to replace it)")
            for child in self.path.iterdir():
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        self.path.mkdir(parents=True, exist_ok=True)
        for table in _TABLES:
            (self.path / table).mkdir(exist_ok=True)
        self._compression = (compression, compression_level)
        self._first: RaceSequence | None = None
        self._frames = 0
        self._chunks: list[dict[str, Any]] = []
        self._step: float | None = None
        self._step_conflict = False
        self._last_time = -np.inf
        self._closed = False

    # ------------------------------------------------------------------ writing
    def _write(self, df: pl.DataFrame, table: str, index: int) -> None:
        compression, level = self._compression
        df.write_parquet(
            self.path / table / _part_name(index), compression=compression, compression_level=level, statistics=True
        )

    def append(self, seq: RaceSequence) -> None:
        """Write one chunk. Chunks must be consecutive in time and come from the same session and driver slots."""
        if self._closed:
            raise ValueError("writer is closed")
        if seq.n_frames == 0:
            return
        if self._first is None:
            self._first = seq
            self._write_track(seq)
        else:
            first = self._first
            if (seq.driver_codes, seq.driver_names, seq.global_names, seq.edge_names, seq.track_static_names) != (
                first.driver_codes, first.driver_names, first.global_names, first.edge_names, first.track_static_names,
            ):  # fmt: skip
                raise ValueError("chunk schema or driver slots differ from the first chunk")
            if not np.array_equal(seq.track_static, first.track_static):
                raise ValueError("chunk comes from a different track")
        if seq.times[0] <= self._last_time:
            raise ValueError("chunks must be consecutive in time (this one starts before the previous one ended)")

        index, offset = len(self._chunks), self._frames
        frame_idx = np.arange(seq.n_frames) + offset
        codes = np.array(seq.driver_codes)

        fi, si = np.nonzero(seq.driver_present)
        drivers = pl.DataFrame(
            {
                "frame": pl.Series(frame_idx[fi], dtype=pl.Int32),
                "t": pl.Series(seq.times[fi], dtype=pl.Float64),
                "driver": pl.Series(codes[si].tolist(), dtype=pl.String),
                "slot": pl.Series(si, dtype=pl.UInt8),
                "track_node": pl.Series(seq.driver_to_track[fi, si], dtype=pl.Int32),
            }
        ).with_columns(
            [
                _float_column(n, seq.driver_x[fi, si, j], seq.driver_mask[fi, si, j])
                for j, n in enumerate(seq.driver_names)
            ]
        )

        yellow, double = _sector_lists(seq)
        glob = pl.DataFrame(
            {"frame": pl.Series(frame_idx, dtype=pl.Int32), "t": pl.Series(seq.times, dtype=pl.Float64)}
        ).with_columns(
            [_float_column(n, seq.global_x[:, j], seq.global_mask[:, j]) for j, n in enumerate(seq.global_names)]
            + [
                pl.Series("yellow_sectors", yellow, dtype=pl.List(pl.Int16)),
                pl.Series("double_yellow_sectors", double, dtype=pl.List(pl.Int16)),
            ]
        )

        ef, eu, ev = np.nonzero(seq.edge_exists)
        edges = pl.DataFrame(
            {
                "frame": pl.Series(frame_idx[ef], dtype=pl.Int32),
                "t": pl.Series(seq.times[ef], dtype=pl.Float64),
                "src": pl.Series(codes[eu].tolist(), dtype=pl.String),
                "dst": pl.Series(codes[ev].tolist(), dtype=pl.String),
                "src_slot": pl.Series(eu, dtype=pl.UInt8),
                "dst_slot": pl.Series(ev, dtype=pl.UInt8),
            }
        ).with_columns(
            [
                _float_column(n, seq.edge_attr[ef, eu, ev, j], seq.edge_attr_mask[ef, eu, ev, j])
                for j, n in enumerate(seq.edge_names)
            ]
        )

        for table, df in (("drivers", drivers), ("global", glob), ("edges", edges)):
            self._write(df, table, index)
        self._chunks.append(
            {
                "index": index, "frame_start": offset, "frame_stop": offset + seq.n_frames,
                "t_start": float(seq.times[0]), "t_stop": float(seq.times[-1]),
                "driver_rows": drivers.height, "edge_rows": edges.height,
            }
        )  # fmt: skip
        self._frames += seq.n_frames
        self._last_time = float(seq.times[-1])
        if seq.step_s is not None:
            if self._step is None:
                self._step = seq.step_s
            elif not np.isclose(self._step, seq.step_s):
                self._step_conflict = True

    def _write_track(self, seq: RaceSequence) -> None:
        cols = [
            _float_column(n, seq.track_static[:, j], seq.track_static_mask[:, j])
            for j, n in enumerate(seq.track_static_names)
        ]
        pl.DataFrame({"node": pl.Series(np.arange(seq.n_track), dtype=pl.Int32)}).with_columns(cols).write_parquet(
            self.path / "track.parquet", compression=self._compression[0], compression_level=self._compression[1]
        )

    def close(self) -> Path:
        """Write the manifest (a dataset without one is incomplete and refuses to open)."""
        if self._closed:
            return self.path
        if self._first is None:
            raise ValueError("nothing was written")
        f = self._first
        manifest = {
            "format": PARQUET_FORMAT, "format_version": FORMAT_VERSION, "driver_codes": f.driver_codes,
            "track_static_names": f.track_static_names, "track_names": f.track_names,
            "driver_names": f.driver_names, "global_names": f.global_names, "edge_names": f.edge_names,
            "flag_bits": FLAG_BITS, "track_length_m": f.track_length_m, "track_spacing_m": f.track_spacing_m,
            "n_track": f.n_track, "xy_bbox": [float(v) for v in f.xy_bbox], "race_distance_m": f.race_distance_m,
            "step_s": None if self._step_conflict else self._step, "n_frames": self._frames,
            "t_start": self._chunks[0]["t_start"], "t_stop": self._chunks[-1]["t_stop"], "chunks": self._chunks,
        }  # fmt: skip
        (self.path / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        self._closed = True
        log.info("Wrote Parquet dataset %s (%d frames in %d chunks)", self.path, self._frames, len(self._chunks))
        return self.path

    def __enter__(self) -> ParquetSequenceWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()


class ParquetSequence:
    """Read access to a Parquet dataset written by :class:`ParquetSequenceWriter`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        manifest = self.path / MANIFEST
        if not manifest.is_file():
            raise FileNotFoundError(f"{self.path} has no {MANIFEST}: not a dataset, or the write was not finished")
        self.manifest: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
        m = self.manifest
        if m.get("format") != PARQUET_FORMAT or m.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported dataset format {m.get('format')!r} v{m.get('format_version')}")
        if m["flag_bits"] != FLAG_BITS:
            raise ValueError(f"flag bit layout of the dataset {m['flag_bits']} differs from {FLAG_BITS}")

    @classmethod
    def open(cls, path: str | Path) -> ParquetSequence:
        return cls(path)

    # ------------------------------------------------------------------ basics
    @property
    def n_frames(self) -> int:
        return int(self.manifest["n_frames"])

    @property
    def driver_codes(self) -> list[str]:
        return list(self.manifest["driver_codes"])

    @property
    def step_s(self) -> float | None:
        return self.manifest["step_s"]

    @property
    def t_range(self) -> tuple[float, float]:
        return float(self.manifest["t_start"]), float(self.manifest["t_stop"])

    def _files(self, table: str) -> list[str]:
        return [str(self.path / table / _part_name(c["index"])) for c in self.manifest["chunks"]]

    def read_track(self) -> pl.DataFrame:
        """The static track table (one row per node)."""
        return pl.read_parquet(self.path / "track.parquet")

    def scan_drivers(self) -> pl.LazyFrame:
        return pl.scan_parquet(self._files("drivers"))

    def scan_edges(self) -> pl.LazyFrame:
        return pl.scan_parquet(self._files("edges"))

    def scan_global(self) -> pl.LazyFrame:
        return pl.scan_parquet(self._files("global"))

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())

    def frame_range(self, start_s: float | None = None, end_s: float | None = None) -> tuple[int, int]:
        """``(start_frame, stop_frame)`` covering session times ``start_s <= t <= end_s``."""
        times = self.scan_global().select("t").collect().to_series().to_numpy()
        lo = 0 if start_s is None else int(np.searchsorted(times, start_s, side="left"))
        hi = len(times) if end_s is None else int(np.searchsorted(times, end_s, side="right"))
        return lo, hi

    def __repr__(self) -> str:
        t0, t1 = self.t_range
        step = f", step {self.step_s:g} s" if self.step_s else ""
        return (
            f"ParquetSequence({self.n_frames} frames, {t0:.1f}-{t1:.1f} s{step}, {len(self.driver_codes)} drivers, "
            f"{len(self.manifest['chunks'])} chunks, {self.disk_bytes() / 1e6:.1f} MB on disk)"
        )

    # ------------------------------------------------------------------ reading
    def read_sequence(self, start_frame: int = 0, stop_frame: int | None = None, every: int = 1) -> RaceSequence:
        """Rebuild a dense :class:`RaceSequence` for frames ``start_frame:stop_frame`` (every n-th frame).

        Only the requested rows are read (Parquet statistics skip the rest), so this is how a huge dataset is
        consumed piece by piece."""
        m = self.manifest
        stop_frame = self.n_frames if stop_frame is None else min(stop_frame, self.n_frames)
        if every < 1 or not 0 <= start_frame < stop_frame:
            raise ValueError(
                f"invalid frame range {start_frame}:{stop_frame} (every {every}) for {self.n_frames} frames"
            )
        T = len(range(start_frame, stop_frame, every))

        def select(lf: pl.LazyFrame) -> pl.LazyFrame:
            cond = (pl.col("frame") >= start_frame) & (pl.col("frame") < stop_frame)
            if every > 1:
                cond = cond & (((pl.col("frame") - start_frame) % every) == 0)
            return lf.filter(cond)

        def local(frames: pl.Series) -> np.ndarray:
            return ((frames.to_numpy().astype(np.int64) - start_frame) // every).astype(int)

        dnames, gnames, enames = m["driver_names"], m["global_names"], m["edge_names"]
        D = len(m["driver_codes"])

        # ---- global rows (one per frame) + flags
        g = select(self.scan_global()).sort("frame").collect()
        if g.height != T:
            raise ValueError(f"expected {T} frames, the dataset has {g.height} in that range")
        gx = g.select(gnames).to_numpy().astype(np.float64).reshape(T, len(gnames))
        global_mask = np.isfinite(gx)
        track = self.read_track()
        snames = m["track_static_names"]
        sx = track.select(snames).to_numpy().astype(np.float64).reshape(len(track), len(snames))
        static_mask = np.isfinite(sx)
        ms = sx[:, snames.index("marshal_sector")].astype(int)
        flag_cols = {n: g[n].to_numpy() > 0.5 for n in ("is_sc", "is_vsc", "is_vsc_ending", "is_red")}
        flags = np.zeros((T, len(track)), dtype=np.uint8)
        for t_idx, (yel, dbl) in enumerate(
            zip(g["yellow_sectors"].to_list(), g["double_yellow_sectors"].to_list(), strict=True)
        ):
            bits = {
                "flag_yellow": np.isin(ms, yel) if yel else False,
                "flag_double_yellow": np.isin(ms, dbl) if dbl else False,
                "flag_sc": flag_cols["is_sc"][t_idx],
                "flag_vsc": flag_cols["is_vsc"][t_idx],
                "flag_vsc_ending": flag_cols["is_vsc_ending"][t_idx],
                "flag_red": flag_cols["is_red"][t_idx],
            }
            for k, name in enumerate(FLAG_BITS):
                flags[t_idx] |= (np.broadcast_to(bits[name], (len(track),)).astype(np.uint8)) << k

        # ---- driver rows (sparse -> dense slots)
        dr = select(self.scan_drivers()).sort(["frame", "slot"]).collect()
        di, ds = local(dr["frame"]), dr["slot"].to_numpy().astype(int)
        dx = dr.select(dnames).to_numpy().astype(np.float64).reshape(dr.height, len(dnames))
        d_mask = np.isfinite(dx)
        driver_x = np.zeros((T, D, len(dnames)), dtype=np.float32)
        driver_mask = np.zeros(driver_x.shape, dtype=bool)
        present = np.zeros((T, D), dtype=bool)
        d2t = np.full((T, D), -1, dtype=np.int32)
        driver_x[di, ds] = np.where(d_mask, dx, 0.0)
        driver_mask[di, ds] = d_mask
        present[di, ds] = True
        d2t[di, ds] = dr["track_node"].to_numpy()

        # ---- edges (sparse -> dense)
        er = select(self.scan_edges()).collect()
        ei, eu, ev = local(er["frame"]), er["src_slot"].to_numpy().astype(int), er["dst_slot"].to_numpy().astype(int)
        ex = er.select(enames).to_numpy().astype(np.float64).reshape(er.height, len(enames))
        e_mask = np.isfinite(ex)
        edge_attr = np.zeros((T, D, D, len(enames)), dtype=np.float32)
        edge_attr_mask = np.zeros(edge_attr.shape, dtype=bool)
        edge_exists = np.zeros((T, D, D), dtype=bool)
        edge_attr[ei, eu, ev] = np.where(e_mask, ex, 0.0)
        edge_attr_mask[ei, eu, ev] = e_mask
        edge_exists[ei, eu, ev] = True

        step = self.step_s * every if self.step_s else None
        return RaceSequence(
            times=g["t"].to_numpy().astype(np.float64), driver_codes=self.driver_codes,
            track_static=np.where(static_mask, sx, 0.0).astype(np.float32), track_static_mask=static_mask,
            track_static_names=list(snames), track_flags=flags, driver_x=driver_x, driver_mask=driver_mask,
            driver_present=present, driver_to_track=d2t, driver_names=list(dnames),
            global_x=np.where(global_mask, gx, 0.0).astype(np.float32), global_mask=global_mask,
            global_names=list(gnames), edge_attr=edge_attr, edge_attr_mask=edge_attr_mask, edge_exists=edge_exists,
            edge_names=list(enames), track_length_m=m["track_length_m"], track_spacing_m=m["track_spacing_m"],
            xy_bbox=np.asarray(m["xy_bbox"], dtype=float), race_distance_m=m["race_distance_m"], step_s=step,
            track_names=list(m["track_names"]),
        ).validate()  # fmt: skip

    def iter_chunks(self, frames_per_chunk: int = DEFAULT_CHUNK_FRAMES, every: int = 1) -> Iterator[RaceSequence]:
        """Walk the dataset in consecutive, bounded-size sequences."""
        if frames_per_chunk < 1:
            raise ValueError("frames_per_chunk must be >= 1")
        for start in range(0, self.n_frames, frames_per_chunk * every):
            yield self.read_sequence(start, min(start + frames_per_chunk * every, self.n_frames), every)

    # ------------------------------------------------------------------ quality, in polars
    def continuity_report(
        self, speed_factor: float = 1.5, slack_m: float = 3.0, regression_m: float = 5.0
    ) -> ContinuityReport:
        """The same check as :meth:`RaceSequence.continuity_report`, evaluated lazily by Polars over the whole dataset
        (no dense arrays): a car must not move further between two consecutive frames than ``speed_factor`` x speed x dt
        + ``slack_m``, and its race progress must not run backwards by more than ``regression_m``."""
        by = "driver"
        lf = (
            self.scan_drivers()
            .select("frame", "t", "slot", by, "x_m", "y_m", "speed_kmh", "progress_m", "lap_fraction")
            .sort("slot", "frame")
            .with_columns(
                dframe=pl.col("frame").diff().over(by),
                dt=pl.col("t").diff().over(by),
                step=(pl.col("x_m").diff().over(by) ** 2 + pl.col("y_m").diff().over(by) ** 2).sqrt(),
                vmax=pl.when(pl.col("speed_kmh").is_not_null() & pl.col("speed_kmh").shift(1).over(by).is_not_null())
                .then(pl.max_horizontal(pl.col("speed_kmh"), pl.col("speed_kmh").shift(1).over(by)))
                .otherwise(None)
                / 3.6,
                dprog=pl.col("progress_m").diff().over(by),
                dfrac=pl.col("lap_fraction").diff().over(by),
            )
            .filter(
                pl.col("dframe") == 1
            )  # only truly consecutive frames; nulls (masked values) drop out of every test
            .with_columns(
                allowed=speed_factor * pl.col("vmax") * pl.col("dt") + slack_m, drive=pl.col("vmax") * pl.col("dt")
            )
        )
        df = lf.collect()
        report = ContinuityReport(n_frames=self.n_frames, max_position_step_ratio=0.0)
        ok = df.filter(pl.col("step").is_not_null() & pl.col("vmax").is_not_null() & pl.col("dprog").is_not_null())
        moving = ok.filter(pl.col("drive") > 1.0)
        if moving.height:
            report.max_position_step_ratio = float((moving["step"] / moving["drive"]).max())
        for row in ok.filter(pl.col("step") > pl.col("allowed")).sort("frame", "slot").iter_rows(named=True):
            report.position_jumps.append((int(row["frame"]), row[by], float(row["step"]), float(row["allowed"])))
        for row in ok.filter(pl.col("dprog") < -regression_m).sort("frame", "slot").iter_rows(named=True):
            report.progress_regressions.append((int(row["frame"]), row[by], float(row["dprog"])))
        report.lap_wraps = int(ok.filter(pl.col("dfrac") < -0.5).height)
        return report


def write_dataset(
    sequence: RaceSequence, path: str | Path, overwrite: bool = False, chunk_frames: int = DEFAULT_CHUNK_FRAMES
) -> Path:
    """Write an in-memory sequence as a Parquet dataset (in chunks of ``chunk_frames`` frames)."""
    with ParquetSequenceWriter(path, overwrite=overwrite) as writer:
        for start in range(0, sequence.n_frames, chunk_frames):
            writer.append(sequence.slice(start, min(start + chunk_frames, sequence.n_frames)))
    return Path(path)
