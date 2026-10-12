"""
cli.py · Formula 1 Strategic Decisions
=========================================================
Command line entry point (``python main.py`` or ``f1-graph``).

Loads a session, prints the lap-level tables, then builds the race graph for each requested sector of the
target lap, plots it on the track layout and exports the model-ready arrays as ``.npz``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from f1strategy.animation import RaceAnimator
from f1strategy.graph.exporter import GraphExporter, suggest_pool_factor
from f1strategy.plotting import plot_race_graph
from f1strategy.reporting import print_driver_table, print_sector_info
from f1strategy.sequence import resolve_step
from f1strategy.sequence.sampling import DEFAULT_CHUNK_FRAMES
from f1strategy.session import SessionAnalyser

SECTORS = ("Sector 1", "Sector 2", "Sector 3")


@dataclass
class RunResult:
    """Files written by :func:`run`."""

    npz_files: list[str] = field(default_factory=list)
    png_files: list[str] = field(default_factory=list)


@dataclass
class ContinuousResult:
    """Files written by :func:`run_continuous`."""

    n_frames: int
    continuity_ok: bool
    parquet_dir: str | None = None
    sequence_file: str | None = None  # the .npz, if requested
    gif_file: str | None = None


def _event(value: str) -> str | int:
    """``"5"`` means round 5 (FastF1 only treats an *int* as a round number), anything else is an event name."""
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="f1-graph", description="Build FastF1 race graphs for ML models.")
    p.add_argument("--year", type=int, default=2025, help="season (default: 2025)")
    p.add_argument(
        "--event",
        type=_event,
        default="Belgian Grand Prix",
        help="event name or round number (default: Belgian Grand Prix)",
    )
    p.add_argument("--session", default="R", help="session identifier: R, Q, S, SQ, FP1.. (default: R)")
    p.add_argument("--lap", type=int, default=44, help="lap whose sector crossings define the snapshots (default: 44)")
    p.add_argument(
        "--sectors",
        nargs="+",
        choices=SECTORS,
        default=list(SECTORS),
        metavar="SECTOR",
        help="'Sector 1' 'Sector 2' 'Sector 3' (default: all)",
    )
    p.add_argument("--out-dir", default="graphs", help="output directory (default: graphs)")
    p.add_argument("--spacing", type=float, default=10.0, help="track node spacing in metres (default: 10)")
    p.add_argument("--max-tokens", type=int, default=512, help="token budget used to pick the track pooling factor")
    p.add_argument(
        "--drs-detection-offset",
        type=float,
        default=None,
        help="estimate DRS detection points as zone start minus this many metres",
    )
    p.add_argument(
        "--cache-dir", default=None, help="FastF1 cache directory (default: $F1_CACHE_DIR or ~/fastf1_cache)"
    )
    c = p.add_argument_group("continuous data (dense time sampling + GIF)")
    c.add_argument(
        "--continuous",
        action="store_true",
        help="also sample the race graph densely over a lap window, save it and render a GIF",
    )
    c.add_argument("--from-lap", type=int, default=None, help="first lap of the window (default: --lap)")
    c.add_argument("--to-lap", type=int, default=None, help="last lap of the window (default: --from-lap)")
    c.add_argument("--step", type=float, default=None, help="seconds between two frames (default: 1.0, or 1/--hz)")
    c.add_argument(
        "--hz",
        type=float,
        default=None,
        help="frames per second of *data*, e.g. 10 (alternative to --step; not the GIF rate, see --fps)",
    )
    c.add_argument(
        "--format",
        dest="dataset_format",
        choices=("parquet", "npz", "both"),
        default="parquet",
        help="dataset format: a Parquet directory (streamed, default), one .npz, or both",
    )
    c.add_argument(
        "--chunk-frames",
        type=int,
        default=DEFAULT_CHUNK_FRAMES,
        help=f"frames per Parquet part / in-memory chunk (bounds the memory use; default: {DEFAULT_CHUNK_FRAMES})",
    )
    c.add_argument(
        "--gif-every",
        type=int,
        default=0,
        help="use every n-th frame in the GIF (default: automatic, at most 300 frames)",
    )
    c.add_argument("--fps", type=float, default=10.0, help="playback frames per second of the GIF (default: 10)")
    c.add_argument("--no-gif", action="store_true", help="in continuous mode: only write the dataset, no GIF")
    c.add_argument(
        "--no-snapshots",
        action="store_true",
        help="skip the sector snapshots (tables, plots, npz); requires --continuous",
    )
    p.add_argument("--no-tables", action="store_true", help="skip the console tables (sector snapshots)")
    p.add_argument("--no-plots", action="store_true", help="skip the PNG plots")
    p.add_argument("--no-export", action="store_true", help="skip the .npz export")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def run(
    analyzer: SessionAnalyser,
    location: str,
    target_lap: int,
    out_dir: str = "graphs",
    sectors: Sequence[str] = SECTORS,
    max_tokens: int = 512,
    tables: bool = True,
    plots: bool = True,
    export: bool = True,
) -> RunResult:
    """Everything ``main`` does after the session is loaded; separated so it can run on any analyser."""
    result = RunResult()
    if tables:
        lap_snapshot = analyzer.all_drivers_at_lap(target_lap=target_lap)
        print_driver_table(lap_snapshot, target_lap)
        print_sector_info(lap_snapshot)
        try:
            print("\n--- Results ---")
            print(analyzer.session.results[["Abbreviation", "GridPosition", "Position", "Status"]])
        except Exception as exc:  # results are not available for every session / stand-in
            print(f"(results unavailable: {exc})")

    print(f"\nTrack: {analyzer.track}")
    os.makedirs(out_dir, exist_ok=True)
    exporter = GraphExporter(normalise=True)
    for sector_name in sectors:
        print(f"Building Graph ({sector_name}) at lap {target_lap}...")
        graph = analyzer.build_session_graph(target_lap=target_lap, sector_name=sector_name)
        g = graph.graph["global"]
        print(
            f"Graph successfully created ({sector_name})! Nodes: {graph.number_of_nodes()}, "
            f"Edges: {graph.number_of_edges()}, track status: {g['track_message']}, "
            f"yellow sectors: {g['n_yellow_sectors']}, DRS enabled: {g['drs_enabled']}"
        )
        suffix = sector_name.lower().replace(" ", "")

        if export:
            arrays = exporter.export(graph)
            factor = suggest_pool_factor(len(arrays.track_x), len(arrays.driver_codes), max_tokens=max_tokens)
            pooled = arrays.pool_track(factor)
            print(
                f"  model input: {pooled.n_tokens} tokens (track pooling factor {factor} for a {max_tokens} token limit)"
            )
            path = f"{out_dir}/graph_lap{target_lap}_{suffix}.npz"
            np.savez_compressed(path, **pooled.to_dict())
            result.npz_files.append(path)
        if plots:
            path = f"{out_dir}/graph_{suffix}.png"
            plot_race_graph(graph, path, f"F1 Race Graph - LAP {target_lap} ({sector_name} - {location})")
            print(f"Saved to: '{path}'\n")
            result.png_files.append(path)
    return result


def run_continuous(
    analyzer: SessionAnalyser,
    location: str,
    from_lap: int,
    to_lap: int | None = None,
    step_s: float | None = None,
    out_dir: str = "graphs",
    gif: bool = True,
    gif_every: int = 0,
    fps: float = 10.0,
    hz: float | None = None,
    dataset_format: str = "parquet",
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
) -> ContinuousResult:
    """Continuous mode: sample the race every ``step_s`` seconds (or at ``hz``) from the moment the leader starts
    ``from_lap`` until the leader completes ``to_lap``, check continuity, save the dataset and render a GIF.

    With a Parquet dataset the race is streamed to disk in chunks (bounded memory), continuity is checked in Polars and
    the GIF is drawn from a decimated read of the dataset; the in-memory path is only used for ``--format npz``."""
    # Auto detect max laps
    try:
        max_session_laps = int(analyzer.session.laps['LapNumber'].max())
        if to_lap is None or to_lap > max_session_laps:
            to_lap = max_session_laps
    except Exception:
        pass

    to_lap = from_lap if to_lap is None else to_lap
    step = resolve_step(step_s, hz)
    os.makedirs(out_dir, exist_ok=True)
    start, end = analyzer.graph_builder.lap_window(from_lap, to_lap)
    laps = f"{from_lap}" if to_lap == from_lap else f"{from_lap}-{to_lap}"
    rates = analyzer.telemetry_rates()
    print(
        f"\nContinuous mode: laps {laps} = session time {start:.1f}-{end:.1f} s, one frame every {step:g} s ({1 / step:g} Hz)"
    )
    print(
        f"  real telemetry: position {rates['position_hz']:.1f} Hz, car data {rates['car_hz']:.1f} Hz "
        f"(merged ~{rates['merged_hz']:.1f} Hz); {1 / step:g} Hz data interpolates between real samples "
        "(see pos_age_s / car_age_s)"
    )

    last_pct = -1

    def show(done: int, total: int) -> None:
        nonlocal last_pct
        pct = 100 * done // total
        if pct // 10 != last_pct // 10 or done == total:
            print(f"  {pct:3d}%  ({done}/{total} frames)")
        last_pct = pct

    use_parquet = dataset_format in ("parquet", "both")
    use_npz = dataset_format in ("npz", "both")
    result = ContinuousResult(n_frames=0, continuity_ok=True)
    dataset = seq = None

    if use_parquet:
        result.parquet_dir = f"{out_dir}/sequence_lap{laps}"
        dataset = analyzer.write_sequence_parquet(
            result.parquet_dir, start, end, step, chunk_frames=chunk_frames, overwrite=True, progress=show
        )
        report = dataset.continuity_report()
        result.n_frames = dataset.n_frames
        print(f"  {dataset}\n  {report.summary()}\nSaved Parquet dataset to: '{result.parquet_dir}'")
    if use_npz:
        seq = (
            dataset.read_sequence() if dataset is not None else analyzer.build_sequence(start, end, step, progress=show)
        )
        report = seq.continuity_report()
        result.sequence_file = f"{out_dir}/sequence_lap{laps}.npz"
        seq.save(result.sequence_file)
        result.n_frames = seq.n_frames
        print(f"  {seq}\n  {report.summary()}\nSaved dataset to: '{result.sequence_file}'")
    result.continuity_ok = report.ok

    if gif:
        every = gif_every if gif_every > 0 else max(1, -(-result.n_frames // 300))
        if dataset is not None:  # read only every n-th frame: memory stays small however long the race is
            seq_gif, gif_step = dataset.read_sequence(0, None, every), every * step
        else:
            seq_gif, gif_step = seq.decimate(every), every * step
        result.gif_file = f"{out_dir}/sequence_lap{laps}.gif"
        animator = RaceAnimator(seq_gif, colors=analyzer.driver_colors(), title=f"{location} - laps {laps}")
        animator.save_gif(result.gif_file, every=1, fps=fps, progress=show)
        print(f"Saved animation to: '{result.gif_file}' ({seq_gif.n_frames} frames, {gif_step * fps:g}x real time)")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    # basicConfig does nothing if the root logger is already configured (notebooks, embedding): set our own level too
    logging.getLogger("f1strategy").setLevel(level)
    if args.no_snapshots and not args.continuous:
        parser.error("--no-snapshots would leave nothing to do: add --continuous")
    if args.step is not None and args.hz is not None:
        parser.error("use either --step or --hz, not both")
    if args.hz is not None and args.hz <= 0:
        parser.error("--hz must be positive")
    analyzer = SessionAnalyser(
        year=args.year,
        location=args.event,
        session_type=args.session,
        track_spacing_m=args.spacing,
        drs_detection_offset_m=args.drs_detection_offset,
        cache_dir=args.cache_dir,
    )
    if not args.no_snapshots:
        run(
            analyzer,
            args.event,
            args.lap,
            out_dir=args.out_dir,
            sectors=args.sectors,
            max_tokens=args.max_tokens,
            tables=not args.no_tables,
            plots=not args.no_plots,
            export=not args.no_export,
        )
    if args.continuous:
        first = args.from_lap if args.from_lap is not None else args.lap
        run_continuous(
            analyzer,
            args.event,
            first,
            args.to_lap,
            step_s=args.step,
            out_dir=args.out_dir,
            gif=not args.no_gif,
            gif_every=args.gif_every,
            fps=args.fps,
            hz=args.hz,
            dataset_format=args.dataset_format,
            chunk_frames=args.chunk_frames,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
