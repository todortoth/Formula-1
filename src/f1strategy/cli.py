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

from f1strategy.graph.exporter import GraphExporter, suggest_pool_factor
from f1strategy.plotting import plot_race_graph
from f1strategy.reporting import print_driver_table, print_sector_info
from f1strategy.session import SessionAnalyser

SECTORS = ("Sector 1", "Sector 2", "Sector 3")


@dataclass
class RunResult:
    """Files written by :func:`run`."""

    npz_files: list[str] = field(default_factory=list)
    png_files: list[str] = field(default_factory=list)


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
    p.add_argument("--no-tables", action="store_true", help="skip the console tables")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    # basicConfig does nothing if the root logger is already configured (notebooks, embedding): set our own level too
    logging.getLogger("f1strategy").setLevel(level)
    analyzer = SessionAnalyser(
        year=args.year,
        location=args.event,
        session_type=args.session,
        track_spacing_m=args.spacing,
        drs_detection_offset_m=args.drs_detection_offset,
        cache_dir=args.cache_dir,
    )
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
