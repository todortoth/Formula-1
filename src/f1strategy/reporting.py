"""
reporting.py · Formula 1 Strategic Decisions
=========================================================
Console tables for the lap-level driver snapshot (position, tyres, pace, nearby drivers, sector gaps).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TextIO

import pandas as pd

from f1strategy.domain.driver import DriverNode
from f1strategy.utils import format_time_mmssms


def _nearby_str(driver: DriverNode) -> str:
    if not driver.nearby_drivers:
        return "None"
    return ", ".join(
        f"{nb['driver_code']} ({'+' if nb['status'] == 'behind' else '-'}{abs(nb['time_gap'])}s)"
        for nb in driver.nearby_drivers
    )


def print_driver_table(lap_snapshot: Sequence[DriverNode], target_lap: int, file: TextIO | None = None) -> None:
    """Position, tyre, pace, current sector status and nearby drivers (within 5 s) per driver."""
    out = {"file": file}
    print(f"\n--- Summarized Data at lap {target_lap}. ---", **out)
    print(
        f"{'Driver':<8} | {'Pos':<4} | {'Tyre':<8} | {'Age':<4} | {'Pits':<5} | {'Avg Lap (s)':<12} | "
        f"{'Current Sector Status':<22} | Nearby Drivers (within 5s)",
        **out,
    )
    print("-" * 135, **out)
    for d in lap_snapshot:
        if d.current_sector_node:
            sector_status = f"{d.current_sector} ({d.current_sector_node.status_message})"
        else:
            sector_status = f"{d.current_sector or 'Unknown'}"
        print(
            f"{d.driver_code:<8} | {d.position!s:<4} | {d.compound!s:<8} | {d.tyre_age!s:<4} | {d.pit_stops:<5} | "
            f"{format_time_mmssms(d.last_3_laps_average_seconds):<12} | {sector_status:<22} | {_nearby_str(d)}",
            **out,
        )


def print_sector_info(lap_snapshot: Sequence[DriverNode], file: TextIO | None = None) -> None:
    """Gap to the leader at the moment the leader finishes each sector."""
    out = {"file": file}
    print("\n--- Sector Info ---", **out)
    for sector_name in ("Sector 1", "Sector 2", "Sector 3"):
        print(f"--- Leader finished: {sector_name} ---", **out)
        print(f"{'Driver':<8} | {'Sector':<12} | {'Lap':<5} | Leader Gap", **out)
        print("-" * 55, **out)
        for d in lap_snapshot:
            info = d.sector_snapshots.get(sector_name)
            if not info:
                continue
            if d.position == 1:
                gap = "Leader"
            elif info["lap_diff"] > 0:
                gap = f"+{info['lap_diff']} lap" if info["lap_diff"] == 1 else f"+{info['lap_diff']} laps"
            elif pd.isna(info["time_gap"]):
                gap = "N/A"
            else:
                gap = f"{'+' if info['time_gap'] > 0 else ''}{info['time_gap']:.3f}s"
            print(f"{d.driver_code:<8} | {info['sector_at_moment']:<12} | Lap {info['lap_number']:<5} | {gap}", **out)
        print("\n", **out)
