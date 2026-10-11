"""
DriverTimingClass.py · Formula 1 Strategic Decisions
=========================================================
Lap-level state of every driver at an arbitrary session time: tyre compound/age/stint, pit stops,
whether the car is in the pit lane right now, recent pace, classified position.

Only information that was already known at time t is used (laps whose ``Time`` <= t count as completed;
the lap in progress only contributes its tyre data, which is known at the start of the lap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from f1strategy.utils import td_seconds

_NAN = float("nan")
_DISTURBED = set("4567")  # SC, red, VSC, VSC ending anywhere in the lap -> lap time is not representative


@dataclass
class TimingState:
    lap_number: int
    laps_completed: int
    compound: str
    tyre_age: int
    fresh_tyre: bool
    stint: int
    pit_stops: int
    in_pit: bool
    last_lap_time_s: float
    pace_last3_s: float
    official_position: float
    finished_or_retired: bool


class DriverTiming:
    """Per-driver lap table with fast 'state at time t' lookups."""

    def __init__(self, laps: pd.DataFrame) -> None:
        self._d: dict[str, dict[str, np.ndarray]] = {
            str(code): _driver_table(group.sort_values("LapNumber")) for code, group in laps.groupby("Driver")
        }

    def has(self, code: str) -> bool:
        return code in self._d

    def last_end_time(self, code: str) -> float:
        e = self._d[code]["end"]
        return float(np.nanmax(e)) if np.isfinite(e).any() else _NAN

    def at(self, code: str, t: float) -> TimingState:
        d = self._d[code]
        n = len(d["lap"])
        cur = int(np.clip(np.searchsorted(d["start"], t, side="right") - 1, 0, n - 1))
        done = np.flatnonzero(d["end"] <= t)  # completed laps (NaN never compares True)
        last_done = int(done[-1]) if len(done) else -1

        # pit stops = pit lane entries so far. Deliberately a pure count of entries <= t: checking whether another
        # lap follows would look into the future, and a snapshot must only depend on what was known at t.
        entered = np.flatnonzero(d["pit_in"] <= t)
        pit_stops = len(entered)
        in_pit = False
        if len(entered):
            i = int(entered[-1])
            exit_t = d["pit_out"][i + 1] if i + 1 < n else np.inf
            in_pit = bool(t < (np.inf if np.isnan(exit_t) else exit_t))

        # pace: mean of the last 3 representative completed laps (no pit in/out lap, no SC/VSC/red)
        clean = [
            i
            for i in done[::-1]
            if np.isfinite(d["laptime"][i])
            and np.isnan(d["pit_in"][i])
            and np.isnan(d["pit_out"][i])
            and not (set(d["track_status"][i]) & _DISTURBED)
        ][:3]
        pace = float(np.mean(d["laptime"][clean])) if clean else _NAN

        end_all = self.last_end_time(code)
        return TimingState(
            lap_number=int(d["lap"][cur]),
            laps_completed=int(d["lap"][last_done]) if last_done >= 0 else 0,
            compound=str(d["compound"][cur]),
            tyre_age=int(d["tyre_life"][cur]) if np.isfinite(d["tyre_life"][cur]) else 0,
            fresh_tyre=bool(d["fresh"][cur]),
            stint=int(d["stint"][cur]) if np.isfinite(d["stint"][cur]) else 0,
            pit_stops=pit_stops,
            in_pit=in_pit,
            last_lap_time_s=float(d["laptime"][last_done]) if last_done >= 0 else _NAN,
            pace_last3_s=pace,
            official_position=float(d["position"][last_done]) if last_done >= 0 else _NAN,
            finished_or_retired=bool(np.isfinite(end_all) and t > end_all + 5.0),
        )


def _driver_table(g: pd.DataFrame) -> dict[str, np.ndarray]:
    """Column arrays of one driver's laps (sorted by lap number), missing columns become NaN."""
    n = len(g)

    def col(name: str, default: Any = np.nan) -> np.ndarray:
        return g[name].to_numpy() if name in g else np.full(n, default, dtype=object)

    def secs(name: str) -> np.ndarray:
        return td_seconds(g[name]) if name in g else np.full(n, np.nan)

    start, end = secs("LapStartTime"), secs("Time")
    for i in range(n):  # a missing lap start = end of the previous lap (or -inf for the first one)
        if np.isnan(start[i]):
            start[i] = end[i - 1] if i > 0 and not np.isnan(end[i - 1]) else (-np.inf if i == 0 else start[i - 1])
    start = np.maximum.accumulate(start)

    return {
        "start": start,
        "end": end,
        "lap": g["LapNumber"].to_numpy(float),
        "laptime": secs("LapTime"),
        "pit_in": secs("PitInTime"),
        "pit_out": secs("PitOutTime"),
        "compound": np.array([str(c).upper() if pd.notna(c) else "UNKNOWN" for c in col("Compound")]),
        "tyre_life": pd.to_numeric(pd.Series(col("TyreLife")), errors="coerce").to_numpy(float),
        "fresh": np.array([bool(v) if pd.notna(v) else False for v in col("FreshTyre", False)]),
        "stint": pd.to_numeric(pd.Series(col("Stint")), errors="coerce").to_numpy(float),
        "position": pd.to_numeric(pd.Series(col("Position")), errors="coerce").to_numpy(float),
        "track_status": np.array([str(v) if pd.notna(v) else "" for v in col("TrackStatus", "")]),
    }
