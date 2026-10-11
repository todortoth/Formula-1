"""
session.py · Formula 1 Strategic Decisions
=========================================================
The core data processor and the main entry point of the library. It introduces the FastF1 API, handles
caching, calculates nearby drivers, evaluates track statuses and builds the telemetry based race graph
(10 m track nodes + driver nodes, see :mod:`f1strategy.graph.builder`).

Typical use::

    analyser = SessionAnalyser(2025, "Belgian Grand Prix", "R")
    G = analyser.build_session_graph(target_lap=44, sector_name="Sector 2")
    arrays = GraphExporter().export(G)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from f1strategy.domain.driver import DriverNode
from f1strategy.domain.sector import SectorNode
from f1strategy.graph.builder import RaceGraphBuilder
from f1strategy.racecontrol.timeline import RaceControlTimeline
from f1strategy.telemetry.store import TelemetryStore
from f1strategy.telemetry.timing import DriverTiming
from f1strategy.track.model import TrackModel

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "~/fastf1_cache"


class SessionAnalyser:
    """Manages F1 session loading, telemetry extraction, and graph representation generation."""

    def __init__(
        self,
        year: int | None = None,
        location: str | int | None = None,
        session_type: str = "R",
        track_spacing_m: float = 10.0,
        drs_detection_points_m: Sequence[float] | None = None,
        drs_detection_offset_m: float | None = None,
        session: Any | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Loads the FastF1 session (with local caching) and prepares everything needed for the graph.

        Args:
            year, location, session_type: what ``fastf1.get_session`` takes (``location``: event name or round number) ("R", "Q", "S", ...).
            track_spacing_m: distance between two track nodes of the graph (default 10 m).
            drs_detection_points_m: optional DRS detection points as distance from the start/finish line (m);
                FastF1 does not provide them.
            drs_detection_offset_m: alternative: detection point = DRS zone start - this offset.
            session: an already loaded FastF1 session (skips the download; see also :meth:`from_session`).
            cache_dir: FastF1 cache directory; defaults to the ``F1_CACHE_DIR`` environment variable, then
                ``~/fastf1_cache``.
        """
        if session is None:
            if year is None or location is None:
                raise ValueError("Pass either `year` and `location`, or an already loaded `session`.")
            import fastf1  # imported lazily: heavy, and not needed when a session is injected

            cache = os.path.expanduser(cache_dir or os.environ.get("F1_CACHE_DIR") or DEFAULT_CACHE_DIR)
            os.makedirs(cache, exist_ok=True)
            fastf1.Cache.enable_cache(cache)

            session = fastf1.get_session(year, location, session_type)
            # telemetry (position + car data), messages and weather are all required for the graph
            session.load(laps=True, telemetry=True, weather=True, messages=True)
        self.session = session

        self._driver_map: dict[str, str] = self._build_driver_map()

        self.track = TrackModel.from_session(
            self.session,
            spacing_m=track_spacing_m,
            drs_detection_points_m=drs_detection_points_m,
            drs_detection_offset_m=drs_detection_offset_m,
        )
        self.timeline = RaceControlTimeline.from_session(self.session)
        self.timing = DriverTiming(self.session.laps)
        # FastF1 keys the telemetry by driver *number*, the rest of this class works with the 3 letter code
        self.telemetry = TelemetryStore(self.session, self.track, self._driver_map)
        self.graph_builder = RaceGraphBuilder(
            self.session, self.track, self.telemetry, self.timing, self.timeline, self._driver_map
        )

    @classmethod
    def from_session(cls, session: Any, **kwargs: Any) -> SessionAnalyser:
        """Build an analyser from an already loaded session object (FastF1 ``Session`` or a compatible stand-in)."""
        return cls(session=session, **kwargs)

    def _build_driver_map(self) -> dict[str, str]:
        """Maps internal FastF1 driver indentifiers to their standart three letter Abbreviations."""
        mapping: dict[str, str] = {}
        for driver in self.session.drivers:
            mapping[driver] = self.session.get_driver(driver)["Abbreviation"]
        return mapping

    def _get_leader_row(self, target_laps_df: pd.DataFrame) -> pd.Series:
        """Identifies and returns the lap dataframe row corresponding to the race leader."""
        leader_row = target_laps_df[target_laps_df["Position"] == 1]
        if leader_row.empty:
            return target_laps_df.iloc[0]
        return leader_row.iloc[0]

    def _compute_nearby_drivers(self, target_driver: str, target_laps_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Calculates drivers within a 5-second window of the target driver during a specific lap."""
        target_row = target_laps_df[target_laps_df["Driver"] == target_driver]
        if target_row.empty or pd.isna(target_row["Time"].values[0]):
            return []

        target_time = target_row["Time"].values[0] / np.timedelta64(1, "s")
        nearby_drivers: list[dict[str, Any]] = []

        for _, row in target_laps_df.iterrows():
            other_driver = row["Driver"]
            if other_driver == target_driver or pd.isna(row["Time"]):
                continue

            other_time = row["Time"] / np.timedelta64(1, "s")
            time_delta = other_time - target_time

            if abs(time_delta) <= 5.0:
                nearby_drivers.append(
                    {
                        "driver_code": other_driver,
                        "time_gap": round(time_delta, 3),
                        "status": "behind" if time_delta > 0 else "ahead",
                    }
                )

        return sorted(nearby_drivers, key=lambda x: abs(x["time_gap"]))

    def _get_track_status_at_time(self, crossing_time: Any, sector_name: str = "Track") -> SectorNode:
        """Finds active track status and returns a constructed SectorNode object."""
        track_statuses = self.session.track_status
        if track_statuses is None or track_statuses.empty or pd.isna(crossing_time):
            return SectorNode(sector_name, "1", "Clear")

        active_statuses = track_statuses[track_statuses["Time"] <= crossing_time]
        if active_statuses.empty:
            return SectorNode(sector_name, "1", "Clear")

        latest_status = active_statuses.iloc[-1]
        return SectorNode(
            sector_name=sector_name,
            track_status_code=str(latest_status["Status"]),
            status_message=str(latest_status["Message"]),
        )

    def _compute_snapshot_at_lap(self, target_lap: int, laps_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Computes sector checkpoint positions, lap counts and leader gaps for all drivers when the leader crosses sectors."""
        target_laps_df = laps_df[laps_df["LapNumber"] == target_lap]
        if target_laps_df.empty:
            return {}

        leader_row = self._get_leader_row(target_laps_df)
        leader_code: str = leader_row["Driver"]
        leader_lap_num: int = int(leader_row["LapNumber"])

        checkpoints: dict[str, tuple[Any, str]] = {
            "Sector 1": (leader_row["Sector1SessionTime"], "Sector1SessionTime"),
            "Sector 2": (leader_row["Sector2SessionTime"], "Sector2SessionTime"),
            "Sector 3": (leader_row["Sector3SessionTime"], "Sector3SessionTime"),
        }

        driver_laps_grouped = dict(iter(laps_df.groupby("Driver")))

        driver_base_status: dict[str, int] = {}
        for driver_code, driver_laps in driver_laps_grouped.items():
            d_target_lap = driver_laps[driver_laps["LapNumber"] == leader_lap_num]
            if d_target_lap.empty:
                d_target_lap = driver_laps[driver_laps["LapNumber"] < leader_lap_num].tail(1)
                if d_target_lap.empty:
                    d_target_lap = driver_laps.head(1)
            d_lap_num = int(d_target_lap["LapNumber"].values[0]) if not d_target_lap.empty else leader_lap_num
            driver_base_status[driver_code] = leader_lap_num - d_lap_num

        results: dict[str, dict[str, Any]] = {}
        for checkpoint_name, (crossing_time, time_col) in checkpoints.items():
            if pd.isna(crossing_time):
                continue

            sector_node_obj = self._get_track_status_at_time(crossing_time, sector_name=checkpoint_name)

            drivers_at_moment: list[dict[str, Any]] = []
            for driver_code, driver_laps in driver_laps_grouped.items():
                active_lap = driver_laps[
                    (driver_laps["LapStartTime"] <= crossing_time) & (driver_laps["Time"] > crossing_time)
                ]

                if active_lap.empty:
                    active_lap = driver_laps[driver_laps["LapStartTime"] <= crossing_time].tail(1)

                if not active_lap.empty:
                    lap_row = active_lap.iloc[0]
                    lap_num = int(lap_row["LapNumber"])

                    s1_t = lap_row["Sector1SessionTime"]
                    s2_t = lap_row["Sector2SessionTime"]

                    if pd.notna(s1_t) and crossing_time <= s1_t:
                        sec = "Sector 1"
                    elif pd.notna(s2_t) and crossing_time <= s2_t:
                        sec = "Sector 2"
                    else:
                        sec = "Sector 3"

                    lap_diff = driver_base_status.get(driver_code, 0)
                    target_lap_num = leader_lap_num - lap_diff
                    target_lap_row = driver_laps[driver_laps["LapNumber"] == target_lap_num]

                    driver_checkpoint_time = target_lap_row[time_col].values[0] if not target_lap_row.empty else pd.nan
                    if pd.notna(driver_checkpoint_time) and pd.notna(crossing_time):
                        time_delta = (driver_checkpoint_time - crossing_time) / np.timedelta64(1, "s")
                        recent_laps_sec = driver_laps["LapTime"].dt.total_seconds().dropna()
                        avg_lap_time = recent_laps_sec.mean() if not recent_laps_sec.empty else 85.0

                        if abs(time_delta) > abs(avg_lap_time):
                            lap_diff += 1
                            target_lap_num = leader_lap_num - lap_diff
                            target_lap_row = driver_laps[driver_laps["LapNumber"] == target_lap_num]
                            if not target_lap_row.empty and pd.notna(target_lap_row[time_col].values[0]):
                                driver_checkpoint_time = target_lap_row[time_col].values[0]
                                time_delta = (driver_checkpoint_time - crossing_time) / np.timedelta64(1, "s")
                            else:
                                time_delta = np.nan
                    else:
                        time_delta = np.nan

                    drivers_at_moment.append(
                        {
                            "driver_code": driver_code,
                            "lap_number": lap_num,
                            "sector_at_moment": sec,
                            "time_gap": time_delta,
                            "lap_diff": lap_diff,
                        }
                    )

            results[checkpoint_name] = {
                "leader_code": leader_code,
                "crossing_time": crossing_time,
                "sector_node": sector_node_obj,
                "drivers_status": drivers_at_moment,
            }

        return results

    def all_drivers_at_lap(self, target_lap: int) -> list[DriverNode]:
        """Extract comprehensive driver details, telemetry averages, nearby drivers, and sector snapshots for a target lap in a single pass."""
        laps_df = self.session.laps
        target_laps_df = laps_df[laps_df["LapNumber"] == target_lap]
        if target_laps_df.empty:
            return []

        sector_snapshots = self._compute_snapshot_at_lap(target_lap, laps_df)
        past_laps_all = laps_df[laps_df["LapNumber"] <= target_lap]
        drivers: list[DriverNode] = []

        for driver_code in self._driver_map.values():
            past_laps = past_laps_all[past_laps_all["Driver"] == driver_code]
            if past_laps.empty:
                continue

            target_lap_row = past_laps[past_laps["LapNumber"] == target_lap]
            if target_lap_row.empty:
                target_lap_row = past_laps.iloc[[-1]]

            pos = int(target_lap_row["Position"].values[0]) if pd.notna(target_lap_row["Position"].values[0]) else 99
            comp = str(target_lap_row["Compound"].values[0])
            age = int(target_lap_row["TyreLife"].values[0]) if pd.notna(target_lap_row["TyreLife"].values[0]) else 0
            pit_count = int(past_laps["PitInTime"].notna().sum())

            recent_laps_second = past_laps.tail(3)["LapTime"].dt.total_seconds()
            avg_time = round(float(recent_laps_second.mean()), 2) if len(recent_laps_second) > 0 else np.nan

            nearby_drivers = self._compute_nearby_drivers(driver_code, target_laps_df)

            driver_sectors = {}
            current_sector = "Sector 1"
            for sec_name, sec_data in sector_snapshots.items():
                for d_status in sec_data.get("drivers_status", []):
                    if d_status["driver_code"] == driver_code:
                        driver_sectors[sec_name] = d_status
                        if sec_name == "Sector 3":
                            current_sector = d_status["sector_at_moment"]

            current_sector_node = None
            if current_sector in sector_snapshots:
                current_sector_node = sector_snapshots[current_sector].get("sector_node")

            driver_obj = DriverNode(
                driver_code=driver_code,
                position=pos,
                compound=comp,
                tyre_age=age,
                pit_stops=pit_count,
                last_3_laps_average_seconds=avg_time,
                nearby_drivers=nearby_drivers,
                sector_snapshots=driver_sectors,
                current_sector=current_sector,
                current_sector_node=current_sector_node,
            )

            drivers.append(driver_obj)

        return drivers

    def build_graph_at_time(self, session_time_s: float) -> nx.DiGraph:
        """Race graph at an arbitrary session time (seconds since the start of the data stream)."""
        return self.graph_builder.build(session_time_s)

    def build_session_graph(self, target_lap: int, sector_name: str | None = None) -> nx.DiGraph:
        """Race graph at the moment the leader of ``target_lap`` crosses the end of ``sector_name``
        ("Sector 1" / "Sector 2" / "Sector 3"); without a sector: when the leader starts that lap.

        The graph has one node per ~10 m of track plus one node per driver, see RaceGraphBuilderClass."""
        t = self.graph_builder.snapshot_time(target_lap, sector_name)
        G = self.graph_builder.build(t)
        G.graph["target_lap"] = target_lap
        G.graph["sector_name"] = sector_name
        return G
