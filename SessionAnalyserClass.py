"""
SessionAnalyserClass.py · Formula 1 Strategic Decisions
=========================================================
The core data processor. It introduces the FastF1 API, handles caching,
 calculates nearby drivers, evaluates track statuses.
"""

import fastf1
import os
import numpy as np
import pandas as pd
import networkx as nx

from DriverNodeClass import DriverNode
from SectorNodeClass import SectorNode
from typing import List, Dict, Any, Tuple


class SessionAnalyser:
    """Manages F1 session loading, telemetry extraction, and graph representation generation."""
    def __init__(self, year: int, location: str, session_type: str = 'R') -> None:
        """Initializes and loads the FastF1 session with local caching enabled."""
        cache_dir = os.path.expanduser('~/fastf1_cache')
        os.makedirs(cache_dir, exist_ok=True)

        fastf1.Cache.enable_cache(cache_dir)

        self.session = fastf1.get_session(year, location, session_type)
        self.session.load()

        self._driver_map: Dict[str, str] = self._build_driver_map()

    def _build_driver_map(self) -> Dict[str, str]:
        """Maps internal FastF1 driver indentifiers to their standart three letter Abbreviations."""
        mapping: Dict[str, str] = {}
        for driver in self.session.drivers:
            mapping[driver] = self.session.get_driver(driver)['Abbreviation']
        return mapping

    def _get_leader_row(self, target_laps_df: pd.DataFrame) -> pd.Series:
        """Identifies and returns the lap dataframe row corresponding to the race leader."""
        leader_row = target_laps_df[target_laps_df['Position'] == 1]
        if leader_row.empty:
            return target_laps_df.iloc[0]
        return leader_row.iloc[0]


    def _compute_nearby_drivers(self, target_driver: str, target_laps_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculates drivers within a 5-second window of the target driver during a specific lap."""
        target_row = target_laps_df[target_laps_df['Driver'] == target_driver]
        if target_row.empty or pd.isna(target_row['Time'].values[0]):
            return []

        target_time = target_row['Time'].values[0] / np.timedelta64(1, 's')
        nearby_drivers: List[Dict[str, Any]] = []

        for _, row in target_laps_df.iterrows():
            other_driver = row['Driver']
            if other_driver == target_driver or pd.isna(row['Time']):
                continue

            other_time = row['Time'] / np.timedelta64(1, 's')
            time_delta = other_time - target_time

            if abs(time_delta) <= 5.0:
                nearby_drivers.append({
                    'driver_code': other_driver,
                    'time_gap': round(time_delta, 3),
                    'status': "behind" if time_delta > 0 else "ahead"
                })

        return sorted(nearby_drivers, key=lambda x: abs(x['time_gap']))

    def _compute_snapshot_at_lap(self, target_lap: int, laps_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Computes sector checkpoint positions, lap counts and leader gaps for all drivers when the leader crosses sectors."""
        target_laps_df = laps_df[laps_df['LapNumber'] == target_lap]
        if target_laps_df.empty:
            return {}

        leader_row = self._get_leader_row(target_laps_df)
        leader_code: str = leader_row['Driver']
        leader_lap_num: int = int(leader_row['LapNumber'])

        checkpoints: Dict[str, Tuple[Any, str]] = {
            "Sector 1": (leader_row['Sector1SessionTime'], 'Sector1SessionTime'),
            "Sector 2": (leader_row['Sector2SessionTime'], 'Sector2SessionTime'),
            "Sector 3": (leader_row['Sector3SessionTime'], 'Sector3SessionTime'),
        }

        driver_laps_grouped = {code: group for code, group in laps_df.groupby('Driver')}

        driver_base_status: Dict[str, int] = {}
        for driver_code, driver_laps in driver_laps_grouped.items():
            d_target_lap = driver_laps[driver_laps['LapNumber'] == leader_lap_num]
            if d_target_lap.empty:
                d_target_lap = driver_laps[driver_laps['LapNumber'] < leader_lap_num].tail(1)
                if d_target_lap.empty:
                    d_target_lap = driver_laps.head(1)
            d_lap_num = int(d_target_lap['LapNumber'].values[0]) if not d_target_lap.empty else leader_lap_num
            driver_base_status[driver_code] = leader_lap_num - d_lap_num

        results: Dict[str, Dict[str, Any]] = {}
        for checkpoint_name, (crossing_time, time_col) in checkpoints.items():
            if pd.isna(crossing_time):
                continue

            status_code, status_msg = self._get_track_status_at_time(crossing_time)
            sector_node_obj = SectorNode(checkpoint_name, status_code, status_msg)

            drivers_at_moment: List[Dict[str, Any]] = []
            for driver_code, driver_laps in driver_laps_grouped.items():
                active_lap = driver_laps[
                    (driver_laps['LapStartTime'] <= crossing_time) &
                    (driver_laps['Time'] > crossing_time)
                ]

                if active_lap.empty:
                    active_lap = driver_laps[driver_laps['LapStartTime'] <= crossing_time].tail(1)

                if not active_lap.empty:
                    lap_row = active_lap.iloc[0]
                    lap_num = int(lap_row['LapNumber'])

                    s1_t = lap_row['Sector1SessionTime']
                    s2_t = lap_row['Sector2SessionTime']

                    if pd.notna(s1_t) and crossing_time <= s1_t:
                        sec = "Sector 1"
                    elif pd.notna(s2_t) and crossing_time <= s2_t:
                        sec = "Sector 2"
                    else:
                        sec = "Sector 3"

                    lap_diff = driver_base_status.get(driver_code, 0)
                    target_lap_num = leader_lap_num - lap_diff
                    target_lap_row = driver_laps[driver_laps['LapNumber'] == target_lap_num]

                    driver_checkpoint_time = target_lap_row[time_col].values[0] if not target_lap_row.empty else pd.nan
                    if pd.notna(driver_checkpoint_time) and pd.notna(crossing_time):
                        time_delta = (driver_checkpoint_time - crossing_time) / np.timedelta64(1, 's')
                        recent_laps_sec = driver_laps['LapTime'].dt.total_seconds().dropna()
                        avg_lap_time = recent_laps_sec.mean() if not recent_laps_sec.empty else 85.0

                        if abs(time_delta) > abs(avg_lap_time):
                            lap_diff += 1
                            target_lap_num = leader_lap_num - lap_diff
                            target_lap_row = driver_laps[driver_laps['LapNumber'] == target_lap_num]
                            if not target_lap_row.empty and pd.notna(target_lap_row[time_col].values[0]):
                                driver_checkpoint_time = target_lap_row[time_col].values[0]
                                time_delta = (driver_checkpoint_time - crossing_time) / np.timedelta64(1, 's')
                            else:
                                time_delta = np.nan
                    else:
                        time_delta = np.nan

                    drivers_at_moment.append({
                        'driver_code': driver_code,
                        'lap_number': lap_num,
                        'sector_at_moment': sec,
                        'time_gap': time_delta,
                        'lap_diff': lap_diff
                    })

            results[checkpoint_name] = {
                'leader_code': leader_code,
                'crossing_time': crossing_time,
                'sector_node': sector_node_obj,
                'drivers_status': drivers_at_moment
            }

        return results

    def _get_track_status_at_time(self, crossing_time: int) -> tuple:
        """Finds the active track status code and message at any given session time"""
        track_statuses = self.session.track_status
        if track_statuses is None or track_statuses.empty:
            return '1', 'Clear'

        active_statuses = track_statuses[track_statuses['Time'] <= crossing_time]
        if active_statuses.empty:
            return '1', 'Clear'

        latest_status = active_statuses.iloc[-1]
        return latest_status['Status'], latest_status['Message']

    def all_drivers_at_lap(self, target_lap: int) -> List[DriverNode]:
        """Extract comprehensive driver details, telemetry averages, nearby drivers, and sector snapshots for a target lap in a single pass."""
        laps_df = self.session.laps
        target_laps_df = laps_df[laps_df['LapNumber'] == target_lap]
        if target_laps_df.empty:
            return []

        sector_snapshots = self._compute_snapshot_at_lap(target_lap, laps_df)
        past_laps_all = laps_df[laps_df['LapNumber'] <= target_lap]
        drivers: List[DriverNode] = []

        for driver_code in self._driver_map.values():
            past_laps = past_laps_all[past_laps_all['Driver'] == driver_code]
            if past_laps.empty:
                continue

            target_lap_row = past_laps[past_laps['LapNumber'] == target_lap]
            if target_lap_row.empty:
                target_lap_row = past_laps.iloc[[-1]]

            pos = int(target_lap_row['Position'].values[0]) if pd.notna(target_lap_row['Position'].values[0]) else 99
            comp = str(target_lap_row['Compound'].values[0])
            age = int(target_lap_row['TyreLife'].values[0]) if pd.notna(target_lap_row['TyreLife'].values[0]) else 0
            pit_count = int(past_laps['PitInTime'].notna().sum())

            recent_laps_second = past_laps.tail(3)['LapTime'].dt.total_seconds()
            avg_time = round(float(recent_laps_second.mean()), 2) if len(recent_laps_second) > 0 else np.nan

            nearby_drivers = self._compute_nearby_drivers(driver_code, target_laps_df)

            driver_sectors = {}
            current_sector = "Sector 1"
            for sec_name, sec_data in sector_snapshots.items():
                for d_status in sec_data.get('drivers_status', []):
                    if d_status['driver_code'] == driver_code:
                        driver_sectors[sec_name] = d_status
                        if sec_name == "Sector 3":
                            current_sector = d_status['sector_at_moment']

            driver_obj = DriverNode(
                driver_code=driver_code,
                position=pos,
                compound=comp,
                tyre_age=age,
                pit_stops=pit_count,
                last_3_laps_average_seconds=avg_time,
                nearby_drivers=nearby_drivers,
                sector_snapshots=driver_sectors,
                current_sector=current_sector
            )

            drivers.append(driver_obj)

        return drivers

    def build_session_graph(self, target_lap: int, sector_name: str) -> nx.Graph:
        """Constructs and returns a NetworkX graph representing sector memberships and driver proximity at a specific sector."""
        G = nx.Graph()

        sector_snapshots = self._compute_snapshot_at_lap(target_lap, self.session.laps)
        target_snapshot = sector_snapshots.get(sector_name, {})
        active_sector_node = target_snapshot.get('sector_node')

        if active_sector_node:
            G.add_node(
                sector_name,
                node_type='sector',
                sector_object=active_sector_node,
                is_sc=active_sector_node.is_safety_car,
                is_vsc=active_sector_node.is_vsc
            )
        else:
            G.add_node(sector_name, node_type='sector')

        drivers = self.all_drivers_at_lap(target_lap)
        processed_proximity_edges = set()

        for driver in drivers:
            sec_info = driver.sector_snapshots.get(sector_name, {})
            driver_sector = sec_info.get('sector_at_moment', driver.current_sector)

            G.add_node(
                driver.driver_code,
                node_type='driver',
                position=driver.position,
                compound=driver.compound,
                tyre_age=driver.tyre_age,
                pit_stops=driver.pit_stops,
                sector=driver_sector,
                sector_snapshots=driver.sector_snapshots
            )

            G.add_edge(driver.driver_code, driver_sector, edge_type='sector_membership')

            if hasattr(driver, 'nearby_drivers') and driver.nearby_drivers:
                for nb in driver.nearby_drivers:
                    neighbor_code = nb['driver_code']
                    edge_tuple = tuple(sorted([driver.driver_code, neighbor_code]))

                    if edge_tuple not in processed_proximity_edges:
                        processed_proximity_edges.add(edge_tuple)
                        G.add_edge(
                            edge_tuple[0],
                            edge_tuple[1],
                            edge_type='proximity',
                            time_gap=abs(nb['time_gap']),
                        )

        return G