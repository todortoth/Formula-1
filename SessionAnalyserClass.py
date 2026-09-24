import fastf1
import os
import numpy as np
import pandas as pd
import networkx as nx
from pandas.core.indexers import check_key_length

from DriverNodeClass import DriverNode
from typing import List, Dict, Any

class SessionAnalyser:
    def __init__(self, year: int, location: str, session_type: str = 'R'):
        cache_dir = os.path.expanduser('~/fastf1_cache')
        os.makedirs(cache_dir, exist_ok=True)

        fastf1.Cache.enable_cache(cache_dir)

        self.session = fastf1.get_session(year, location, session_type)
        self.session.load()

    def all_drivers_at_lap(self, target_lap: int) -> List[DriverNode]:
        """Extract all drivers and driver information at target_lap from session.drivers"""
        drivers = []

        laps_df = self.session.laps
        target_laps_df = laps_df[laps_df['LapNumber'] == target_lap]
        if target_laps_df.empty:
            return []

        leader_row = target_laps_df[target_laps_df['Position'] == 1]
        if leader_row.empty:
            leader_row = target_laps_df.iloc[[0]]

        past_laps_all = laps_df[laps_df['LapNumber'] <= target_lap]


        for driver in self.session.drivers:
            driver_code = self.session.get_driver(driver)['Abbreviation']
            past_laps = past_laps_all[past_laps_all['Driver'] == driver_code]

            if past_laps.empty:
                continue

            target_lap_row = past_laps[past_laps['LapNumber'] == target_lap]
            if target_lap_row.empty:
                target_lap_row = past_laps.iloc[[-1]]

            pos = target_lap_row['Position'].values[0]
            comp = target_lap_row['Compound'].values[0]
            age = target_lap_row['TyreLife'].values[0]
            pit_count = past_laps['PitInTime'].notna().sum()

            recent_laps_second = past_laps.tail(3)['LapTime'].dt.total_seconds()
            avg_time = round(recent_laps_second.mean(), 2) if len(recent_laps_second) > 0 else np.nan

            nearby_drivers = self.nearby_drivers(driver_code, target_laps_df)

            driver_obj = DriverNode(
                driver_code=driver_code,
                position=pos,
                compound=comp,
                tyre_age=age,
                pit_stops=pit_count,
                last_3_laps_average_seconds=avg_time,
                nearby_drivers=nearby_drivers
            )

            drivers.append(driver_obj)

        return drivers

    @staticmethod
    def nearby_drivers(target_driver: str, target_laps_df) -> List[DriverNode]:
        target_row = target_laps_df[target_laps_df['Driver'] == target_driver]
        if target_row.empty or pd.isna(target_row['Time'].values[0]):
            return []

        target_time = target_row['Time'].values[0] / np.timedelta64(1, 's')
        nearby_drivers = []

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

        nearby_drivers = sorted(nearby_drivers, key=lambda x: abs(x['time_gap']))
        return nearby_drivers

    def sector_info(self, target_lap: int) -> Dict[str, Any]:
        laps_df = self.session.laps
        target_laps_df = laps_df[laps_df['LapNumber'] == target_lap]

        if target_laps_df.empty:
            return {}

        leader_row = target_laps_df[target_laps_df['Position'] == 1]
        if leader_row.empty:
            leader_row = target_laps_df.iloc[[0]]

        leader_code = leader_row['Driver'].values[0]
        leader_lap_num = int(leader_row['LapNumber'].values[0])

        s1_crossing_time = leader_row['Sector1SessionTime'].values[0]
        s2_crossing_time = leader_row['Sector2SessionTime'].values[0]
        lap_end_crossing_time = leader_row['Sector3SessionTime'].values[0]

        driver_base_status = {}
        for driver in self.session.drivers:
            driver_code = self.session.get_driver(driver)['Abbreviation']
            driver_laps = laps_df[laps_df['Driver'] == driver_code]
            if driver_laps.empty:
                continue

            d_target_lap = driver_laps[driver_laps['LapNumber'] == leader_lap_num]
            if d_target_lap.empty:
                d_target_lap = driver_laps[driver_laps['LapNumber'] < leader_lap_num].tail(1)
                if d_target_lap.empty:
                    d_target_lap = driver_laps.head(1)
            d_lap_num = int(d_target_lap['LapNumber'].values[0])
            driver_base_status[driver_code] = {
                'lap_diff': leader_lap_num - d_lap_num
            }

        checkpoints = {
            "Sector 1": (s1_crossing_time, 'Sector1SessionTime'),
            "Sector 2": (s2_crossing_time, 'Sector2SessionTime'),
            "Sector 3": (lap_end_crossing_time, 'Sector3SessionTime'),
        }

        results = {}

        for checkpoint_name, (crossing_time, time_col) in checkpoints.items():
            if pd.isna(crossing_time):
                continue

            drivers_at_moment = []

            for driver in self.session.drivers:
                driver_code = self.session.get_driver(driver)['Abbreviation']
                driver_laps = laps_df[laps_df['Driver'] == driver_code]

                if driver_laps.empty:
                    continue

                active_lap = driver_laps[
                    (driver_laps['LapStartTime'] <= crossing_time) &
                    (driver_laps['Time'] >= crossing_time)
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

                    lap_diff = driver_base_status.get(driver_code, {}).get('lap_diff', 0)
                    driver_checkpoint_time = lap_row[time_col]

                    if lap_diff > 0:
                        time_delta = np.nan
                    elif pd.notna(driver_checkpoint_time) and pd.notna(crossing_time):
                        time_delta = (driver_checkpoint_time - crossing_time) / np.timedelta64(1, 's')

                        recent_laps_sec = driver_laps['LapTime'].dt.total_seconds().dropna()
                        avg_lap_time = recent_laps_sec.mean() if not recent_laps_sec.empty else 85.0

                        if time_delta < -avg_lap_time:
                            lap_diff += 1
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
                'drivers_status': drivers_at_moment
            }
        return results

    def build_session_graph(self, target_lap: int) -> nx.Graph:
        G = nx.Graph()

        for sector in ["Sector 1", "Sector 2", "Sector 3"]:
            G.add_node(sector, node_type='sector')

        drivers = self.all_drivers_at_lap(target_lap)

        processed_proximity_edges = set()

        for driver in drivers:
            G.add_node(
                driver.driver_code,
                node_type='driver',
                position=driver.position,
                compound=driver.compound,
                tyre_age=driver.tyre_age,
                pit_stops=driver.pit_stops,
                sector=driver.current_sector
            )

            G.add_edge(driver.driver_code, driver.current_sector, edge_type='sector_membership')

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