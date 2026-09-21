import fastf1
import os
import numpy as np
import pandas as pd
import networkx as nx

from DriverNodeClass import DriverNode
from typing import List

class SessionAnalyser:
    def __init__(self, year: int, location: str, session_type: str = 'R'):
        cache_dir = os.path.expanduser('~/fastf1_cache')
        os.makedirs(cache_dir, exist_ok=True)

        fastf1.Cache.enable_cache(cache_dir)

        self.session = fastf1.get_session(year, location, session_type)
        self.session.load()

    def extract_all_drivers_at_lap(self, target_lap: int) -> List[DriverNode]:
        """Extract all drivers and driver information at target_lap from session.drivers"""
        drivers = []

        laps_df = self.session.laps
        target_laps_df = laps_df[laps_df['LapNumber'] == target_lap]
        if target_laps_df.empty:
            return []

        leader_row = target_laps_df[target_laps_df['Position'] == 1]
        if leader_row.empty:
            leader_row = target_laps_df.iloc[[0]]

        reference_session_time = leader_row['LapStartTime'].values[0]
        past_laps_all = laps_df[laps_df['LapNumber'] <= target_lap]


        for driver in self.session.drivers:
            driver_code = self.session.get_driver(driver)['Abbreviation']

            driver_laps = self.session.laps[self.session.laps['Driver'] == driver_code]
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

            current_sector = "Sector 1"
            try:
                active_lap = driver_laps[
                    (driver_laps['LapStartTime'] <= reference_session_time) &
                    (driver_laps['Time'] >= reference_session_time)
                ]

                if active_lap.empty:
                    active_lap = driver_laps[
                        (driver_laps['LapStartTime'] <= reference_session_time) &
                        (driver_laps['LapStartTime'] + pd.Timedelta(seconds=120) >= reference_session_time)
                    ].tail(1)

                if active_lap.empty:
                    active_lap = driver_laps[driver_laps['LapStartTime'] <= reference_session_time].tail(1)

                if not active_lap.empty:
                    active_lap_num = active_lap['LapNumber'].values[0]
                    lap_row = active_lap.iloc[0]

                    telemetry = self.session.laps.pick_drivers(driver_code).pick_laps(active_lap_num).get_telemetry()

                    if not telemetry.empty and 'Distance' in telemetry.columns:
                        time_diffs = (telemetry['SessionTime'] - reference_session_time).abs()
                        closest_idx = time_diffs.argsort().iloc[0]
                        current_distance = telemetry['Distance'].iloc[closest_idx]

                        s1_time = lap_row['Sector1SessionTime']
                        s2_time = lap_row['Sector2SessionTime']

                        s1_dist, s2_dist = 0.0, 0.0
                        if pd.notna(s1_time):
                            s1_idx = (telemetry['SessionTime'] - s1_time).abs().argsort().iloc[0]
                            s1_dist = telemetry['Distance'].iloc[s1_idx]
                        if pd.notna(s2_time):
                            s2_idx = (telemetry['SessionTime'] - s2_time).abs().argsort().iloc[0]
                            s2_dist = telemetry['Distance'].iloc[s2_idx]

                        if s1_dist == 0: s1_dist = 3000.0
                        if s2_dist == 0: s2_dist = 7000.0

                        if s1_dist > 0 and current_distance <= s1_dist:
                            current_sector = "Sector 1"
                        elif s2_dist > 0 and current_distance <= s2_dist:
                            current_sector = "Sector 2"
                        else:
                            current_sector = "Sector 3"
            except Exception:
                pass

            nearby_drivers = self.extract_nearby_drivers(driver_code, target_laps_df)

            driver_obj = DriverNode(
                driver_code=driver_code,
                position=pos,
                compound=comp,
                tyre_age=age,
                pit_stops=pit_count,
                last_3_laps_average_seconds=avg_time,
                nearby_drivers=nearby_drivers,
                current_sector=current_sector
            )

            drivers.append(driver_obj)

        return drivers

    @staticmethod
    def extract_nearby_drivers(target_driver: str, target_laps_df) -> List[DriverNode]:
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

    def build_session_graph(self, target_lap: int):
        G = nx.Graph()

        for sector in ["Sector 1", "Sector 2", "Sector 3"]:
            G.add_node(sector, node_type='sector')

        drivers = self.extract_all_drivers_at_lap(target_lap)

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