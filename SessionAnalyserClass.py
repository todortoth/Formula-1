import fastf1
import os
import numpy as np
import pandas as pd

from DriverNodeClass import DriverNode
from typing import List

class SessionAnalyser:
    def __init__(self, year: int, location: str, session_type: str = 'R'):
        cache_dir = os.path.expanduser('~/fastf1_cache')
        os.makedirs(cache_dir, exist_ok=True)

        fastf1.Cache.enable_cache(cache_dir)

        self.session = fastf1.get_session(year, location, session_type)
        self.session.load()

    def get_estimated_fuel(self, current_lap: int) -> float:
        """Helper to compute linear fuel degradation per lap"""
        if current_lap <= 0:
            return 110.0
        if current_lap >= self.session.laps['LapNumber'].max():
            return 1.0

        fuel_per_lap = (110.0 - 1.0) / self.session.laps['LapNumber'].max()
        return round (110.0 - (fuel_per_lap * current_lap), 2)

    def extract_all_drivers_at_lap(self, target_lap: int) -> List[DriverNode]:
        """Extract all drivers and driver information at target_lap from session.drivers"""
        drivers = []
        fuel_load = self.get_estimated_fuel(target_lap)

        target_laps_df = self.session.laps[self.session.laps['LapNumber'] == target_lap]

        for driver in self.session.drivers:
            driver_code = self.session.get_driver(driver)['Abbreviation']
            print(f'\nDriver: {driver}, Code: {driver_code}')

            driver_laps = self.session.laps[self.session.laps['Driver'] == driver_code]
            past_laps = driver_laps[driver_laps['LapNumber'] <= target_lap]

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

            nearby_drivers = self.extract_nearby_drivers(target_lap, driver_code, target_laps_df)

            driver_obj = DriverNode(
                driver_code=driver_code,
                position=pos,
                compound=comp,
                tyre_age=age,
                pit_stops=pit_count,
                last_3_laps_average_seconds=avg_time,
                fuel_remaining=fuel_load,
                nearby_drivers=nearby_drivers
            )

            drivers.append(driver_obj)

        return drivers

    def extract_nearby_drivers(self, target_lap: int, target_driver: str, target_laps_df) -> List[DriverNode]:
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
