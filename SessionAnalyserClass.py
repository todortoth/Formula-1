import fastf1
import os
import numpy as np

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

        for driver in self.session.drivers:
            driver_code = self.session.get_driver(driver)['Abbreviation']

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

            driver_obj = DriverNode(
                driver_code=driver_code,
                position=pos,
                compound=comp,
                tyre_age=age,
                pit_stops=pit_count,
                last_3_laps_average_seconds=avg_time,
                fuel_remaining=fuel_load
            )

            drivers.append(driver_obj)

        return drivers