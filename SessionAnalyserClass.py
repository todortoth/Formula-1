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

            nearby_drivers = self.extract_nearby_drivers(target_lap, driver_code)

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

    def extract_nearby_drivers(self, target_lap: int, target_driver: str) -> List[DriverNode]:
        target_lap_telemetry = self.session.laps.pick_drivers(target_driver).pick_laps(target_lap).get_telemetry()
        if target_lap_telemetry.empty:
            return []

        idx = min(100, len(target_lap_telemetry) // 2)
        target_time = target_lap_telemetry['SessionTime'].iloc[idx].total_seconds()

        target_distance = target_lap_telemetry['Distance'].iloc[idx]

        nearby_drivers = []

        for driver in self.session.drivers:
            print(f'Driver:{driver}', end='\t')
            driver_code = self.session.get_driver(driver)['Abbreviation']
            if driver_code == target_driver:
                continue

            try:
                other_telemetry = self.session.laps.pick_drivers(driver_code).pick_laps(target_lap).get_telemetry()
                if other_telemetry.empty:
                    continue

                other_telemetry = other_telemetry.drop_duplicates(subset=['Distance'])

                other_time_at_distance = np.interp(
                    target_distance,
                    other_telemetry['Distance'].values,
                    other_telemetry['SessionTime'].dt.total_seconds().values
                )

                time_delta = other_time_at_distance - target_time

                if abs(time_delta) <= 5.0:
                    nearby_drivers.append({
                        'driver_code': driver_code,
                        'time_gap': round(time_delta, 3),
                        'status': "behind" if time_delta > 0 else "ahead"
                    })

            except Exception:
                continue

        nearby_drivers = sorted(nearby_drivers, key=lambda x: abs(x['time_gap']))
        return nearby_drivers
