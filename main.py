from SessionAnalyserClass import SessionAnalyser

if __name__ == "__main__":
    analyzer = SessionAnalyser(year=2025, location='Hungarian Grand Prix', session_type='R')

    lap_snapshot = analyzer.extract_all_drivers_at_lap(target_lap=20)

    print(f"{'Driver':<8} | {'Pos':<4} | {'Tyre':<8} | {'Age':<4} | {'Pits':<5} | {'Avg Lap (s)':<12} | {'Fuel (kg)':<10} | {'Nearby Drivers (within 5s)'}")
    print("-" * 110)

    for driver in lap_snapshot:

        nearby_str = "None"

        if hasattr(driver, 'nearby_drivers') and driver.nearby_drivers:
            nearby_list = []
            for nb in driver.nearby_drivers:
                prefix = "+" if nb['status'] == 'behind' else "-"
                nearby_list.append(f"{nb['driver_code']} ({prefix}{abs(nb['time_gap'])}s)")
            nearby_str = ", ".join(nearby_list)


        print(f"{driver.driver_code:<8} | "
              f"{str(driver.position):<4} | "
              f"{str(driver.compound):<8} | "
              f"{str(driver.tyre_age):<4} | "
              f"{driver.pit_stops:<5} | "
              f"{str(driver.last_3_laps_average_seconds):<12} | "
              f"{driver.fuel_remaining} | "
              f"{nearby_str}")
