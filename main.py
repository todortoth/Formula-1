from SessionAnalyserClass import SessionAnalyser

if __name__ == "__main__":
    analyzer = SessionAnalyser(year=2025, location='Hungarian Grand Prix', session_type='R')

    lap_snapshot = analyzer.extract_all_drivers_at_lap(target_lap=20)

    print(f"{'Driver':<8} | {'Pos':<4} | {'Tyre':<8} | {'Age':<4} | {'Pits':<5} | {'Avg Lap (s)':<12} | {'Fuel (kg)'}")
    print("-" * 65)

    for driver in lap_snapshot:
        print(f"{driver.driver_code:<8} | "
              f"{str(driver.position):<4} | "
              f"{str(driver.compound):<8} | "
              f"{str(driver.tyre_age):<4} | "
              f"{driver.pit_stops:<5} | "
              f"{str(driver.last_3_laps_average_seconds):<12} | "
              f"{driver.fuel_remaining}")

