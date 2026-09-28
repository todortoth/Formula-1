from SessionAnalyserClass import SessionAnalyser

import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

def format_time_mmssms(seconds):
    """Converting seconds or Timedelta to MIN:SEC:MILISEC format"""
    if pd.isna(seconds):
        return "N/A"

    if isinstance(seconds, (pd.Timedelta, np.timedelta64)):
        total_seconds = pd.Timedelta(seconds).total_seconds()
    else:
        total_seconds = float(seconds)

    minutes = int(total_seconds // 60)
    secs = int(total_seconds % 60)
    millisecs = int(round((total_seconds % 1) * 1000))

    return f"{minutes:01d}:{secs:02d}:{millisecs:03d}"

if __name__ == "__main__":
    location = 'Hungarian Grand Prix'
    analyzer = SessionAnalyser(year=2025, location=location, session_type='R')
    target_lap = 65

    print(f"\n--- Summarized Data at lap {target_lap}. ---")
    lap_snapshot = analyzer.all_drivers_at_lap(target_lap=target_lap)

    print(f"{'Driver':<8} | {'Pos':<4} | {'Tyre':<8} | {'Age':<4} | {'Pits':<5} | {'Avg Lap (s)':<12} | {'Nearby Drivers (within 5s)'}")
    print("-" * 110)

    for driver in lap_snapshot:
        nearby_str = "None"
        if hasattr(driver, 'nearby_drivers') and driver.nearby_drivers:
            nearby_list = []
            for nb in driver.nearby_drivers:
                prefix = "+" if nb['status'] == 'behind' else "-"
                nearby_list.append(f"{nb['driver_code']} ({prefix}{abs(nb['time_gap'])}s)")
            nearby_str = ", ".join(nearby_list)
        formatted_avg_lap = format_time_mmssms(driver.last_3_laps_average_seconds)

        print(f"{driver.driver_code:<8} | "
              f"{str(driver.position):<4} | "
              f"{str(driver.compound):<8} | "
              f"{str(driver.tyre_age):<4} | "
              f"{driver.pit_stops:<5} | "
              f"{formatted_avg_lap:<12} | "
              f"{nearby_str}")

    print("\n--- Sector Info ---")

    for sector_name in ["Sector 1", "Sector 2", "Sector 3"]:
        print(
            f"--- Leader finished: {sector_name} ---")
        print(f"{'Driver':<8} | {'Sector':<16} | {'Lap':<5} | {'Leader Gap'}")
        print("-" * 60)

        for driver in lap_snapshot:
            sec_info = driver.sector_snapshots.get(sector_name)
            if not sec_info:
                continue

            if driver.position == 1:
                gap_str = "Leader"
            elif sec_info['lap_diff'] > 0:
                gap_str = f"+{sec_info['lap_diff']} lap" if sec_info['lap_diff'] == 1 else f"+{sec_info['lap_diff']} laps"
            elif pd.isna(sec_info['time_gap']):
                gap_str = "N/A"
            else:
                prefix = "+" if sec_info['time_gap'] > 0 else ""
                gap_str = f"{prefix}{sec_info['time_gap']:.3f}s"
            print(f"{driver.driver_code:<8} | {sec_info['sector_at_moment']:<16} | Lap {sec_info['lap_number']:<5} | {gap_str}")
        print("\n")

    os.makedirs("graphs", exist_ok=True)
    sectors_to_plot = ["Sector 1", "Sector 2", "Sector 3"]

    for sector_name in sectors_to_plot:
        print(f"Building Graph ({sector_name}) at lap {target_lap}...")
        race_graph = analyzer.build_session_graph(target_lap=target_lap, sector_name=sector_name)

        print(
            f"Graph successfully created ({sector_name})! Nodes: {race_graph.number_of_nodes()}, Edges: {race_graph.number_of_edges()}")

        plt.figure(figsize=(12, 8))
        node_colors = ['lightblue' if race_graph.nodes[n].get('node_type') == 'driver' else 'orange' for n in
                       race_graph.nodes]

        pos = nx.spring_layout(race_graph, seed=42)
        nx.draw(
            race_graph,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=700,
            font_size=10,
            font_weight='bold'
        )
        file_suffix = sector_name.lower().replace(" ", "")
        filename = f"graphs/graph_hu_25_{file_suffix}.png"

        plt.title(f"F1 Race Graph - LAP {target_lap} ({sector_name} - {location})")
        plt.savefig(filename)
        print(f"Saved to: '{filename}'\n")
        plt.close()

