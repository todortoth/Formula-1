from SessionAnalyserClass import SessionAnalyser

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np

def format_time_mmssms(seconds):
    """Másodpercet vagy Timedelta-t alakít PERC:MÁSODPERC:MILISEC formátumra"""
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
    analyzer = SessionAnalyser(year=2025, location='Hungarian Grand Prix', session_type='R')
    target_lap = 70
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

    snapshots = analyzer.sector_info(target_lap=target_lap)

    for sector_name, data in snapshots.items():
        print(
            f"--- Leader ({data['leader_code']}) finished: {sector_name} ---")
        print(f"{'Driver':<8} | {'Sector':<16} | {'Lap':<5} | {'Leader'}")
        print("-" * 60)

        for info in data['drivers_status']:
            if info['driver_code'] == data['leader_code']:
                gap_str = "Leader"
            elif info['lap_diff'] > 0:
                gap_str = f"+{info['lap_diff']} lap" if info['lap_diff'] == 1 else f"+{info['lap_diff']} laps"
            elif pd.isna(info['time_gap']):
                gap_str = "N/A"
            else:
                prefix = "+" if info['time_gap'] > 0 else ""
                gap_str = f"{prefix}{info['time_gap']:.3f}s"
            print(f"{info['driver_code']:<8} | {info['sector_at_moment']:<16} | Lap {info['lap_number']:<5} | {gap_str}")
        print("\n")
'''
    race_graph = analyzer.build_session_graph(target_lap=target_lap)

    print(f"Gráf sikeresen létrejött!")
    print(f"Csomópontok száma: {race_graph.number_of_nodes()}")
    print(f"Élek száma: {race_graph.number_of_edges()}")

    plt.figure(figsize=(12, 8))
    node_colors = ['lightblue' if race_graph.nodes[n].get('node_type') == 'driver' else 'orange' for n in
                   race_graph.nodes]

    pos = nx.spring_layout(race_graph, seed=42)
    nx.draw(race_graph, pos, with_labels=True, node_color=node_colors, node_size=700, font_size=10, font_weight='bold')
    plt.title("F1 Verseny Gráf - 20. kör (2025 Magyar Nagydíj)")
    plt.savefig("graph/graph_hu_25.png")
    plt.show()
'''

