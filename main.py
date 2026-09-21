from SessionAnalyserClass import SessionAnalyser

import matplotlib.pyplot as plt
import networkx as nx

if __name__ == "__main__":
    analyzer = SessionAnalyser(year=2025, location='Hungarian Grand Prix', session_type='R')
    target_lap = 1
    lap_snapshot = analyzer.extract_all_drivers_at_lap(target_lap=target_lap)

    print(f"{'Driver':<8} | {'Pos':<4} | {'Tyre':<8} | {'Age':<4} | {'Pits':<5} | {'Avg Lap (s)':<12} | {'Sector':<8} | {'Nearby Drivers (within 5s)'}")
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
              f"{driver.current_sector:<8} | "
              f"{nearby_str}")

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