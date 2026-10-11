"""RaceGraphBuilder: structure, flags, proximity edges, causality."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from f1strategy import DriverTiming, RaceGraphBuilder, SessionAnalyser
from f1strategy.graph.builder import track_node_id
from tests.helpers import clone_session, true_gap
from tests.synthetic.session import make_synthetic_session


class TestGraph:
    def test_structure(self, G_sc, analyser):
        n = analyser.track.n
        types = [d["node_type"] for _, d in G_sc.nodes(data=True)]
        assert types.count("track") == n and types.count("driver") == 6
        kinds = [e["edge_type"] for *_, e in G_sc.edges(data=True)]
        assert kinds.count("track_next") == n and kinds.count("track_prev") == n
        assert kinds.count("on_node") == 6 and kinds.count("occupied_by") == 6
        assert G_sc.has_edge(track_node_id(n - 1), track_node_id(0))  # ring
        assert G_sc.graph["target_lap"] == 3 and G_sc.graph["sector_name"] == "Sector 2"

    def test_snapshot_time_is_leader_sector_crossing(self, G_sc, analyser, syn):
        ver = syn.session.laps.query("Driver == 'VER' and LapNumber == 3").iloc[0]
        assert G_sc.graph["time_s"] == pytest.approx(ver.Sector2SessionTime.total_seconds())
        lap_start = analyser.graph_builder.snapshot_time(4)
        assert lap_start == pytest.approx(ver.Time.total_seconds())
        with pytest.raises(ValueError):
            analyser.graph_builder.snapshot_time(99)

    def test_driver_order_and_positions(self, G_sc):
        order = [d.driver_code for d in G_sc.graph["driver_nodes"]]
        assert order == ["VER", "NOR", "LEC", "HAM", "ALO", "SAI"]
        assert [G_sc.nodes[c]["position"] for c in order] == [1, 2, 3, 4, 5, 6]
        assert G_sc.nodes["VER"]["gap_to_leader_s"] == 0.0 and np.isnan(G_sc.nodes["VER"]["gap_ahead_s"])
        assert np.isnan(G_sc.nodes["SAI"]["gap_behind_s"])

    def test_driver_attached_to_correct_track_node(self, G_sc, analyser):
        tr = analyser.track
        for code in G_sc.graph["driver_codes"]:
            ((_, tid, e),) = [x for x in G_sc.out_edges(code, data=True) if x[2]["edge_type"] == "on_node"]
            k = G_sc.nodes[tid]["index"]
            assert abs(tr.circular_delta(tr.s[k], G_sc.nodes[code]["track_s_m"])) <= tr.spacing_m / 2 + 1e-6
            assert abs(e["offset_m"]) <= tr.spacing_m / 2 + 1e-6

    def test_safety_car_flags_every_node(self, G_sc, analyser):
        track_nodes = [d for _, d in G_sc.nodes(data=True) if d["node_type"] == "track"]
        assert all(d["flag_sc"] and d["flag_label"] == "SC" and not d["flag_vsc"] for d in track_nodes)
        g = G_sc.graph["global"]
        assert g["is_sc"] and not g["is_green"] and g["drs_enabled"] is False and g["track_status"] == "4"
        assert g["leader_lap"] == 3 and g["laps_remaining"] == 3 and g["race_fraction"] == 0.5

    def test_sector_yellow_flags_only_hit_their_marshal_sector(self, analyser):
        G = analyser.build_graph_at_time(121.0)  # S3 double yellow + S8 yellow
        for _, d in G.nodes(data=True):
            if d["node_type"] != "track":
                continue
            assert d["flag_double_yellow"] == (d["marshal_sector"] == 3)
            assert d["flag_yellow"] == (d["marshal_sector"] == 8)
            assert d["flag_label"] == (
                "DOUBLE YELLOW" if d["marshal_sector"] == 3 else "YELLOW" if d["marshal_sector"] == 8 else "CLEAR"
            )
        assert G.graph["global"]["n_yellow_sectors"] == 2 and G.graph["global"]["is_yellow"]

    def test_vsc_and_clear(self, analyser):
        vsc = analyser.build_graph_at_time(340.0)
        assert all(d["flag_vsc"] for _, d in vsc.nodes(data=True) if d["node_type"] == "track")
        end = analyser.build_graph_at_time(356.0)
        assert all(
            d["flag_vsc_ending"] and d["flag_label"] == "VSC ENDING"
            for _, d in end.nodes(data=True)
            if d["node_type"] == "track"
        )
        green = analyser.build_graph_at_time(365.0)
        assert green.graph["global"]["is_green"]
        assert all(d["flag_label"] == "CLEAR" for _, d in green.nodes(data=True) if d["node_type"] == "track")

    def test_proximity_edges_are_antisymmetric_and_correct(self, G_sc, syn):
        for u, v, e in G_sc.edges(data=True):
            if e["edge_type"] != "proximity":
                continue
            r = G_sc.edges[v, u]
            assert r["gap_s"] == pytest.approx(-e["gap_s"]) and r["lap_diff"] == -e["lap_diff"]
            assert (
                r["circular_gap_m"] == pytest.approx(-e["circular_gap_m"])
                and r["order_neighbor"] == e["order_neighbor"]
            )
        e = G_sc.edges["NOR", "VER"]
        assert e["gap_s"] > 0 and e["gap_s"] == pytest.approx(
            true_gap(syn.truth, "NOR", "VER", G_sc.graph["time_s"]), abs=0.03
        )
        assert e["drs_window"] and e["order_neighbor"] and e["lap_diff"] == 0
        assert not G_sc.edges["VER", "NOR"]["drs_window"]  # the leader is not "in DRS range" of the follower
        e2 = G_sc.edges["HAM", "LEC"]  # 0.56 s behind -> inside the window
        assert e2["drs_window"] == (0 < e2["gap_s"] <= 1.0) and e2["drs_window"]
        e3 = G_sc.edges["ALO", "LEC"]  # 1.4 s behind -> outside the window
        assert 1.0 < e3["gap_s"] and not e3["drs_window"]
        assert not G_sc.edges["VER", "LEC"]["order_neighbor"]

    def test_lapped_stationary_car_is_connected_physically(self, analyser, syn):
        """SAI stops at 3.4 laps; when VER passes it one lap later they are 'lap_diff' 1 apart but metres away."""
        t = float(
            np.interp(
                4.4 * syn.truth["L"],
                syn.truth["S"]("VER", np.arange(0, syn.truth["t_total"], 0.01)),
                np.arange(0, syn.truth["t_total"], 0.01),
            )
        )
        G = analyser.build_graph_at_time(t)
        e = G.edges["VER", "SAI"]
        assert abs(e["circular_gap_m"]) < 5 and e["lap_diff"] == -1
        assert not np.isfinite(e["gap_s"]) or abs(e["gap_s"]) > 40  # one lap apart in race terms
        assert G.nodes["SAI"]["finished_or_retired"]  # retired a lap ago
        assert G.nodes["SAI"]["speed_kmh"] == 0 and G.nodes["VER"]["speed_kmh"] > 100

    def test_pit_state_in_graph(self, analyser, syn):
        laps = syn.session.laps
        t_in = laps.query("Driver == 'NOR' and LapNumber == 3").iloc[0].PitInTime.total_seconds()
        G = analyser.build_graph_at_time(t_in + 1)
        assert G.nodes["NOR"]["in_pit"] and G.nodes["NOR"]["pit_stops"] == 1 and not G.nodes["VER"]["in_pit"]

    def test_drs_open_flag_in_graph(self, analyser, syn):
        tru = syn.truth
        t = float(
            np.interp(0.10 * tru["L"] + 2 * tru["L"], tru["S"]("VER", np.arange(0, 400, 0.01)), np.arange(0, 400, 0.01))
        )
        G = analyser.build_graph_at_time(t)
        assert G.graph["global"]["drs_enabled"] and G.nodes["VER"]["drs_open"]
        tr_node = G.nodes[track_node_id(G.nodes["VER"]["track_node"])]
        assert tr_node["drs_zone"] >= 1

    def test_graph_is_causal_for_laps_not_yet_run(self, analyser, syn):
        G = analyser.build_graph_at_time(70.0)  # during lap 1: nothing completed
        for c in G.graph["driver_codes"]:
            assert np.isnan(G.nodes[c]["last_lap_time_s"]) and G.nodes[c]["laps_completed"] == 0

    def test_nearby_driver_compat_list(self, G_sc):
        nodes = {d.driver_code: d for d in G_sc.graph["driver_nodes"]}
        assert (
            nodes["VER"].nearby_drivers[0]["driver_code"] == "NOR"
            and nodes["VER"].nearby_drivers[0]["status"] == "behind"
        )
        assert nodes["NOR"].nearby_drivers[0]["status"] in ("ahead", "behind")

    def test_old_lap_level_api_still_works(self, analyser):
        drivers = analyser.all_drivers_at_lap(2)
        assert len(drivers) == 6 and drivers[0].driver_code


def _builder(analyser, **kwargs):
    return RaceGraphBuilder(analyser.session, analyser.track, analyser.telemetry, analyser.timing,
                            analyser.timeline, analyser._driver_map, **kwargs)  # fmt: skip


class TestBuilderInvariants:
    TIMES = np.arange(35.0, 380.0, 11.0)

    def test_invariants_hold_throughout_the_race(self, analyser):
        n = analyser.track.n
        for t in self.TIMES:
            G = analyser.build_graph_at_time(float(t))
            assert G.number_of_nodes() == n + 6
            drivers = G.graph["driver_nodes"]
            assert [d.position for d in drivers] == list(range(1, 7))
            progress = [d.progress_m for d in drivers]
            assert progress == sorted(progress, reverse=True)  # position order == progress order
            assert G.graph["time_s"] == t
            for d in drivers:
                out = [e["edge_type"] for _, _, e in G.out_edges(d.driver_code, data=True)]
                assert out.count("on_node") == 1  # every car sits on exactly one track node
                assert d.gap_to_leader_s == 0 or d.gap_to_leader_s > 0 or np.isnan(d.gap_to_leader_s)
            for u, v, e in G.edges(data=True):
                if e["edge_type"] == "proximity":
                    assert G.has_edge(v, u)  # proximity is stored in both directions

    def test_gaps_to_leader_are_non_negative_and_ordered_during_a_normal_stretch(self, analyser):
        G = analyser.build_graph_at_time(100.0)
        gaps = [d.gap_to_leader_s for d in G.graph["driver_nodes"]]
        assert gaps[0] == 0.0 and all(g > 0 for g in gaps[1:]) and gaps == sorted(gaps)

    def test_the_track_ring_can_be_walked_in_both_directions(self, G_sc):
        n = sum(1 for _, d in G_sc.nodes(data=True) if d["node_type"] == "track")
        for kind, step in (("track_next", +1), ("track_prev", -1)):
            node, seen = "T0000", []
            for _ in range(n):
                seen.append(node)
                (node,) = [v for _, v, e in G_sc.out_edges(node, data=True) if e["edge_type"] == kind]
            assert node == "T0000" and len(set(seen)) == n
            assert G_sc.nodes[seen[1]]["index"] == step % n

    def test_edge_lengths_equal_the_node_spacing(self, G_sc, analyser):
        lengths = {e["length_m"] for *_, e in G_sc.edges(data=True) if e["edge_type"].startswith("track_")}
        assert lengths == {analyser.track.spacing_m}

    def test_graph_level_attributes(self, G_sc, analyser):
        assert G_sc.graph["track"] is analyser.track and G_sc.graph["event"] == "Synthetic Grand Prix"
        assert G_sc.graph["driver_codes"] == ["VER", "NOR", "LEC", "HAM", "ALO", "SAI"]
        assert G_sc.graph["race_control"].is_safety_car and isinstance(G_sc, nx.DiGraph)

    def test_global_feature_keys_match_the_exporter_schema(self, G_sc):
        from f1strategy.graph.exporter import GLOBAL_FEATURES

        assert set(GLOBAL_FEATURES) <= set(G_sc.graph["global"])

    def test_track_node_attribute_keys_are_complete(self, G_sc):
        keys = set(G_sc.nodes["T0000"])
        assert {"s_m", "x_m", "y_m", "z_m", "heading_rad", "curvature", "gradient", "sector_idx", "corner_number",
                "marshal_sector", "drs_zone", "is_drs_start", "is_drs_end", "is_drs_detection", "flag_yellow",
                "flag_double_yellow", "flag_sc", "flag_vsc", "flag_vsc_ending", "flag_red", "flag_label", "n_drivers"} <= keys  # fmt: skip

    def test_n_drivers_counts_cars_on_each_node(self, G_sc):
        counts = [d["n_drivers"] for _, d in G_sc.nodes(data=True) if d["node_type"] == "track"]
        assert sum(counts) == 6 and max(counts) >= 1

    def test_driver_attribute_values_are_physical(self, G_sc):
        for code in G_sc.graph["driver_codes"]:
            d = G_sc.nodes[code]
            assert 0 <= d["speed_kmh"] < 400 and 0 <= d["throttle"] <= 100 and 0 <= d["gear"] <= 8
            assert d["current_sector"] in ("Sector 1", "Sector 2", "Sector 3")
            assert -np.pi <= d["heading_rad"] <= np.pi and 0 <= d["lap_fraction"] < 1


class TestProximityWindows:
    def test_zero_windows_remove_every_proximity_edge(self, analyser):
        G = _builder(analyser, proximity_window_s=0.0, proximity_window_m=0.0).build(150.0)
        assert not [1 for *_, e in G.edges(data=True) if e["edge_type"] == "proximity"]

    def test_huge_windows_connect_everybody_with_everybody(self, analyser):
        G = _builder(analyser, proximity_window_s=1e6, proximity_window_m=1e6).build(150.0)
        assert len([1 for *_, e in G.edges(data=True) if e["edge_type"] == "proximity"]) == 6 * 5

    def test_narrow_time_window_only_keeps_close_cars(self, analyser):
        G = _builder(analyser, proximity_window_s=0.5, proximity_window_m=0.0).build(150.0)
        gaps = [abs(e["gap_s"]) for *_, e in G.edges(data=True) if e["edge_type"] == "proximity"]
        assert gaps and max(gaps) <= 0.5

    def test_drs_window_requires_a_gap_of_at_most_one_second_on_the_same_lap(self, analyser):
        G = _builder(analyser, proximity_window_s=1e6, proximity_window_m=1e6).build(150.0)
        for u, v, e in G.edges(data=True):
            if e["edge_type"] == "proximity":
                assert e["drs_window"] == bool(0 < e["gap_s"] <= 1.0 and e["lap_diff"] == 0)

    def test_order_neighbors_are_exactly_consecutive_positions(self, analyser):
        G = _builder(analyser, proximity_window_s=1e6, proximity_window_m=1e6).build(150.0)
        for u, v, e in G.edges(data=True):
            if e["edge_type"] == "proximity":
                assert e["order_neighbor"] == (abs(G.nodes[u]["position"] - G.nodes[v]["position"]) == 1)

    def test_gap_sign_convention(self, analyser):
        G = _builder(analyser, proximity_window_s=1e6, proximity_window_m=1e6).build(150.0)
        assert G.edges["NOR", "VER"]["gap_s"] > 0 > G.edges["VER", "NOR"]["gap_s"]  # + = target is ahead of source
        assert G.edges["NOR", "VER"]["progress_gap_m"] > 0 > G.edges["VER", "NOR"]["progress_gap_m"]


class TestBuilderWithIncompleteData:
    def test_no_weather_gives_nan_weather_features(self):
        a = SessionAnalyser.from_session(make_synthetic_session(with_weather=False).session)
        g = a.build_graph_at_time(150.0).graph["global"]
        assert all(np.isnan(g[k]) for k in ("air_temp", "track_temp", "humidity", "pressure", "rainfall", "wind_speed"))

    def test_weather_values_come_from_the_latest_row_before_t(self, analyser):
        g = analyser.build_graph_at_time(150.0).graph["global"]
        assert (g["air_temp"], g["track_temp"], g["humidity"], g["pressure"], g["wind_speed"]) == (24.0, 38.0, 50.0, 1010.0, 2.5)  # fmt: skip
        assert g["wind_direction_sin"] == pytest.approx(1.0) and abs(g["wind_direction_cos"]) < 1e-9

    def test_no_race_control_feeds_at_all(self):
        syn = make_synthetic_session(with_messages=False, with_track_status=False)
        G = SessionAnalyser.from_session(syn.session).build_graph_at_time(184.0)
        g = G.graph["global"]
        assert g["is_green"] and not g["is_sc"] and g["drs_enabled"] is None and g["n_yellow_sectors"] == 0
        assert all(d["flag_label"] == "CLEAR" for _, d in G.nodes(data=True) if d["node_type"] == "track")

    def test_no_track_status_only(self):
        syn = make_synthetic_session(with_track_status=False)
        g = SessionAnalyser.from_session(syn.session).build_graph_at_time(121.0).graph["global"]
        assert g["n_yellow_sectors"] == 2 and not g["is_sc"]  # sector yellows still come from the messages

    def test_missing_total_laps_and_event_name(self, syn):
        sess = clone_session(syn.session, total_laps=None, event={})
        G = SessionAnalyser.from_session(sess).build_graph_at_time(150.0)
        g = G.graph["global"]
        assert g["total_laps"] is None and np.isnan(g["laps_remaining"]) and np.isnan(g["race_fraction"])
        assert G.graph["event"] is None

    def test_driver_without_lap_data_is_left_out(self, analyser, syn):
        timing = DriverTiming(syn.session.laps[syn.session.laps["Driver"] != "ALO"])
        b = RaceGraphBuilder(analyser.session, analyser.track, analyser.telemetry, timing, analyser.timeline, analyser._driver_map)  # fmt: skip
        G = b.build(150.0)
        assert "ALO" not in G.graph["driver_codes"] and len(G.graph["driver_codes"]) == 5

    def test_snapshot_time_needs_the_requested_timestamp(self):
        a = SessionAnalyser.from_session(make_synthetic_session(with_sector_times=False).session)
        with pytest.raises(ValueError, match="no timestamp"):
            a.graph_builder.snapshot_time(3, "Sector 1")
        assert a.graph_builder.snapshot_time(3, "Sector 3") > 0  # sector 3 ends at the lap end, still available

    def test_sector3_falls_back_to_the_lap_end_time(self, analyser, syn):
        laps = syn.session.laps.copy()
        laps["Sector3SessionTime"] = pd.NaT
        b = _builder(SessionAnalyser.from_session(clone_session(syn.session, laps=laps)))
        ref = analyser.graph_builder.snapshot_time(3, "Sector 3")
        assert b.snapshot_time(3, "Sector 3") == pytest.approx(ref, abs=0.01)

    def test_unknown_sector_name_means_lap_start(self, analyser):
        assert analyser.graph_builder.snapshot_time(3, "Sector 9") == analyser.graph_builder.snapshot_time(3)

    def test_leader_falls_back_to_first_row_without_classified_positions(self, syn):
        laps = syn.session.laps.copy()
        laps["Position"] = np.nan
        a = SessionAnalyser.from_session(clone_session(syn.session, laps=laps))
        assert a.graph_builder.snapshot_time(3, "Sector 2") > 0  # picks *a* row instead of crashing

    def test_build_works_for_every_sector_of_every_lap(self, analyser):
        for lap in range(1, 7):
            for sector in (None, "Sector 1", "Sector 2", "Sector 3"):
                G = analyser.build_session_graph(lap, sector)
                assert G.graph["target_lap"] == lap and G.graph["sector_name"] == sector


class TestLegacyLapApi:
    def test_all_drivers_at_lap_basics(self, analyser):
        drivers = analyser.all_drivers_at_lap(3)
        assert [d.driver_code for d in drivers] and {d.driver_code for d in drivers} == {"VER", "NOR", "LEC", "HAM", "ALO", "SAI"}  # fmt: skip
        by = {d.driver_code: d for d in drivers}
        assert by["VER"].position == 1 and by["NOR"].pit_stops == 1 and by["VER"].pit_stops == 0
        assert by["HAM"].compound == "SOFT" and by["VER"].compound == "MEDIUM"
        assert by["VER"].last_3_laps_average_seconds > 0

    def test_nearby_drivers_are_symmetric(self, analyser):
        by = {d.driver_code: d for d in analyser.all_drivers_at_lap(3)}
        gap = {n["driver_code"]: n for n in by["VER"].nearby_drivers}["NOR"]
        back = {n["driver_code"]: n for n in by["NOR"].nearby_drivers}["VER"]
        assert gap["status"] == "behind" and back["status"] == "ahead"
        assert gap["time_gap"] == pytest.approx(-back["time_gap"])

    def test_track_status_lookup_uses_the_corrected_codes(self, analyser):
        assert analyser._get_track_status_at_time(pd.to_timedelta(190, unit="s"), "Sector 1").is_safety_car
        assert analyser._get_track_status_at_time(pd.to_timedelta(340, unit="s"), "Sector 1").is_vsc
        assert analyser._get_track_status_at_time(pd.to_timedelta(356, unit="s"), "Sector 1").is_vsc_ending
        assert analyser._get_track_status_at_time(pd.to_timedelta(10, unit="s"), "Sector 1").is_green_flag

    def test_sector_snapshots_exist_for_all_three_sectors(self, analyser):
        d = analyser.all_drivers_at_lap(3)[0]
        assert set(d.sector_snapshots) == {"Sector 1", "Sector 2", "Sector 3"}
