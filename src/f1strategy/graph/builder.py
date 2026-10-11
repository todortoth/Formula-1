"""
RaceGraphBuilderClass.py · Formula 1 Strategic Decisions
=========================================================
Builds the race graph for one moment (session time) as a ``networkx.DiGraph``.

Node types
----------
``track``  one node per ~10 m of circuit, ids "T0000".."T{N-1}" (ring, node 0 = start/finish line)
           static : s_m, x_m, y_m, z_m, heading, curvature, gradient, sector_idx (0-2), corner_number,
                    marshal_sector, drs_zone, is_drs_start / is_drs_end / is_drs_detection
           dynamic: flag_yellow, flag_double_yellow, flag_sc, flag_vsc, flag_vsc_ending, flag_red,
                    flag_label, n_drivers (cars currently mapped to this node)
``driver`` one node per car, id = 3 letter code. Attributes = DriverNode.to_attributes()
           (x, y, z, speed, velocity, heading, inputs, DRS, track position, gaps, tyres, pit state, pace, ...)

Edge types (every relation is stored in both directions so message passing works on a DiGraph)
----------
``track_next`` / ``track_prev``   T_k -> T_k+1 and back, ``length_m``
``on_node`` / ``occupied_by``     driver -> its nearest track node and back; ``offset_m`` (along track from the node
                                  centre) and ``lateral_m`` (from the racing line)
``proximity``                     driver <-> driver when the true time gap <= ``proximity_window_s`` OR they are
                                  physically within ``proximity_window_m`` (also catches lapped traffic).
                                  Attributes are seen from the *source*: gap_s > 0 means the target is ahead.

Graph level (``G.graph``): ``time_s``, ``global`` (track status, SC/VSC, DRS enabled, weather, race progress ...),
``track`` (TrackModel), ``driver_codes``, ``driver_nodes`` (DriverNode objects), ``race_control``, ``event``.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from f1strategy.domain.driver import DriverNode
from f1strategy.racecontrol.timeline import RaceControlState, RaceControlTimeline
from f1strategy.telemetry.store import DriverSample, TelemetryStore
from f1strategy.telemetry.timing import DriverTiming
from f1strategy.track.model import TrackModel
from f1strategy.utils import td_seconds

log = logging.getLogger(__name__)
_NAN = float("nan")
DRS_WINDOW_S = 1.0  # a follower within 1 s of the car ahead may use DRS (approximated with the true time gap)


def track_node_id(k: int) -> str:
    return f"T{k:04d}"


class RaceGraphBuilder:
    """Combines TrackModel, TelemetryStore, DriverTiming and RaceControlTimeline into graph snapshots."""

    def __init__(
        self,
        session: Any,
        track: TrackModel,
        telemetry: TelemetryStore,
        timing: DriverTiming,
        timeline: RaceControlTimeline,
        driver_map: dict[str, str],
        proximity_window_s: float = 5.0,
        proximity_window_m: float = 50.0,
    ) -> None:
        self.session = session
        self.track = track
        self.telemetry = telemetry
        self.timing = timing
        self.timeline = timeline
        self.driver_map = driver_map
        self.proximity_window_s = proximity_window_s
        self.proximity_window_m = proximity_window_m
        self._weather = self._prepare_weather(session)

    # ------------------------------------------------------------------ snapshot time
    def snapshot_time(self, target_lap: int, sector_name: str | None = None) -> float:
        """Session time at which the leader of ``target_lap`` crosses ``sector_name`` ("Sector 1".."Sector 3");
        without a sector the moment the leader starts that lap."""
        laps = self.session.laps
        rows = laps[laps["LapNumber"] == target_lap]
        if rows.empty:
            raise ValueError(f"No data for lap {target_lap}")
        leader = rows[rows["Position"] == 1]
        row = leader.iloc[0] if not leader.empty else rows.iloc[0]
        column = {
            "Sector 1": "Sector1SessionTime",
            "Sector 2": "Sector2SessionTime",
            "Sector 3": "Sector3SessionTime",
        }.get(sector_name or "", "LapStartTime")
        value = row[column]
        if pd.isna(value) and column == "Sector3SessionTime":
            value = row["Time"]
        if pd.isna(value):
            raise ValueError(f"Lap {target_lap}: no timestamp for '{sector_name or 'lap start'}'")
        return float(pd.Timedelta(value).total_seconds())

    # ------------------------------------------------------------------ driver level
    def driver_nodes(self, t: float) -> list[DriverNode]:
        """All drivers at session time ``t``, ordered by live race position (leader first)."""
        samples: dict[str, DriverSample] = {
            c: self.telemetry.sample(c, t) for c in self.telemetry.codes if self.timing.has(c)
        }
        order = sorted(samples, key=lambda c: -samples[c].progress_m)
        rank = {c: i + 1 for i, c in enumerate(order)}
        L = self.track.length_m
        leader = order[0] if order else None

        nodes: list[DriverNode] = []
        for i, code in enumerate(order):
            smp, tim = samples[code], self.timing.at(code, t)
            ahead = order[i - 1] if i > 0 else None
            behind = order[i + 1] if i + 1 < len(order) else None
            node = DriverNode(
                driver_code=code,
                position=rank[code],
                compound=tim.compound,
                tyre_age=tim.tyre_age,
                pit_stops=tim.pit_stops,
                last_3_laps_average_seconds=round(tim.pace_last3_s, 3) if np.isfinite(tim.pace_last3_s) else _NAN,
                current_sector=f"Sector {int(self.track.sector_idx[smp.track_node]) + 1}"
                if smp.track_node >= 0 and self.track.sector_idx[smp.track_node] >= 0
                else None,
                x_m=smp.x_m,
                y_m=smp.y_m,
                z_m=smp.z_m,
                speed_kmh=smp.speed_kmh,
                vx_mps=smp.vx_mps,
                vy_mps=smp.vy_mps,
                heading_rad=smp.heading_rad,
                heading_valid=smp.heading_valid,
                heading_rel_track_rad=smp.heading_rel_track_rad,
                throttle=smp.throttle,
                brake=smp.brake,
                gear=smp.gear,
                rpm=smp.rpm,
                drs_open=smp.drs_raw in (10, 12, 14),
                drs_eligible=smp.drs_raw == 8,
                track_node=smp.track_node,
                track_s_m=smp.track_s_m,
                lateral_offset_m=smp.lateral_offset_m,
                progress_m=smp.progress_m,
                lap_number=tim.lap_number,
                lap_fraction=smp.track_s_m / L,
                on_track=smp.on_track,
                gap_to_leader_s=0.0 if code == leader else self.telemetry.gap_seconds(code, leader, t),
                gap_ahead_s=self.telemetry.gap_seconds(code, ahead, t) if ahead else _NAN,
                gap_behind_s=self.telemetry.gap_seconds(behind, code, t) if behind else _NAN,
                official_position=tim.official_position,
                last_lap_time_s=tim.last_lap_time_s,
                fresh_tyre=tim.fresh_tyre,
                stint=tim.stint,
                in_pit=tim.in_pit,
                laps_completed=tim.laps_completed,
                finished_or_retired=tim.finished_or_retired,
            )
            nodes.append(node)
        return nodes

    # ------------------------------------------------------------------ global level
    @staticmethod
    def _prepare_weather(session: Any) -> pd.DataFrame | None:
        try:
            w = session.weather_data
        except Exception:
            return None
        if w is None or len(w) == 0:
            return None
        w = w.copy()
        w["_t"] = td_seconds(w["Time"])
        return w.sort_values("_t").reset_index(drop=True)

    def _weather_at(self, t: float) -> dict[str, float]:
        keys = ("AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall", "WindSpeed", "WindDirection")
        if self._weather is None:
            return dict.fromkeys(keys, _NAN)
        i = max(int(np.searchsorted(self._weather["_t"].to_numpy(), t, side="right")) - 1, 0)
        row = self._weather.iloc[i]
        return {k: float(row[k]) if k in row and pd.notna(row[k]) else _NAN for k in keys}

    def global_features(self, t: float, drivers: list[DriverNode], rc: RaceControlState) -> dict[str, Any]:
        total_laps = getattr(self.session, "total_laps", None)
        total_laps = int(total_laps) if total_laps is not None and pd.notna(total_laps) else None
        leader_lap = drivers[0].lap_number if drivers else 0
        w = self._weather_at(t)
        wind_dir = np.deg2rad(w["WindDirection"]) if np.isfinite(w["WindDirection"]) else _NAN
        return {
            "time_s": t,
            "leader_lap": leader_lap,
            "total_laps": total_laps,
            "laps_remaining": (total_laps - leader_lap) if total_laps else _NAN,
            "race_fraction": (leader_lap / total_laps) if total_laps else _NAN,
            "track_status": rc.track_status,
            "track_message": rc.track_message,
            "is_green": rc.is_clear,
            "is_yellow": rc.is_yellow,
            "is_sc": rc.is_safety_car,
            "is_vsc": rc.is_vsc,
            "is_vsc_ending": rc.is_vsc_ending,
            "is_red": rc.is_red,
            "sc_in_this_lap": rc.sc_in_this_lap,
            "yellow_unlocalised": rc.yellow_unlocalised,
            "n_yellow_sectors": len(rc.sector_flags),
            "drs_enabled": rc.drs_enabled,
            "air_temp": w["AirTemp"],
            "track_temp": w["TrackTemp"],
            "humidity": w["Humidity"],
            "pressure": w["Pressure"],
            "rainfall": w["Rainfall"],
            "wind_speed": w["WindSpeed"],
            "wind_direction_sin": float(np.sin(wind_dir)) if np.isfinite(wind_dir) else _NAN,
            "wind_direction_cos": float(np.cos(wind_dir)) if np.isfinite(wind_dir) else _NAN,
            "n_drivers": len(drivers),
        }

    # ------------------------------------------------------------------ graph
    def build(self, t: float) -> nx.DiGraph:
        """Graph of the whole race at session time ``t`` (seconds)."""
        tr = self.track
        rc = self.timeline.state_at(t)
        drivers = self.driver_nodes(t)
        G = nx.DiGraph(
            time_s=t, track=tr, driver_codes=[d.driver_code for d in drivers], event=self._event_name(), race_control=rc
        )
        G.graph["global"] = self.global_features(t, drivers, rc)

        # ---- track nodes with static layout + dynamic flags
        yellow = np.isin(tr.marshal_sector, [s for s, f in rc.sector_flags.items() if f == "YELLOW"])
        double = np.isin(tr.marshal_sector, [s for s, f in rc.sector_flags.items() if f == "DOUBLE YELLOW"])
        occupancy = np.bincount([d.track_node for d in drivers if d.track_node >= 0], minlength=tr.n)
        for k in range(tr.n):
            red, sc, vsc, vsc_end = rc.is_red, rc.is_safety_car, rc.is_vsc, rc.is_vsc_ending
            if red:
                label = "RED"
            elif sc:
                label = "SC"
            elif vsc_end:
                label = "VSC ENDING"
            elif vsc:
                label = "VSC"
            elif double[k]:
                label = "DOUBLE YELLOW"
            elif yellow[k]:
                label = "YELLOW"
            else:
                label = "CLEAR"
            G.add_node(
                track_node_id(k),
                node_type="track",
                index=k,
                s_m=float(tr.s[k]),
                x_m=float(tr.x[k]),
                y_m=float(tr.y[k]),
                z_m=float(tr.z[k]),
                heading_rad=float(tr.heading[k]),
                curvature=float(tr.curvature[k]),
                gradient=float(tr.gradient[k]),
                sector_idx=int(tr.sector_idx[k]),
                corner_number=int(tr.corner_number[k]),
                marshal_sector=int(tr.marshal_sector[k]),
                drs_zone=int(tr.drs_zone[k]),
                is_drs_start=bool(tr.is_drs_start[k]),
                is_drs_end=bool(tr.is_drs_end[k]),
                is_drs_detection=bool(tr.is_drs_detection[k]),
                flag_yellow=bool(yellow[k]),
                flag_double_yellow=bool(double[k]),
                flag_sc=sc,
                flag_vsc=vsc,
                flag_vsc_ending=vsc_end,
                flag_red=red,
                flag_label=label,
                n_drivers=int(occupancy[k]),
            )
        for k in range(tr.n):
            a, b = track_node_id(k), track_node_id((k + 1) % tr.n)
            G.add_edge(a, b, edge_type="track_next", length_m=tr.spacing_m)
            G.add_edge(b, a, edge_type="track_prev", length_m=tr.spacing_m)

        # ---- driver nodes + attachment to the track
        for d in drivers:
            G.add_node(d.driver_code, node_type="driver", **d.to_attributes())
            if d.track_node >= 0:
                tid = track_node_id(d.track_node)
                offset = float(tr.circular_delta(tr.s[d.track_node], d.track_s_m))
                G.add_edge(d.driver_code, tid, edge_type="on_node", offset_m=offset, lateral_m=d.lateral_offset_m)
                G.add_edge(tid, d.driver_code, edge_type="occupied_by", offset_m=offset, lateral_m=d.lateral_offset_m)

        # ---- driver <-> driver proximity
        L = tr.length_m
        for i, a in enumerate(drivers):
            for b in drivers[i + 1 :]:
                circ = float(tr.circular_delta(a.track_s_m, b.track_s_m))  # + : b ahead of a on the track
                d_prog = b.progress_m - a.progress_m  # + : b ahead of a in the race
                if d_prog >= 0:
                    gap = self.telemetry.gap_seconds(a.driver_code, b.driver_code, t)
                else:
                    gap = -self.telemetry.gap_seconds(b.driver_code, a.driver_code, t)
                near_time = np.isfinite(gap) and abs(gap) <= self.proximity_window_s
                near_space = abs(circ) <= self.proximity_window_m
                if not (near_time or near_space):
                    continue
                lap_diff = int(round((d_prog - circ) / L))
                neighbours = abs(a.position - b.position) == 1
                for src, dst, sign in ((a, b, 1.0), (b, a, -1.0)):
                    follower_in_window = (
                        bool(0 < sign * gap <= DRS_WINDOW_S and lap_diff == 0) if np.isfinite(gap) else False
                    )
                    G.add_edge(
                        src.driver_code,
                        dst.driver_code,
                        edge_type="proximity",
                        gap_s=float(sign * gap) if np.isfinite(gap) else _NAN,
                        progress_gap_m=float(sign * d_prog),
                        circular_gap_m=float(sign * circ),
                        lap_diff=int(sign * lap_diff),
                        rel_speed_kmh=float(dst.speed_kmh - src.speed_kmh),
                        order_neighbor=bool(neighbours),
                        drs_window=follower_in_window,
                    )

        # old-style convenience list: time_gap > 0 -> other car is behind
        for d in drivers:
            near = []
            for _, nb, e in G.out_edges(d.driver_code, data=True):
                if (
                    e["edge_type"] == "proximity"
                    and np.isfinite(e["gap_s"])
                    and abs(e["gap_s"]) <= self.proximity_window_s
                ):
                    near.append(
                        {
                            "driver_code": nb,
                            "time_gap": round(-e["gap_s"], 3),
                            "status": "behind" if e["gap_s"] < 0 else "ahead",
                        }
                    )
            d.nearby_drivers = sorted(near, key=lambda x: abs(x["time_gap"]))
        G.graph["driver_nodes"] = drivers
        return G

    def _event_name(self) -> str | None:
        try:
            return str(self.session.event["EventName"])
        except Exception:
            return None
