"""
GraphExporterClass.py · Formula 1 Strategic Decisions
=========================================================
Converts a race graph (RaceGraphBuilder.build) into plain numpy arrays with a *fixed, named feature
schema* - the interface between the data pipeline and a model.

Why not feed the networkx graph directly? A model needs the same columns in the same order every time,
no strings, no NaN. Here: categorical -> one-hot (compound, sector), angles -> sin/cos, missing values -> 0
plus a boolean ``*_mask`` (True = value was present).

Token view for a BERT style encoder
-----------------------------------
tokens = [1 global token] + [D driver tokens] + [N track tokens]. Each token type has its own feature
width, so the model needs one ``nn.Linear`` per type projecting into the hidden size, plus a token-type
embedding (``token_layout``), then ``inputs_embeds=`` instead of ``input_ids=``.
BERT's limit is 512 positions. 10 m nodes give ~550 (Monza) to ~700 (Spa) track tokens, so
``pool_track(factor)`` merges neighbouring nodes (``suggest_pool_factor`` picks the factor).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import networkx as nx
import numpy as np

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]  # anything else -> compound_unknown

TRACK_FEATURES = [
    "s_m",
    "x_m",
    "y_m",
    "z_m",
    "heading_sin",
    "heading_cos",
    "curvature",
    "gradient",
    "sector_1",
    "sector_2",
    "sector_3",
    "is_corner",
    "corner_number",
    "marshal_sector",
    "in_drs_zone",
    "is_drs_start",
    "is_drs_end",
    "is_drs_detection",
    "flag_yellow",
    "flag_double_yellow",
    "flag_sc",
    "flag_vsc",
    "flag_vsc_ending",
    "flag_red",
    "n_drivers",
]
DRIVER_FEATURES = (
    [
        "x_m",
        "y_m",
        "z_m",
        "speed_kmh",
        "vx_mps",
        "vy_mps",
        "heading_sin",
        "heading_cos",
        "heading_rel_track_sin",
        "heading_rel_track_cos",
        "throttle",
        "brake",
        "gear",
        "rpm",
        "drs_open",
        "drs_eligible",
        "track_s_m",
        "lap_fraction",
        "lateral_offset_m",
        "progress_m",
        "lap_number",
        "position",
        "official_position",
        "gap_to_leader_s",
        "gap_ahead_s",
        "gap_behind_s",
        "tyre_age",
        "fresh_tyre",
        "stint",
        "pit_stops",
        "in_pit",
        "on_track",
        "pace_last3_s",
        "last_lap_time_s",
        "finished_or_retired",
    ]
    + [f"compound_{c.lower()}" for c in COMPOUNDS]
    + ["compound_unknown"]
)
GLOBAL_FEATURES = [
    "time_s",
    "leader_lap",
    "laps_remaining",
    "race_fraction",
    "is_green",
    "is_yellow",
    "is_sc",
    "is_vsc",
    "is_vsc_ending",
    "is_red",
    "sc_in_this_lap",
    "yellow_unlocalised",
    "n_yellow_sectors",
    "drs_enabled",
    "air_temp",
    "track_temp",
    "humidity",
    "pressure",
    "rainfall",
    "wind_speed",
    "wind_direction_sin",
    "wind_direction_cos",
    "n_drivers",
]
EDGE_FEATURES = [
    "gap_s",
    "progress_gap_m",
    "circular_gap_m",
    "lap_diff",
    "rel_speed_kmh",
    "order_neighbor",
    "drs_window",
]

# Rough physical scales for normalise=True (x/y/z and distances are handled per track). Real training
# should replace these with statistics computed on the training set.
_SCALES = {
    "curvature": 0.02,
    "gradient": 0.1,
    "speed_kmh": 350.0,
    "vx_mps": 100.0,
    "vy_mps": 100.0,
    "throttle": 100.0,
    "gear": 8.0,
    "rpm": 13000.0,
    "lateral_offset_m": 20.0,
    "lap_number": 80.0,
    "position": 20.0,
    "official_position": 20.0,
    "gap_to_leader_s": 100.0,
    "gap_ahead_s": 10.0,
    "gap_behind_s": 10.0,
    "tyre_age": 50.0,
    "stint": 4.0,
    "pit_stops": 4.0,
    "pace_last3_s": 120.0,
    "last_lap_time_s": 120.0,
    "corner_number": 20.0,
    "marshal_sector": 40.0,
    "n_drivers": 20.0,
    "time_s": 7200.0,
    "leader_lap": 80.0,
    "laps_remaining": 80.0,
    "n_yellow_sectors": 5.0,
    "air_temp": 40.0,
    "track_temp": 60.0,
    "humidity": 100.0,
    "pressure": 1100.0,
    "wind_speed": 10.0,
    "gap_s": 5.0,
    "progress_gap_m": 500.0,
    "circular_gap_m": 100.0,
    "rel_speed_kmh": 100.0,
}
_POOL_MAX = ("is_", "in_", "flag_", "corner_number")  # prefixes pooled with max (events must survive pooling)
_POOL_SUM = ("n_drivers",)
_POOL_CENTER = ("marshal_sector",)


@dataclass
class GraphArrays:
    """Fixed-schema arrays of one race graph snapshot."""

    time_s: float
    track_length_m: float
    track_spacing_m: float
    track_x: np.ndarray
    track_mask: np.ndarray
    track_names: list[str]
    driver_x: np.ndarray
    driver_mask: np.ndarray
    driver_names: list[str]
    driver_codes: list[str]
    global_x: np.ndarray
    global_mask: np.ndarray
    global_names: list[str]
    track_edge_index: np.ndarray  # [2, E] ring, both directions
    driver_to_track: np.ndarray  # [D] track token of every driver (-1 = unknown)
    driver_edge_index: np.ndarray  # [2, E'] indices into driver_x
    driver_edge_attr: np.ndarray  # [E', F]
    driver_edge_mask: np.ndarray
    edge_names: list[str]
    pool_factor: int = 1

    def validate(self) -> GraphArrays:
        """Raise ``ValueError`` if shapes or indices are inconsistent; returns ``self`` for chaining."""

        def check(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(f"inconsistent GraphArrays: {msg}")

        n, d = len(self.track_x), len(self.driver_codes)
        check(
            self.track_x.shape == self.track_mask.shape,
            f"track_mask {self.track_mask.shape} vs track_x {self.track_x.shape}",
        )
        check(self.track_x.shape[1] == len(self.track_names), "track feature names")
        check(self.driver_x.shape == (d, len(self.driver_names)), "driver_x shape")
        check(self.driver_x.shape == self.driver_mask.shape, "driver_mask shape")
        check(self.global_x.shape == (1, len(self.global_names)) == self.global_mask.shape, "global shape")
        check(self.driver_edge_attr.shape == (self.driver_edge_index.shape[1], len(self.edge_names)), "edge_attr shape")
        check(self.driver_edge_attr.shape == self.driver_edge_mask.shape, "edge mask shape")
        check(
            self.driver_edge_index.size == 0 or 0 <= self.driver_edge_index.min() and self.driver_edge_index.max() < d,
            "driver edge index out of range",
        )
        check(self.track_edge_index.size == 0 or self.track_edge_index.max() < n, "track edge index out of range")
        check(self.driver_to_track.shape == (d,) and self.driver_to_track.max(initial=-1) < n, "driver_to_track")
        return self

    # ------------------------------------------------------------------ token view
    @property
    def n_tokens(self) -> int:
        return 1 + len(self.driver_codes) + len(self.track_x)

    def token_layout(self) -> dict[str, np.ndarray]:
        """token_type_ids (0 global, 1 driver, 2 track) and a per-type position id: drivers = race position order,
        track = index along the lap (the lap is a ring: position n-1 is adjacent to position 0)."""
        d, n = len(self.driver_codes), len(self.track_x)
        return {
            "token_type_ids": np.array([0] + [1] * d + [2] * n, dtype=np.int64),
            "position_ids": np.concatenate([[0], np.arange(d), np.arange(n)]).astype(np.int64),
        }

    def pool_track(self, factor: int) -> GraphArrays:
        """Merge every ``factor`` neighbouring track nodes into one token (flags/markers: max, continuous: mean)."""
        if factor <= 1:
            return self
        n = len(self.track_x)
        m = int(np.ceil(n / factor))
        out = np.zeros((m, self.track_x.shape[1]), dtype=self.track_x.dtype)
        out_mask = np.zeros((m, self.track_mask.shape[1]), dtype=bool)
        for g in range(m):
            sl = slice(g * factor, min((g + 1) * factor, n))
            block, bmask = self.track_x[sl], self.track_mask[sl]
            for j, name in enumerate(self.track_names):
                if name.startswith(_POOL_MAX):
                    out[g, j] = block[:, j].max()
                elif name in _POOL_SUM:
                    out[g, j] = block[:, j].sum()
                elif name in _POOL_CENTER:
                    out[g, j] = block[len(block) // 2, j]
                else:
                    out[g, j] = block[:, j].mean()
            out_mask[g] = bmask.any(axis=0)
        idx = np.arange(m)
        ring = np.concatenate([np.stack([idx, (idx + 1) % m]), np.stack([(idx + 1) % m, idx])], axis=1)
        d2t = np.where(self.driver_to_track >= 0, self.driver_to_track // factor, -1)
        return replace(
            self,
            track_x=out,
            track_mask=out_mask,
            track_edge_index=ring,
            driver_to_track=d2t,
            track_spacing_m=self.track_spacing_m * factor,
            pool_factor=self.pool_factor * factor,
        ).validate()

    def to_dict(self) -> dict[str, Any]:
        """Flat dict of arrays, e.g. for ``np.savez_compressed(path, **arrays.to_dict())``."""
        skip = {"track_names", "driver_names", "global_names", "edge_names", "driver_codes"}
        return {k: v for k, v in self.__dict__.items() if k not in skip and isinstance(v, (np.ndarray, int, float))}

    def to_torch(self) -> dict[str, Any]:
        """Same as to_dict but as torch tensors (torch is an optional dependency)."""
        import torch  # noqa: WPS433

        return {k: torch.as_tensor(v) for k, v in self.to_dict().items() if isinstance(v, np.ndarray)}


def suggest_pool_factor(n_track: int, n_drivers: int, max_tokens: int = 512) -> int:
    """Smallest pooling factor so that global + drivers + pooled track tokens fit into ``max_tokens``."""
    budget = max_tokens - 1 - n_drivers
    if budget <= 0:
        raise ValueError("Driver tokens alone exceed the token budget")
    return max(1, int(np.ceil(n_track / budget)))


class GraphExporter:
    """networkx race graph -> GraphArrays."""

    def __init__(self, normalise: bool = False) -> None:
        self.normalise = normalise

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _f(v: Any) -> float:
        if v is None:
            return float("nan")
        return float(v)

    def _finish(
        self,
        X: np.ndarray,
        names: list[str],
        L: float,
        bbox: tuple[float, float, float, float] | None,
        race_distance_m: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(X)
        X = np.where(mask, X, 0.0).astype(np.float32)
        if self.normalise:
            for j, name in enumerate(names):
                if name in ("x_m", "y_m") and bbox:
                    cx, cy, half = bbox[0], bbox[1], bbox[2]
                    X[:, j] = (X[:, j] - (cx if name == "x_m" else cy)) / half
                elif name == "z_m":
                    X[:, j] = X[:, j] / 100.0
                elif name in ("s_m", "track_s_m"):
                    X[:, j] = X[:, j] / L
                elif name == "progress_m":  # ~0..1 over the race (a plain division by L would give a lap count)
                    X[:, j] = X[:, j] / (race_distance_m or 60 * L)
                elif name in _SCALES:
                    X[:, j] = X[:, j] / _SCALES[name]
            X = np.where(mask, X, 0.0).astype(np.float32)
        return X, mask

    # ------------------------------------------------------------------ main entry
    def export(self, G: nx.DiGraph) -> GraphArrays:
        track = G.graph["track"]
        L = track.length_m
        t_nodes = sorted((d for _, d in G.nodes(data=True) if d.get("node_type") == "track"), key=lambda d: d["index"])
        codes: list[str] = list(G.graph["driver_codes"])
        d_nodes = [G.nodes[c] for c in codes]
        xy = np.array([[d["x_m"], d["y_m"]] for d in t_nodes])
        bbox = (
            (xy[:, 0].max() + xy[:, 0].min()) / 2,
            (xy[:, 1].max() + xy[:, 1].min()) / 2,
            max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) / 2,
        )

        # ---- track
        T = np.array(
            [
                [
                    d["s_m"],
                    d["x_m"],
                    d["y_m"],
                    d["z_m"],
                    np.sin(d["heading_rad"]),
                    np.cos(d["heading_rad"]),
                    d["curvature"],
                    d["gradient"],
                    d["sector_idx"] == 0,
                    d["sector_idx"] == 1,
                    d["sector_idx"] == 2,
                    d["corner_number"] > 0,
                    d["corner_number"],
                    d["marshal_sector"],
                    d["drs_zone"] > 0,
                    d["is_drs_start"],
                    d["is_drs_end"],
                    d["is_drs_detection"],
                    d["flag_yellow"],
                    d["flag_double_yellow"],
                    d["flag_sc"],
                    d["flag_vsc"],
                    d["flag_vsc_ending"],
                    d["flag_red"],
                    d["n_drivers"],
                ]
                for d in t_nodes
            ],
            dtype=float,
        ).reshape(len(t_nodes), len(TRACK_FEATURES))
        T, T_mask = self._finish(T, TRACK_FEATURES, L, bbox)

        # ---- drivers
        rows = []
        for d in d_nodes:
            h, hr = d["heading_rad"], d["heading_rel_track_rad"]
            comp = str(d["compound"]).upper()
            onehot = [comp == c for c in COMPOUNDS]
            rows.append(
                [
                    d["x_m"],
                    d["y_m"],
                    d["z_m"],
                    d["speed_kmh"],
                    d["vx_mps"],
                    d["vy_mps"],
                    np.sin(h),
                    np.cos(h),
                    np.sin(hr),
                    np.cos(hr),
                    d["throttle"],
                    d["brake"],
                    d["gear"],
                    d["rpm"],
                    d["drs_open"],
                    d["drs_eligible"],
                    d["track_s_m"],
                    d["lap_fraction"],
                    d["lateral_offset_m"],
                    d["progress_m"],
                    d["lap_number"],
                    d["position"],
                    d["official_position"],
                    d["gap_to_leader_s"],
                    d["gap_ahead_s"],
                    d["gap_behind_s"],
                    d["tyre_age"],
                    d["fresh_tyre"],
                    d["stint"],
                    d["pit_stops"],
                    d["in_pit"],
                    d["on_track"],
                    d["last_3_laps_average_seconds"],
                    d["last_lap_time_s"],
                    d["finished_or_retired"],
                    *onehot,
                    not any(onehot),
                ]
            )
        D = np.array(rows, dtype=float).reshape(len(d_nodes), len(DRIVER_FEATURES))
        total_laps = G.graph["global"].get("total_laps")
        D, D_mask = self._finish(D, DRIVER_FEATURES, L, bbox, race_distance_m=total_laps * L if total_laps else None)

        # ---- global
        g = G.graph["global"]
        gl = np.array([[self._f(g.get(name)) for name in GLOBAL_FEATURES]], dtype=float)
        gl, gl_mask = self._finish(gl, GLOBAL_FEATURES, L, bbox)

        # ---- edges
        n = len(t_nodes)
        idx = np.arange(n)
        ring = np.concatenate([np.stack([idx, (idx + 1) % n]), np.stack([(idx + 1) % n, idx])], axis=1)
        d2t = np.array([G.nodes[c]["track_node"] for c in codes], dtype=np.int64)
        pos = {c: i for i, c in enumerate(codes)}
        src, dst, attrs = [], [], []
        for u, v, e in G.edges(data=True):
            if e.get("edge_type") == "proximity":
                src.append(pos[u])
                dst.append(pos[v])
                attrs.append([self._f(e[k]) for k in EDGE_FEATURES])
        E = np.array(attrs, dtype=float).reshape(len(attrs), len(EDGE_FEATURES))
        E, E_mask = self._finish(E, EDGE_FEATURES, L, bbox)

        return GraphArrays(
            time_s=float(G.graph["time_s"]),
            track_length_m=L,
            track_spacing_m=track.spacing_m,
            track_x=T,
            track_mask=T_mask,
            track_names=list(TRACK_FEATURES),
            driver_x=D,
            driver_mask=D_mask,
            driver_names=list(DRIVER_FEATURES),
            driver_codes=codes,
            global_x=gl,
            global_mask=gl_mask,
            global_names=list(GLOBAL_FEATURES),
            track_edge_index=ring,
            driver_to_track=d2t,
            driver_edge_index=np.array([src, dst], dtype=np.int64).reshape(2, len(src)),
            driver_edge_attr=E,
            driver_edge_mask=E_mask,
            edge_names=list(EDGE_FEATURES),
        ).validate()
