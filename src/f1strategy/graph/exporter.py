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

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import networkx as nx
import numpy as np

# The six per-node flags (bit k of the packed flag byte = FLAG_BITS[k]) and the track columns that change over time.
# NOTE: the order of FLAG_BITS is part of the on-disk format of sequences; changing it requires a new format version.
FLAG_BITS = ["flag_yellow", "flag_double_yellow", "flag_sc", "flag_vsc", "flag_vsc_ending", "flag_red"]
TRACK_DYNAMIC = [*FLAG_BITS, "n_drivers"]

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
        "lap_fraction_sin",
        "lap_fraction_cos",
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
        "pos_age_s",
        "car_age_s",
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
    "pos_age_s": 0.5,
    "car_age_s": 0.5,
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
    xy_bbox: tuple[float, float, float] = (0.0, 0.0, 1.0)  # centre x, centre y, half extent of the track [m]
    race_distance_m: float | None = None  # scheduled race distance, used to scale progress_m
    normalised: bool = False

    def normalised_copy(self) -> GraphArrays:
        """Copy with rough physical scaling applied (see ``GraphExporter``); masked entries stay 0."""
        if self.normalised:
            raise ValueError("arrays are already normalised")
        L = self.track_length_m

        def scale(X: np.ndarray, mask: np.ndarray, names: list[str]) -> np.ndarray:
            X = X.copy()
            for j, name in enumerate(names):
                if name in ("x_m", "y_m"):
                    X[:, j] = (X[:, j] - self.xy_bbox[0 if name == "x_m" else 1]) / self.xy_bbox[2]
                elif name == "z_m":
                    X[:, j] = X[:, j] / 100.0
                elif name in ("s_m", "track_s_m"):
                    X[:, j] = X[:, j] / L
                elif name == "progress_m":  # ~0..1 over the race (a plain division by L would give a lap count)
                    X[:, j] = X[:, j] / (self.race_distance_m or 60 * L)
                elif name in _SCALES:
                    X[:, j] = X[:, j] / _SCALES[name]
            return np.where(mask, X, 0.0).astype(np.float32)

        return replace(
            self,
            track_x=scale(self.track_x, self.track_mask, self.track_names),
            driver_x=scale(self.driver_x, self.driver_mask, self.driver_names),
            global_x=scale(self.global_x, self.global_mask, self.global_names),
            driver_edge_attr=scale(self.driver_edge_attr, self.driver_edge_mask, self.edge_names),
            normalised=True,
        )

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
        out = {k: v for k, v in self.__dict__.items() if k not in skip and isinstance(v, (np.ndarray, int, float))}
        out["xy_bbox"] = np.asarray(self.xy_bbox, dtype=float)
        out["race_distance_m"] = float("nan") if self.race_distance_m is None else float(self.race_distance_m)
        out["normalised"] = int(self.normalised)
        return out

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


@dataclass
class StaticTrack:
    """Everything about the track that does not change during a session, in exporter units."""

    names: list[str]  # static track columns (TRACK_FEATURES without TRACK_DYNAMIC)
    values: np.ndarray  # [N, Fs] float32
    mask: np.ndarray  # [N, Fs] bool
    xy_bbox: tuple[float, float, float]
    length_m: float
    spacing_m: float


@dataclass
class FrameRecord:
    """One snapshot without the track part: what the dense sampling path produces per frame (cheap, no graph)."""

    time_s: float
    driver_codes: list[str]  # leader first
    driver_x: np.ndarray  # [D, Fd] float32
    driver_mask: np.ndarray
    driver_to_track: np.ndarray  # [D] int
    global_x: np.ndarray  # [Fg] float32
    global_mask: np.ndarray
    edge_src: np.ndarray  # [E] indices into driver_codes
    edge_dst: np.ndarray
    edge_attr: np.ndarray  # [E, Fe] float32
    edge_mask: np.ndarray
    track_flags: np.ndarray  # [N] uint8, bit k = FLAG_BITS[k]
    race_distance_m: float | None = None

    @classmethod
    def from_arrays(cls, arr: GraphArrays) -> FrameRecord:
        """Reduce a full ``GraphArrays`` (raw, un-normalised) to a record."""
        flags = np.zeros(len(arr.track_x), dtype=np.uint8)
        for k, name in enumerate(FLAG_BITS):
            flags |= ((arr.track_x[:, arr.track_names.index(name)] > 0.5).astype(np.uint8)) << k
        return cls(
            time_s=arr.time_s, driver_codes=list(arr.driver_codes), driver_x=arr.driver_x, driver_mask=arr.driver_mask,
            driver_to_track=arr.driver_to_track, global_x=arr.global_x[0], global_mask=arr.global_mask[0],
            edge_src=arr.driver_edge_index[0], edge_dst=arr.driver_edge_index[1], edge_attr=arr.driver_edge_attr,
            edge_mask=arr.driver_edge_mask, track_flags=flags, race_distance_m=arr.race_distance_m,
        )  # fmt: skip


def pack_flags(yellow: np.ndarray, double: np.ndarray, rc: Any) -> np.ndarray:
    """Per-node flag bits (``FLAG_BITS`` order) from the yellow / double-yellow node masks and the race control state."""
    bits = {
        "flag_yellow": yellow,
        "flag_double_yellow": double,
        "flag_sc": rc.is_safety_car,
        "flag_vsc": rc.is_vsc,
        "flag_vsc_ending": rc.is_vsc_ending,
        "flag_red": rc.is_red,
    }
    out = np.zeros(len(yellow), dtype=np.uint8)
    for k, name in enumerate(FLAG_BITS):
        out |= np.where(bits[name], np.uint8(1) << k, np.uint8(0)).astype(np.uint8)
    return out


class GraphExporter:
    """networkx race graph -> GraphArrays (and, for dense sampling, snapshot -> FrameRecord)."""

    def __init__(self, normalise: bool = False) -> None:
        self.normalise = normalise

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _f(v: Any) -> float:
        if v is None:
            return float("nan")
        return float(v)

    @staticmethod
    def _finish(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Zero-fill missing values (NaN) and return ``(float32 values, boolean presence mask)``."""
        mask = np.isfinite(X)
        return np.where(mask, X, 0.0).astype(np.float32), mask

    @staticmethod
    def static_columns(a: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Static track feature columns (float64) from per-node arrays - shared by the graph and the dense path,
        so both compute exactly the same numbers."""
        sector = np.asarray(a["sector_idx"])
        corner = np.asarray(a["corner_number"])
        return {
            "s_m": np.asarray(a["s"], float),
            "x_m": np.asarray(a["x"], float),
            "y_m": np.asarray(a["y"], float),
            "z_m": np.asarray(a["z"], float),
            "heading_sin": np.sin(np.asarray(a["heading"], float)),
            "heading_cos": np.cos(np.asarray(a["heading"], float)),
            "curvature": np.asarray(a["curvature"], float),
            "gradient": np.asarray(a["gradient"], float),
            "sector_1": (sector == 0).astype(float),
            "sector_2": (sector == 1).astype(float),
            "sector_3": (sector == 2).astype(float),
            "is_corner": (corner > 0).astype(float),
            "corner_number": corner.astype(float),
            "marshal_sector": np.asarray(a["marshal_sector"]).astype(float),
            "in_drs_zone": (np.asarray(a["drs_zone"]) > 0).astype(float),
            "is_drs_start": np.asarray(a["is_drs_start"]).astype(float),
            "is_drs_end": np.asarray(a["is_drs_end"]).astype(float),
            "is_drs_detection": np.asarray(a["is_drs_detection"]).astype(float),
        }

    @classmethod
    def static_track(cls, track: Any) -> StaticTrack:
        """Static track data straight from a ``TrackModel`` (no graph needed)."""
        cols = cls.static_columns(
            {
                "s": track.s, "x": track.x, "y": track.y, "z": track.z, "heading": track.heading,
                "curvature": track.curvature, "gradient": track.gradient, "sector_idx": track.sector_idx,
                "corner_number": track.corner_number, "marshal_sector": track.marshal_sector,
                "drs_zone": track.drs_zone, "is_drs_start": track.is_drs_start, "is_drs_end": track.is_drs_end,
                "is_drs_detection": track.is_drs_detection,
            }
        )  # fmt: skip
        names = [n for n in TRACK_FEATURES if n not in TRACK_DYNAMIC]
        values, mask = cls._finish(np.column_stack([cols[n] for n in names]))
        return StaticTrack(
            names=names, values=values, mask=mask, xy_bbox=cls._bbox(track.x, track.y),
            length_m=track.length_m, spacing_m=track.spacing_m,
        )  # fmt: skip

    @staticmethod
    def _bbox(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        return (
            float((x.max() + x.min()) / 2),
            float((y.max() + y.min()) / 2),
            float(max(np.ptp(x), np.ptp(y)) / 2),
        )

    @staticmethod
    def driver_row(d: Mapping[str, Any]) -> list[float]:
        """Exported feature vector (``DRIVER_FEATURES`` order) of one driver from its node attributes."""
        h, hr = d["heading_rad"], d["heading_rel_track_rad"]
        comp = str(d["compound"]).upper()
        onehot = [comp == c for c in COMPOUNDS]
        return [
            d["x_m"], d["y_m"], d["z_m"], d["speed_kmh"], d["vx_mps"], d["vy_mps"],
            np.sin(h), np.cos(h), np.sin(hr), np.cos(hr),
            d["throttle"], d["brake"], d["gear"], d["rpm"], d["drs_open"], d["drs_eligible"],
            d["track_s_m"], d["lap_fraction"],
            np.sin(2 * np.pi * d["lap_fraction"]), np.cos(2 * np.pi * d["lap_fraction"]),
            d["lateral_offset_m"], d["progress_m"], d["lap_number"], d["position"], d["official_position"],
            d["gap_to_leader_s"], d["gap_ahead_s"], d["gap_behind_s"],
            d["tyre_age"], d["fresh_tyre"], d["stint"], d["pit_stops"], d["in_pit"], d["on_track"],
            d["pos_age_s"], d["car_age_s"], d["last_3_laps_average_seconds"], d["last_lap_time_s"],
            d["finished_or_retired"], *onehot, not any(onehot),
        ]  # fmt: skip

    @classmethod
    def global_row(cls, g: Mapping[str, Any]) -> list[float]:
        return [cls._f(g.get(name)) for name in GLOBAL_FEATURES]

    @classmethod
    def edge_row(cls, e: Mapping[str, Any]) -> list[float]:
        return [cls._f(e[k]) for k in EDGE_FEATURES]

    # ------------------------------------------------------------------ dense path (no graph)
    def export_snapshot(self, snap: Any, flags: np.ndarray, track_length_m: float) -> FrameRecord:
        """``Snapshot`` -> ``FrameRecord``: the same numbers ``export`` produces, without building a graph."""
        drivers = snap.drivers
        attrs = [d.to_attributes() for d in drivers]
        D, mask = self._finish(np.array([self.driver_row(a) for a in attrs], dtype=float).reshape(len(drivers), -1))
        gl, gl_mask = self._finish(np.array([self.global_row(snap.global_features)], dtype=float))
        E, E_mask = self._finish(
            np.array([self.edge_row(e) for _, _, e in snap.edges], dtype=float).reshape(len(snap.edges), -1)
        )
        total_laps = snap.global_features.get("total_laps")
        return FrameRecord(
            time_s=float(snap.time_s), driver_codes=[d.driver_code for d in drivers], driver_x=D, driver_mask=mask,
            driver_to_track=np.array([d.track_node for d in drivers], dtype=np.int64), global_x=gl[0],
            global_mask=gl_mask[0], edge_src=np.array([e[0] for e in snap.edges], dtype=np.int64),
            edge_dst=np.array([e[1] for e in snap.edges], dtype=np.int64), edge_attr=E, edge_mask=E_mask,
            track_flags=flags, race_distance_m=float(total_laps * track_length_m) if total_laps else None,
        )  # fmt: skip

    # ------------------------------------------------------------------ graph path
    def export(self, G: nx.DiGraph) -> GraphArrays:
        track = G.graph["track"]
        L = track.length_m
        t_nodes = sorted((d for _, d in G.nodes(data=True) if d.get("node_type") == "track"), key=lambda d: d["index"])
        codes: list[str] = list(G.graph["driver_codes"])
        d_nodes = [G.nodes[c] for c in codes]
        bbox = self._bbox(np.array([d["x_m"] for d in t_nodes]), np.array([d["y_m"] for d in t_nodes]))

        # ---- track: static columns via the shared helper, dynamic ones from the node attributes
        col = lambda key: np.array([d[key] for d in t_nodes])  # noqa: E731
        cols = self.static_columns(
            {
                "s": col("s_m"), "x": col("x_m"), "y": col("y_m"), "z": col("z_m"), "heading": col("heading_rad"),
                "curvature": col("curvature"), "gradient": col("gradient"), "sector_idx": col("sector_idx"),
                "corner_number": col("corner_number"), "marshal_sector": col("marshal_sector"),
                "drs_zone": col("drs_zone"), "is_drs_start": col("is_drs_start"), "is_drs_end": col("is_drs_end"),
                "is_drs_detection": col("is_drs_detection"),
            }
        )  # fmt: skip
        for name in TRACK_DYNAMIC:
            cols[name] = col(name).astype(float)
        T, T_mask = self._finish(np.column_stack([cols[n] for n in TRACK_FEATURES]))

        # ---- drivers
        D, D_mask = self._finish(np.array([self.driver_row(d) for d in d_nodes], dtype=float).reshape(len(d_nodes), -1))
        total_laps = G.graph["global"].get("total_laps")

        # ---- global
        gl, gl_mask = self._finish(np.array([self.global_row(G.graph["global"])], dtype=float))

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
                attrs.append(self.edge_row(e))
        E, E_mask = self._finish(np.array(attrs, dtype=float).reshape(len(attrs), len(EDGE_FEATURES)))

        arrays = GraphArrays(
            time_s=float(G.graph["time_s"]), track_length_m=L, track_spacing_m=track.spacing_m,
            track_x=T, track_mask=T_mask, track_names=list(TRACK_FEATURES),
            driver_x=D, driver_mask=D_mask, driver_names=list(DRIVER_FEATURES), driver_codes=codes,
            global_x=gl, global_mask=gl_mask, global_names=list(GLOBAL_FEATURES),
            track_edge_index=ring, driver_to_track=d2t,
            driver_edge_index=np.array([src, dst], dtype=np.int64).reshape(2, len(src)),
            driver_edge_attr=E, driver_edge_mask=E_mask, edge_names=list(EDGE_FEATURES),
            xy_bbox=bbox, race_distance_m=float(total_laps * L) if total_laps else None,
        ).validate()  # fmt: skip
        return arrays.normalised_copy().validate() if self.normalise else arrays
