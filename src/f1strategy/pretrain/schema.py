"""
schema.py - feature schema, pre-training groups, leak maps and normalisation for the F1 graph BERT.

Everything here is derived from the Parquet dataset written by ``f1strategy`` (manifest ``format_version`` 1):
  track.parquet            static track features, one row per ~10 m node
  drivers/part-*.parquet   one row per (frame, driver)
  global/part-*.parquet    one row per frame (+ flagged marshal sectors as lists)
  edges/part-*.parquet     one row per proximity edge and frame

The name lists below are checked against the manifest of every race when a cache is built, so a schema change
in the data pipeline fails loudly instead of silently shifting columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ----------------------------------------------------------------------------------------------- sizes
MAX_DRIVERS = 22  # padded driver slots (2019-2025 grids have 20; headroom for odd entries)
N_TRACK_TOKENS = 128  # the ring of ~10 m nodes is pooled to this many tokens, whatever the circuit length
MAX_MARSHAL = 40  # marshal sector ids are 1..n (about 18-30 on real circuits)

# ----------------------------------------------------------------------------------------------- names
TRACK_STATIC = [
    "s_m", "x_m", "y_m", "z_m", "heading_sin", "heading_cos", "curvature", "gradient",
    "sector_1", "sector_2", "sector_3", "is_corner", "corner_number", "marshal_sector",
    "in_drs_zone", "is_drs_start", "is_drs_end", "is_drs_detection",
]  # fmt: skip
FLAG_BITS = ["flag_yellow", "flag_double_yellow", "flag_sc", "flag_vsc", "flag_vsc_ending", "flag_red"]
COMPOUNDS = ["soft", "medium", "hard", "intermediate", "wet", "unknown"]
COMPOUND_COLS = [f"compound_{c}" for c in COMPOUNDS]
DRIVER_FEATURES = [
    "x_m", "y_m", "z_m", "speed_kmh", "vx_mps", "vy_mps", "heading_sin", "heading_cos",
    "heading_rel_track_sin", "heading_rel_track_cos", "throttle", "brake", "gear", "rpm",
    "drs_open", "drs_eligible", "track_s_m", "lap_fraction", "lap_fraction_sin", "lap_fraction_cos",
    "lateral_offset_m", "progress_m", "lap_number", "position", "official_position",
    "gap_to_leader_s", "gap_ahead_s", "gap_behind_s", "tyre_age", "fresh_tyre", "stint", "pit_stops",
    "in_pit", "on_track", "pos_age_s", "car_age_s", "pace_last3_s", "last_lap_time_s",
    "finished_or_retired",
] + COMPOUND_COLS  # fmt: skip
GLOBAL_FEATURES = [
    "time_s", "leader_lap", "laps_remaining", "race_fraction", "is_green", "is_yellow", "is_sc", "is_vsc",
    "is_vsc_ending", "is_red", "sc_in_this_lap", "yellow_unlocalised", "n_yellow_sectors", "drs_enabled",
    "air_temp", "track_temp", "humidity", "pressure", "rainfall", "wind_speed", "wind_direction_sin",
    "wind_direction_cos", "n_drivers",
]  # fmt: skip
EDGE_FEATURES = [
    "gap_s",
    "progress_gap_m",
    "circular_gap_m",
    "lap_diff",
    "rel_speed_kmh",
    "order_neighbor",
    "drs_window",
]

F_TRACK_STATIC = len(TRACK_STATIC)
F_DRIVER = len(DRIVER_FEATURES)
F_GLOBAL = len(GLOBAL_FEATURES)
F_EDGE = len(EDGE_FEATURES)
N_FLAGS = len(FLAG_BITS)

DI = {n: i for i, n in enumerate(DRIVER_FEATURES)}
GI = {n: i for i, n in enumerate(GLOBAL_FEATURES)}
EI = {n: i for i, n in enumerate(EDGE_FEATURES)}
TI = {n: i for i, n in enumerate(TRACK_STATIC)}

# binary features are predicted with BCE, the rest (continuous) with a Huber loss; compounds with cross-entropy
BIN_DRIVER = ["brake", "drs_open", "drs_eligible", "fresh_tyre", "in_pit", "on_track", "finished_or_retired"]
BIN_GLOBAL = [
    "is_green", "is_yellow", "is_sc", "is_vsc", "is_vsc_ending", "is_red", "sc_in_this_lap",
    "yellow_unlocalised", "drs_enabled", "rainfall",
]  # fmt: skip

# ----------------------------------------------------------------------------------------------- pretext task A
# Telemetry channels that are masked (cell-wise, 15 %) and the input columns that must be hidden with them.
# speed_kmh is ~0.96 * |(vx, vy)| for a car on track (measured on the 2019 Australian GP), so masking speed alone
# would be a copy task: the velocity components go with it.
TELEMETRY_TARGETS = ["speed_kmh", "throttle", "brake", "rpm"]
TELEMETRY_HIDES = {
    "speed_kmh": ["speed_kmh", "vx_mps", "vy_mps"],
    "throttle": ["throttle"],
    "brake": ["brake"],
    "rpm": ["rpm"],
}
# physical scale of the normalised value, to report errors in real units
PHYS_SCALE = {
    "speed_kmh": 350.0,
    "throttle": 100.0,
    "rpm": 13000.0,
    "tyre_age": 40.0,
    "pace_last3_s": 3.0,
    "position": 20.0,
}

# ----------------------------------------------------------------------------------------------- pretext task D
# Strategy state groups, masked per driver and group (15 %).
STRATEGY_GROUPS = {
    "compound": COMPOUND_COLS,
    "tyre": ["tyre_age", "stint", "pit_stops", "fresh_tyre"],
    "pace": ["pace_last3_s", "last_lap_time_s"],
    "gaps": ["gap_to_leader_s", "gap_ahead_s", "gap_behind_s", "position", "official_position"],
}

# ----------------------------------------------------------------------------------------------- edge leaks
# Proximity-edge features that repeat a masked driver feature, hidden on every edge touching that driver.
EDGE_LEAKS = {
    "speed": ["rel_speed_kmh"],  # target speed - source speed
    "gaps": ["gap_s", "order_neighbor", "drs_window"],  # drs_window is "within 1.0 s of the car ahead"
}

# ----------------------------------------------------------------------------------------------- pretext task C
# Global status features that restate the flag state (hidden together with the track flags, all of them targets).
GLOBAL_FLAG_FEATURES = [
    "is_green", "is_yellow", "is_sc", "is_vsc", "is_vsc_ending", "is_red", "sc_in_this_lap",
    "yellow_unlocalised", "n_yellow_sectors", "drs_enabled",
]  # fmt: skip
# Hidden (and predicted) when only some marshal sectors are masked: they would reveal whether a masked sector is flagged.
GLOBAL_LOCAL_FLAG_FEATURES = ["is_green", "is_yellow", "yellow_unlocalised", "n_yellow_sectors"]
FLAGGED_FRAME_COLS = ["is_yellow", "is_sc", "is_vsc", "is_vsc_ending", "is_red"]


# ----------------------------------------------------------------------------------------------- normalisation
@dataclass(frozen=True)
class RaceNorm:
    """Per-race constants: circuit coordinates differ per track, so x/y/z and distances are scaled per race."""

    track_length_m: float
    race_distance_m: float
    cx: float
    cy: float
    sxy: float  # half extent of the circuit bounding box
    zc: float
    pace_ref_s: float = 90.0  # median representative lap pace of the race: pace features are centred on it

    @staticmethod
    def from_track(
        track_x, track_y, track_z, track_length_m: float, race_distance_m: float, pace_ref_s: float = 90.0
    ) -> RaceNorm:
        cx = 0.5 * (float(np.min(track_x)) + float(np.max(track_x)))
        cy = 0.5 * (float(np.min(track_y)) + float(np.max(track_y)))
        sxy = 0.5 * max(float(np.ptp(track_x)), float(np.ptp(track_y)), 1.0)
        return RaceNorm(
            float(track_length_m), float(race_distance_m), cx, cy, sxy, float(np.mean(track_z)), float(pace_ref_s)
        )

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _spec(names: list[str], table: dict[str, tuple[float, float, float]]):
    off = np.zeros(len(names), np.float32)
    scale = np.ones(len(names), np.float32)
    asinh_a = np.zeros(len(names), np.float32)
    for i, n in enumerate(names):
        if n in table:
            off[i], scale[i], asinh_a[i] = table[n]
    return off, scale, asinh_a


def _apply(x: np.ndarray, spec) -> np.ndarray:
    """Normalise the last axis; NaN stays NaN (it means 'missing'). Gap-like features use asinh to tame the tails."""
    off, scale, a = spec
    x = x.astype(np.float32, copy=False)
    lin = (x - off) / scale
    with np.errstate(invalid="ignore", divide="ignore"):
        nl = np.arcsinh(x / np.where(a > 0, a, 1.0)) / scale
    out = np.where(a > 0, nl, lin)
    return np.clip(out, -10.0, 10.0).astype(np.float32)


def driver_spec(rn: RaceNorm):
    L, R = rn.track_length_m, rn.race_distance_m
    return _spec(DRIVER_FEATURES, {
        "x_m": (rn.cx, rn.sxy, 0), "y_m": (rn.cy, rn.sxy, 0), "z_m": (rn.zc, 50.0, 0),
        "speed_kmh": (0, 350, 0), "vx_mps": (0, 100, 0), "vy_mps": (0, 100, 0),
        "throttle": (0, 100, 0), "gear": (0, 8, 0), "rpm": (0, 13000, 0),
        "track_s_m": (0, L, 0), "lateral_offset_m": (0, 10, 0), "progress_m": (0, R, 0),
        "lap_number": (0, 70, 0), "position": (0, 20, 0), "official_position": (0, 20, 0),
        "gap_to_leader_s": (0, 2, 10), "gap_ahead_s": (0, 2, 10), "gap_behind_s": (0, 2, 10),
        "tyre_age": (0, 40, 0), "stint": (0, 4, 0), "pit_stops": (0, 3, 0),
        "pos_age_s": (0, 0.5, 0), "car_age_s": (0, 0.5, 0),
        "pace_last3_s": (rn.pace_ref_s, 3, 0), "last_lap_time_s": (rn.pace_ref_s, 10, 0),
    })  # fmt: skip


def track_spec(rn: RaceNorm):
    return _spec(TRACK_STATIC, {
        "s_m": (0, rn.track_length_m, 0), "x_m": (rn.cx, rn.sxy, 0), "y_m": (rn.cy, rn.sxy, 0),
        "z_m": (rn.zc, 50.0, 0), "curvature": (0, 0.02, 0), "gradient": (0, 0.1, 0),
        "corner_number": (0, 20, 0), "marshal_sector": (0, 30, 0),
    })  # fmt: skip


def global_spec(rn: RaceNorm):
    return _spec(GLOBAL_FEATURES, {
        "time_s": (0, 8000, 0), "leader_lap": (0, 70, 0), "laps_remaining": (0, 70, 0),
        "n_yellow_sectors": (0, 5, 0), "air_temp": (0, 40, 0), "track_temp": (0, 60, 0),
        "humidity": (0, 100, 0), "pressure": (0, 1000, 0), "wind_speed": (0, 10, 0), "n_drivers": (0, 20, 0),
    })  # fmt: skip


def edge_spec(rn: RaceNorm):
    L = rn.track_length_m
    return _spec(EDGE_FEATURES, {
        "gap_s": (0, 2, 10), "progress_gap_m": (0, L, 0), "circular_gap_m": (0, L, 0),
        "lap_diff": (0, 3, 0), "rel_speed_kmh": (0, 350, 0),
    })  # fmt: skip


normalise_driver = lambda x, rn: _apply(x, driver_spec(rn))  # noqa: E731
normalise_track = lambda x, rn: _apply(x, track_spec(rn))  # noqa: E731
normalise_global = lambda x, rn: _apply(x, global_spec(rn))  # noqa: E731
normalise_edge = lambda x, rn: _apply(x, edge_spec(rn))  # noqa: E731


def check_manifest(manifest: dict) -> None:
    """Fail loudly when the dataset schema differs from what this package was written for."""
    if manifest.get("format") != "f1strategy-parquet" or manifest.get("format_version") != 1:
        raise ValueError(f"unsupported dataset format: {manifest.get('format')} v{manifest.get('format_version')}")
    expected = {
        "track_static_names": TRACK_STATIC, "driver_names": DRIVER_FEATURES,
        "global_names": GLOBAL_FEATURES, "edge_names": EDGE_FEATURES, "flag_bits": FLAG_BITS,
    }  # fmt: skip
    for key, names in expected.items():
        if list(manifest[key]) != names:
            raise ValueError(
                f"manifest '{key}' differs from the schema this package expects:\n  {manifest[key]}\n  {names}"
            )
