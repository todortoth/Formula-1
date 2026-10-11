"""
DriverNodeClass.py · Formula 1 Strategic Decisions
=========================================================
Defines the DriverNode class, acting as a structured container
for an individual driver's telemetry compound, position, etc.

The first block of fields is the original (lap-level) data. Everything below
"--- telemetry ---" is filled by the telemetry based snapshot builder (RaceGraphBuilderClass)
and has defaults, so older code constructing a DriverNode with only the original fields still works.
Units: metres, km/h, m/s, radians, seconds.
"""

from dataclasses import dataclass, field, fields
from typing import Any

from f1strategy.domain.sector import SectorNode

_NAN = float("nan")

# Fields that are containers / objects and therefore NOT plain node features
_NON_FEATURE_FIELDS = {"nearby_drivers", "sector_snapshots", "current_sector_node"}


@dataclass
class DriverNode:
    """Represents a driver's state and telemetry data at a specific point in a session."""

    driver_code: str
    position: int
    compound: str
    tyre_age: int
    pit_stops: int
    last_3_laps_average_seconds: float
    nearby_drivers: list[dict[str, Any]] = field(default_factory=list)
    sector_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_sector: str | None = None
    current_sector_node: SectorNode | None = None

    # --- telemetry: kinematics (world coordinates of the FastF1 position feed, converted to metres) ---
    x_m: float = _NAN
    y_m: float = _NAN
    z_m: float = _NAN
    speed_kmh: float = _NAN
    vx_mps: float = _NAN
    vy_mps: float = _NAN
    heading_rad: float = _NAN  # direction of travel, atan2(vy, vx)
    heading_valid: bool = False  # False -> car (almost) stationary, heading taken from the track tangent
    heading_rel_track_rad: float = _NAN  # heading minus track tangent: ~0 when aligned, large when spinning

    # --- telemetry: driver inputs ---
    throttle: float = _NAN
    brake: float = _NAN
    gear: float = _NAN
    rpm: float = _NAN
    drs_open: bool = False
    drs_eligible: bool = False  # raw DRS channel == 8 ("detected / eligible"), semantics not officially documented

    # --- position on the track graph ---
    track_node: int = -1  # index of the nearest 10 m track node
    track_s_m: float = _NAN  # distance from the start/finish line along the lap
    lateral_offset_m: float = _NAN  # signed distance from the racing line (+ = left)
    progress_m: float = _NAN  # cumulative race distance (laps * length + track_s_m)
    lap_number: int = 0
    lap_fraction: float = _NAN
    on_track: bool = True  # FastF1 position Status == 'OnTrack'

    # --- timing / strategy ---
    gap_to_leader_s: float = _NAN
    gap_ahead_s: float = _NAN
    gap_behind_s: float = _NAN
    official_position: float = _NAN  # classified position at the end of the last completed lap
    last_lap_time_s: float = _NAN
    fresh_tyre: bool = False
    stint: int = 0
    in_pit: bool = False
    laps_completed: int = 0
    finished_or_retired: bool = False

    def to_attributes(self) -> dict[str, Any]:
        """Flat dict of node attributes (everything except container/object fields)."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in _NON_FEATURE_FIELDS}
