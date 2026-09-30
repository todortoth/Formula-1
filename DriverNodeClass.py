"""
DriverNodeClass.py · Formula 1 Strategic Decisions
=========================================================
Defines the DriverNode class, acting as a structured container
for an individual driver's telemetry compound, position, etc.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

from SectorNodeClass import SectorNode


@dataclass
class DriverNode:
    """Represents a driver's state and telemetry data at a specific point in a session."""
    driver_code: str
    position: int
    compound: str
    tyre_age: int
    pit_stops: int
    last_3_laps_average_seconds: float
    nearby_drivers: List[Dict[str, Any]] = field(default_factory=list)
    sector_snapshots: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    current_sector: str = None
    current_sector_node: SectorNode = None
