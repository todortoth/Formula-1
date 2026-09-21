from dataclasses import dataclass

@dataclass
class DriverNode:
    driver_code: str
    position: int
    compound: str
    tyre_age: int
    pit_stops: int
    last_3_laps_average_seconds: float
    fuel_remaining: float
