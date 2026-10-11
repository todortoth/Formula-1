"""
SectorNodeClass.py · Formula 1 Strategic Decisions
=========================================================
Represents the Race Control state (flags, SC, VSC) that applies to a piece of track.

FastF1 track status codes (fastf1.api.track_status_data):
    '1' track clear   '2' yellow   '4' safety car   '5' red flag
    '6' VSC deployed  '7' VSC ending (status '1' marks the actual end)
(NOTE: '3' has never been observed. An earlier version of this class treated '7' as
 the red flag, which is actually VSC-ending.)

Sector-specific yellow / double-yellow flags are not part of the track status feed, they only
exist as race control messages (marshal sector N). That is passed in as ``local_flag``.
"""

STATUS_CLEAR = "1"
STATUS_YELLOW = "2"
STATUS_SAFETY_CAR = "4"
STATUS_RED = "5"
STATUS_VSC = "6"
STATUS_VSC_ENDING = "7"


class SectorNode:
    """Represents a Track Sector with dynamic Race Control information (flags, SC, VSC)."""

    def __init__(
        self, sector_name: str, track_status_code: str, status_message: str, local_flag: str | None = None
    ) -> None:
        self.sector_name = sector_name
        self.track_status_code = str(track_status_code)
        self.status_message = status_message
        self.local_flag = local_flag  # None | 'YELLOW' | 'DOUBLE YELLOW' (marshal-sector specific)

        code = self.track_status_code
        self.is_safety_car: bool = code == STATUS_SAFETY_CAR
        self.is_vsc: bool = code == STATUS_VSC
        self.is_vsc_ending: bool = code == STATUS_VSC_ENDING
        self.is_red_flag: bool = code == STATUS_RED
        self.is_double_yellow: bool = local_flag == "DOUBLE YELLOW"
        self.is_yellow_flag: bool = (code == STATUS_YELLOW) or (local_flag in ("YELLOW", "DOUBLE YELLOW"))
        self.is_green_flag: bool = (code == STATUS_CLEAR) and not self.is_yellow_flag

    @property
    def flag_label(self) -> str:
        """Single label with the most restrictive condition first."""
        if self.is_red_flag:
            return "RED"
        if self.is_safety_car:
            return "SC"
        if self.is_vsc_ending:
            return "VSC ENDING"
        if self.is_vsc:
            return "VSC"
        if self.is_double_yellow:
            return "DOUBLE YELLOW"
        if self.is_yellow_flag:
            return "YELLOW"
        return "CLEAR"

    def get_display_label(self) -> str:
        """Returns a formatted label for graph visualization."""
        if self.flag_label == "CLEAR":
            return self.sector_name
        return f"{self.sector_name}\n[{self.flag_label}]"

    def __repr__(self) -> str:
        return f"SectorNode({self.sector_name}, Status: {self.status_message}, Flag: {self.flag_label})"
