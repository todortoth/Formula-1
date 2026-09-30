"""
SectorNode.py · Formula 1 Strategic Decisions
=========================================================
Represents a track sector with dynamic Race Controll attributes.
"""

class SectorNode:
    """Represents a Track Sector with dynamic Race Controll information (flags, SC, VSC)."""
    def __init__(self, sector_name: str, track_status_code: str, status_message: str) -> None:
        self.sector_name = sector_name
        self.track_status_code = track_status_code
        self.status_message = status_message

        self.is_safety_car: bool = (self.track_status_code == "4")
        self.is_vsc: bool = (self.track_status_code == "6")
        self.is_yellow_flag: bool = (self.track_status_code == "2")
        self.is_red_flag: bool = (self.track_status_code == "7")
        self.is_green_flag: bool = (self.track_status_code == "1")

    def get_display_label(self) -> str:
        """Returns a formatted label for graph visualization."""
        if self.is_safety_car:
            return f"{self.sector_name}\n[SC]"
        elif self.is_vsc:
            return f"{self.sector_name}\n[VSC]"
        elif self.is_yellow_flag:
            return f"{self.sector_name}\n[YELLOW]"
        elif self.is_red_flag:
            return f"{self.sector_name}\n[RED]"
        return self.sector_name

    def __repr__(self) -> str:
        return f"SectorNode({self.sector_name}, Status: {self.status_message})"
