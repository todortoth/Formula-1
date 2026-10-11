"""SectorNode (race control state of a sector) and DriverNode (container of a driver's state)."""

from __future__ import annotations

import math
from dataclasses import fields

import pytest

from f1strategy.domain.driver import DriverNode
from f1strategy.domain.sector import (
    STATUS_CLEAR,
    STATUS_RED,
    STATUS_SAFETY_CAR,
    STATUS_VSC,
    STATUS_VSC_ENDING,
    STATUS_YELLOW,
    SectorNode,
)


class TestSectorNode:
    def test_status_code_constants_match_fastf1(self):
        # fastf1.api.track_status_data: 1 clear, 2 yellow, 4 SC, 5 red, 6 VSC, 7 VSC ending
        assert (STATUS_CLEAR, STATUS_YELLOW, STATUS_SAFETY_CAR, STATUS_RED, STATUS_VSC, STATUS_VSC_ENDING) == (
            "1", "2", "4", "5", "6", "7",
        )  # fmt: skip

    def test_regression_seven_is_vsc_ending_not_red(self):
        node = SectorNode("S", "7", "VSCEnding")
        assert node.is_vsc_ending and not node.is_red_flag and node.flag_label == "VSC ENDING"
        assert SectorNode("S", "5", "Red").is_red_flag

    @pytest.mark.parametrize(
        "code, local, label",
        [
            ("1", None, "CLEAR"),
            ("2", None, "YELLOW"),
            ("1", "YELLOW", "YELLOW"),
            ("1", "DOUBLE YELLOW", "DOUBLE YELLOW"),
            ("4", None, "SC"),
            ("6", None, "VSC"),
            ("7", None, "VSC ENDING"),
            ("5", None, "RED"),
            # priority: the most restrictive condition wins
            ("4", "DOUBLE YELLOW", "SC"),
            ("5", "YELLOW", "RED"),
            ("6", "DOUBLE YELLOW", "VSC"),
            ("7", "YELLOW", "VSC ENDING"),
        ],
    )
    def test_flag_label(self, code, local, label):
        assert SectorNode("Sector 1", code, "msg", local_flag=local).flag_label == label

    def test_green_flag_only_when_clear_and_no_local_flag(self):
        assert SectorNode("S", "1", "x").is_green_flag
        assert not SectorNode("S", "1", "x", local_flag="YELLOW").is_green_flag
        assert not SectorNode("S", "2", "x").is_green_flag
        assert not SectorNode("S", "4", "x").is_green_flag

    def test_code_is_coerced_to_str(self):
        assert SectorNode("S", 4, "x").is_safety_car  # ints from other sources must still work

    def test_display_label(self):
        assert SectorNode("Sector 2", "1", "x").get_display_label() == "Sector 2"
        assert SectorNode("Sector 2", "4", "x").get_display_label() == "Sector 2\n[SC]"

    def test_repr_contains_state(self):
        r = repr(SectorNode("Sector 3", "2", "Yellow"))
        assert "Sector 3" in r and "Yellow" in r and "YELLOW" in r


class TestDriverNode:
    def _legacy(self):
        return DriverNode("VER", 1, "SOFT", 5, 1, 91.234)

    def test_legacy_positional_construction_still_works(self):
        d = self._legacy()
        assert (d.driver_code, d.position, d.compound, d.tyre_age, d.pit_stops) == ("VER", 1, "SOFT", 5, 1)
        assert d.nearby_drivers == [] and d.sector_snapshots == {} and d.current_sector is None

    def test_telemetry_defaults_are_nan_or_neutral(self):
        d = self._legacy()
        for name in ("x_m", "y_m", "speed_kmh", "heading_rad", "gap_ahead_s", "progress_m"):
            assert math.isnan(getattr(d, name))
        assert d.track_node == -1 and d.heading_valid is False and d.in_pit is False and d.on_track is True

    def test_mutable_defaults_are_not_shared_between_instances(self):
        a, b = self._legacy(), self._legacy()
        a.nearby_drivers.append({"x": 1})
        a.sector_snapshots["S1"] = {}
        assert b.nearby_drivers == [] and b.sector_snapshots == {}

    def test_to_attributes_excludes_containers_and_includes_everything_else(self):
        attrs = self._legacy().to_attributes()
        assert {"nearby_drivers", "sector_snapshots", "current_sector_node"}.isdisjoint(attrs)
        expected = {f.name for f in fields(DriverNode)} - {"nearby_drivers", "sector_snapshots", "current_sector_node"}
        assert set(attrs) == expected
        assert attrs["driver_code"] == "VER" and attrs["last_3_laps_average_seconds"] == 91.234
