"""Console tables of the lap-level snapshot."""

from __future__ import annotations

import io

import numpy as np

from f1strategy.domain.driver import DriverNode
from f1strategy.reporting import print_driver_table, print_sector_info


def _out(fn, *args):
    buf = io.StringIO()
    fn(*args, file=buf)
    return buf.getvalue()


def _driver(code="AAA", position=2, **kw):
    return DriverNode(code, position, "SOFT", 7, 1, kw.pop("avg", 91.5), **kw)


class TestDriverTable:
    def test_one_row_per_driver_with_header(self, analyser):
        text = _out(print_driver_table, analyser.all_drivers_at_lap(3), 3)
        lines = text.strip().splitlines()
        assert "lap 3" in lines[0] and lines[1].startswith("Driver") and set(lines[2]) == {"-"}
        assert [ln.split("|")[0].strip() for ln in lines[3:]] == ["VER", "NOR", "LEC", "HAM", "ALO", "SAI"]

    def test_nearby_drivers_are_rendered_with_sign(self):
        d = _driver(nearby_drivers=[{"driver_code": "BBB", "time_gap": 0.5, "status": "behind"},
                                    {"driver_code": "CCC", "time_gap": -1.25, "status": "ahead"}])  # fmt: skip
        text = _out(print_driver_table, [d], 10)
        assert "BBB (+0.5s)" in text and "CCC (-1.25s)" in text

    def test_no_nearby_and_unknown_sector(self):
        text = _out(print_driver_table, [_driver()], 1)
        assert text.rstrip().endswith("None") and "Unknown" in text

    def test_pace_is_formatted_as_minutes(self):
        assert "1:31:500" in _out(print_driver_table, [_driver(avg=91.5)], 1)
        assert "N/A" in _out(print_driver_table, [_driver(avg=float("nan"))], 1)

    def test_sector_status_text(self, analyser):
        text = _out(print_driver_table, analyser.all_drivers_at_lap(3), 3)
        assert "SCDeployed" in text and "AllClear" in text

    def test_prints_to_stdout_by_default(self, capsys):
        print_driver_table([_driver()], 5)
        assert "AAA" in capsys.readouterr().out


class TestSectorInfo:
    def _snap(self, lap_diff=0, gap=0.0):
        return {
            "driver_code": "AAA",
            "lap_number": 3,
            "sector_at_moment": "Sector 2",
            "time_gap": gap,
            "lap_diff": lap_diff,
        }

    def test_leader_and_gap_formats(self):
        leader = _driver("LDR", 1, sector_snapshots={"Sector 1": self._snap()})
        behind = _driver("BHD", 2, sector_snapshots={"Sector 1": self._snap(gap=0.264)})
        ahead = _driver("AHD", 3, sector_snapshots={"Sector 1": self._snap(gap=-0.1)})
        text = _out(print_sector_info, [leader, behind, ahead])
        assert "Leader" in text and "+0.264s" in text and "-0.100s" in text

    def test_lapped_and_unknown_gaps(self):
        one = _driver("L1", 5, sector_snapshots={"Sector 1": self._snap(lap_diff=1)})
        two = _driver("L2", 6, sector_snapshots={"Sector 1": self._snap(lap_diff=2)})
        nan = _driver("NAN", 7, sector_snapshots={"Sector 1": self._snap(gap=np.nan)})
        text = _out(print_sector_info, [one, two, nan])
        assert "+1 lap\n" in text and "+2 laps" in text and "N/A" in text

    def test_drivers_without_snapshot_for_a_sector_are_skipped(self):
        text = _out(print_sector_info, [_driver("AAA", 2)])
        assert "AAA" not in text and text.count("--- Leader finished") == 3

    def test_real_snapshot(self, analyser):
        text = _out(print_sector_info, analyser.all_drivers_at_lap(3))
        assert "Leader finished: Sector 2" in text and text.count("Leader\n") >= 3
