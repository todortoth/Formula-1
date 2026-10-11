"""Edge cases of the original lap-level API (`all_drivers_at_lap` and helpers) that the graph code no longer uses."""

from __future__ import annotations

import pandas as pd
import pytest

from f1strategy import SessionAnalyser
from tests.helpers import clone_session


def _secs(x):
    return pd.to_timedelta(x, unit="s")


class TestEmptyAndMissingData:
    def test_unknown_lap_gives_empty_results(self, analyser, syn):
        assert analyser.all_drivers_at_lap(99) == []
        assert analyser._compute_snapshot_at_lap(99, syn.session.laps) == {}

    @pytest.mark.parametrize("which", ["none", "empty", "all_later", "nat_crossing"])
    def test_track_status_defaults_to_clear_when_nothing_is_known(self, syn, which):
        ts = syn.session.track_status
        status = {
            "none": None,
            "empty": ts.iloc[0:0],
            "all_later": ts.assign(Time=ts.Time + _secs(1000)),
            "nat_crossing": ts,
        }[which]
        a = SessionAnalyser.from_session(clone_session(syn.session, track_status=status))
        crossing = pd.NaT if which == "nat_crossing" else _secs(50)
        node = a._get_track_status_at_time(crossing, "Sector 1")
        assert node.is_green_flag and node.sector_name == "Sector 1" and node.status_message == "Clear"

    def test_retired_driver_without_a_row_for_the_target_lap_uses_the_last_known_one(self, analyser):
        sai = next(d for d in analyser.all_drivers_at_lap(5) if d.driver_code == "SAI")
        assert sai.position == 99 and sai.compound == "HARD" and sai.pit_stops == 0  # 99 = legacy 'unclassified'

    def test_leader_row_falls_back_to_the_first_row_when_nobody_is_classified_first(self, analyser, syn):
        laps = syn.session.laps[syn.session.laps.LapNumber == 3].copy()
        laps["Position"] = 5.0
        assert analyser._get_leader_row(laps).equals(laps.iloc[0])

    def test_nearby_drivers_empty_without_a_lap_end_time(self, analyser, syn):
        laps = syn.session.laps[syn.session.laps.LapNumber == 3].copy()
        laps.loc[laps.Driver == "VER", "Time"] = pd.NaT
        assert analyser._compute_nearby_drivers("VER", laps) == []
        assert analyser._compute_nearby_drivers("ZZZ", laps) == []


class TestLappedDriver:
    @pytest.fixture()
    def lapped(self, syn):
        laps = syn.session.laps.copy()
        mask = laps.Driver == "SAI"
        for col in ("Time", "LapStartTime", "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime"):
            laps.loc[mask, col] = laps.loc[mask, col] + _secs(75)  # slower than a lap behind the leader
        return SessionAnalyser.from_session(clone_session(syn.session, laps=laps))

    def test_a_car_more_than_a_lap_down_is_reported_as_lapped(self, lapped):
        sai = next(d for d in lapped.all_drivers_at_lap(3) if d.driver_code == "SAI")
        for sector in ("Sector 1", "Sector 2", "Sector 3"):
            snap = sai.sector_snapshots[sector]
            assert snap["lap_diff"] == 1 and 10 < snap["time_gap"] < 30  # gap to the leader's previous-lap crossing

    def test_a_lapped_car_has_no_nearby_cars_within_five_seconds(self, lapped):
        sai = next(d for d in lapped.all_drivers_at_lap(3) if d.driver_code == "SAI")
        assert sai.nearby_drivers == []
