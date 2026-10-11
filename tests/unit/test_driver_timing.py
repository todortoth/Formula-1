"""DriverTiming: tyres, pit stops, pace, retirement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1strategy import DriverTiming


class TestTiming:
    def test_pit_stop_and_tyre_change(self, syn):
        timing, laps = DriverTiming(syn.session.laps), syn.session.laps
        nor3 = laps[(laps.Driver == "NOR") & (laps.LapNumber == 3)].iloc[0]
        nor4 = laps[(laps.Driver == "NOR") & (laps.LapNumber == 4)].iloc[0]
        t_in, t_out = nor3.PitInTime.total_seconds(), nor4.PitOutTime.total_seconds()
        before, during, after = timing.at("NOR", t_in - 5), timing.at("NOR", t_in + 1), timing.at("NOR", t_out + 1)
        assert (before.pit_stops, before.in_pit, before.compound, before.stint) == (0, False, "MEDIUM", 1)
        assert during.in_pit and during.pit_stops == 1
        assert (after.in_pit, after.pit_stops, after.compound, after.tyre_age, after.stint) == (False, 1, "HARD", 1, 2)
        assert timing.at("VER", t_in + 1).pit_stops == 0

    def test_retirement(self, syn):
        timing = DriverTiming(syn.session.laps)
        end = timing.last_end_time("SAI")
        assert not timing.at("SAI", end - 5).finished_or_retired and timing.at("SAI", end + 30).finished_or_retired
        assert not timing.at("VER", end + 30).finished_or_retired
        after = timing.at("SAI", end + 30)
        # FastF1 appends a partial last lap (Time = abort time, LapTime NaT) for a retiring driver: it counts as ended
        # but has no lap time, and it must not leak into pace
        assert after.laps_completed == 4 and np.isnan(after.last_lap_time_s)
        assert after.pace_last3_s == pytest.approx(np.mean([58.362145992, 58.915906952, 58.362145992]), abs=0.6)

    def test_pace_ignores_safety_car_and_is_causal(self, syn):
        timing, laps = DriverTiming(syn.session.laps), syn.session.laps
        v = laps[laps.Driver == "VER"].set_index("LapNumber")
        t_after_5 = v.loc[5, "Time"].total_seconds() + 1
        expected = np.mean([v.loc[k, "LapTime"].total_seconds() for k in (5, 2, 1)])  # laps 3 and 4 had a SC
        assert timing.at("VER", t_after_5).pace_last3_s == pytest.approx(expected)
        t_mid_lap_2 = v.loc[2, "LapStartTime"].total_seconds() + 10
        assert timing.at("VER", t_mid_lap_2).last_lap_time_s == pytest.approx(v.loc[1, "LapTime"].total_seconds())
        assert timing.at("VER", t_mid_lap_2).lap_number == 2

    def test_official_position_is_from_last_completed_lap(self, syn):
        timing, laps = DriverTiming(syn.session.laps), syn.session.laps
        t = laps[(laps.Driver == "VER") & (laps.LapNumber == 2)].iloc[0].Time.total_seconds() + 1
        assert timing.at("LEC", t).official_position == 3.0


def _secs(*values):
    return pd.to_timedelta(list(values), unit="s")


def _laps(**cols):
    """One driver 'AAA' with 4 laps of 90 s starting at t=0; every column can be overridden."""
    n = 4
    base = {
        "Driver": ["AAA"] * n,
        "LapNumber": [1.0, 2.0, 3.0, 4.0],
        "LapStartTime": _secs(0, 90, 180, 270),
        "Time": _secs(90, 180, 270, 360),
        "LapTime": _secs(90, 90, 90, 90),
    }
    base.update(cols)
    return pd.DataFrame(base)


class TestTimingEdgeCases:
    def test_missing_optional_columns_fall_back_to_neutral_values(self):
        st = DriverTiming(_laps()).at("AAA", 100.0)
        assert (st.compound, st.tyre_age, st.stint, st.fresh_tyre, st.pit_stops, st.in_pit) == ("UNKNOWN", 0, 0, False, 0, False)  # fmt: skip
        assert np.isnan(st.official_position) and st.lap_number == 2 and st.laps_completed == 1

    def test_none_and_nan_compound_become_unknown_and_values_are_upper_cased(self):
        t = DriverTiming(_laps(Compound=["soft", None, np.nan, "Medium"]))
        assert [t.at("AAA", s).compound for s in (10, 100, 200, 300)] == ["SOFT", "UNKNOWN", "UNKNOWN", "MEDIUM"]

    def test_missing_lap_start_is_filled_from_the_previous_lap_end(self):
        starts = _secs(0, 90, 180, 270)
        starts = starts.where([True, False, True, True])  # lap 2 start is NaT
        t = DriverTiming(_laps(LapStartTime=starts))
        assert t.at("AAA", 89.0).lap_number == 1 and t.at("AAA", 91.0).lap_number == 2

    def test_missing_first_lap_start_means_lap_one_from_the_beginning(self):
        starts = _secs(0, 90, 180, 270).where([False, True, True, True])
        assert DriverTiming(_laps(LapStartTime=starts)).at("AAA", -50.0).lap_number == 1

    def test_unknown_driver(self):
        t = DriverTiming(_laps())
        assert t.has("AAA") and not t.has("ZZZ")
        with pytest.raises(KeyError):
            t.at("ZZZ", 1.0)

    def test_before_the_first_lap_nothing_is_completed(self):
        st = DriverTiming(_laps()).at("AAA", -10.0)
        assert st.lap_number == 1 and st.laps_completed == 0 and np.isnan(st.last_lap_time_s) and not st.finished_or_retired  # fmt: skip

    def test_laps_are_sorted_even_if_the_table_is_not(self):
        df = _laps().iloc[::-1].reset_index(drop=True)
        assert DriverTiming(df).at("AAA", 200.0).lap_number == 3

    def test_pit_entry_counts_immediately_and_never_looks_at_future_rows(self):
        """A pit entry on the last known lap counts as a stop; the count must not depend on later rows."""
        pit_in = _secs(0, 0, 0, 355).where([False, False, False, True])
        full = DriverTiming(_laps(PitInTime=pit_in))
        truncated = DriverTiming(_laps(PitInTime=pit_in).iloc[:4])
        for t in (300.0, 356.0, 400.0):
            assert full.at("AAA", t).pit_stops == truncated.at("AAA", t).pit_stops
        assert full.at("AAA", 356.0).pit_stops == 1 and full.at("AAA", 300.0).pit_stops == 0

    def test_in_pit_lasts_until_the_next_laps_pit_out_time(self):
        t = DriverTiming(_laps(PitInTime=_secs(0, 85, 0, 0).where([False, True, False, False]),
                               PitOutTime=_secs(0, 0, 200, 0).where([False, False, True, False])))  # fmt: skip
        assert [t.at("AAA", s).in_pit for s in (80, 86, 150, 199, 201)] == [False, True, True, True, False]

    def test_car_that_never_leaves_the_pits_stays_in_pit(self):
        t = DriverTiming(_laps(PitInTime=_secs(0, 85, 0, 0).where([False, True, False, False])))
        assert t.at("AAA", 340.0).in_pit

    def test_two_stops(self):
        pit_in = _secs(0, 80, 0, 350).where([False, True, False, True])
        t = DriverTiming(_laps(PitInTime=pit_in))
        assert [t.at("AAA", s).pit_stops for s in (50, 100, 340, 352)] == [0, 1, 1, 2]

    def test_pace_is_nan_when_every_lap_was_disturbed_or_a_pit_lap(self):
        t = DriverTiming(_laps(TrackStatus=["14", "4", "6", "7"]))
        assert np.isnan(t.at("AAA", 365.0).pace_last3_s)
        t2 = DriverTiming(_laps(PitInTime=_secs(80, 170, 260, 350), PitOutTime=_secs(1, 91, 181, 271)))
        assert np.isnan(t2.at("AAA", 365.0).pace_last3_s)

    def test_yellow_flag_laps_still_count_for_pace_but_sc_laps_do_not(self):
        t = DriverTiming(_laps(LapTime=_secs(90, 91, 92, 93), TrackStatus=["1", "2", "14", "1"]))
        assert t.at("AAA", 365.0).pace_last3_s == pytest.approx(np.mean([93, 91, 90]))

    def test_pace_uses_at_most_the_last_three_clean_laps(self):
        t = DriverTiming(_laps(LapTime=_secs(100, 90, 91, 92)))
        assert t.at("AAA", 365.0).pace_last3_s == pytest.approx(np.mean([90, 91, 92]))

    def test_finished_flag_has_a_five_second_grace_period(self):
        t = DriverTiming(_laps())
        assert not t.at("AAA", 364.9).finished_or_retired and t.at("AAA", 365.1).finished_or_retired

    def test_last_end_time_is_nan_without_any_timestamps(self):
        df = _laps(Time=pd.to_timedelta([None] * 4, unit="s"))
        t = DriverTiming(df)
        assert np.isnan(t.last_end_time("AAA")) and not t.at("AAA", 1e6).finished_or_retired

    def test_several_drivers_do_not_mix(self):
        a, b = _laps(), _laps().assign(Driver="BBB", Compound="HARD")
        t = DriverTiming(pd.concat([a, b]))
        assert t.at("AAA", 10).compound == "UNKNOWN" and t.at("BBB", 10).compound == "HARD"

    def test_numeric_columns_with_missing_values(self):
        t = DriverTiming(
            _laps(TyreLife=[1.0, np.nan, 3.0, 4.0], Stint=[1.0, 1.0, np.nan, 2.0], FreshTyre=[True, None, False, True])
        )
        assert (t.at("AAA", 100).tyre_age, t.at("AAA", 100).fresh_tyre) == (0, False)
        assert t.at("AAA", 200).stint == 0 and t.at("AAA", 300).tyre_age == 4
