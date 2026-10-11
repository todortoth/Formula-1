"""RaceControlTimeline: replay of track status + race control messages."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from f1strategy import RaceControlTimeline
from f1strategy.domain.sector import SectorNode


class TestTimeline:
    @pytest.mark.parametrize(
        "t, status, sectors, drs, sc_in",
        [
            (50.0, "1", {}, False, False),
            (96.0, "2", {3: "YELLOW"}, False, False),
            (101.0, "2", {3: "YELLOW"}, True, False),
            (111.0, "2", {3: "DOUBLE YELLOW"}, True, False),
            (121.0, "2", {3: "DOUBLE YELLOW", 8: "YELLOW"}, True, False),
            (129.0, "1", {}, True, False),
            (181.0, "4", {}, False, False),
            (236.0, "4", {}, False, True),
            (246.0, "1", {}, False, False),
            (251.0, "1", {}, True, False),
            (340.0, "6", {}, True, False),
            (356.0, "7", {}, True, False),
            (361.0, "1", {}, True, False),
        ],
    )
    def test_state(self, syn, t, status, sectors, drs, sc_in):
        rc = RaceControlTimeline.from_session(syn.session).state_at(t)
        assert (rc.track_status, rc.sector_flags, rc.drs_enabled, rc.sc_in_this_lap) == (status, sectors, drs, sc_in)

    def test_before_first_event_and_missing_feeds(self, syn):
        assert RaceControlTimeline.from_session(syn.session).state_at(-5.0).is_clear
        empty = RaceControlTimeline(None, None, None)
        st = empty.state_at(100.0)
        assert st.is_clear and st.drs_enabled is None  # unknown, not "disabled"

    def test_helpers(self, syn):
        tl = RaceControlTimeline.from_session(syn.session)
        assert tl.state_at(184).is_safety_car and tl.state_at(340).is_vsc and tl.state_at(356).is_vsc_ending
        assert tl.state_at(96).is_yellow and not tl.state_at(96).yellow_unlocalised

    def test_yellow_without_sector_messages_is_flagged_unlocalised(self, syn):
        tl = RaceControlTimeline(syn.session.track_status, None, None)
        assert tl.state_at(96).yellow_unlocalised

    def test_sector_node_status_codes(self):
        assert SectorNode("S", "5", "Red").is_red_flag and not SectorNode("S", "7", "x").is_red_flag
        assert SectorNode("S", "7", "VSCEnding").flag_label == "VSC ENDING"
        assert SectorNode("S", "1", "Clear", local_flag="DOUBLE YELLOW").flag_label == "DOUBLE YELLOW"
        assert SectorNode("S", "1", "Clear").is_green_flag
        assert not SectorNode("S", "1", "Clear", local_flag="YELLOW").is_green_flag


T0 = pd.Timestamp("2025-01-01 12:00:00")
MSG_COLS = ["Time", "Category", "Message", "Status", "Flag", "Scope", "Sector", "RacingNumber", "Lap"]


def _msg(t, category="Flag", message="", flag=None, scope=None, sector=None, tz=None):
    when = T0 + pd.to_timedelta(t, unit="s")
    if tz:
        when = when.tz_localize(tz)
    return dict(zip(MSG_COLS, [when, category, message, None, flag, scope, sector, None, None], strict=True))


def _tl(rows, status=None, t0=T0):
    df = pd.DataFrame(rows, columns=MSG_COLS) if rows else None
    ts = None
    if status:
        ts = pd.DataFrame({"Time": pd.to_timedelta([s[0] for s in status], unit="s"),
                           "Status": [s[1] for s in status], "Message": [s[2] for s in status]})  # fmt: skip
    return RaceControlTimeline(ts, df, t0)


class TestTimelineEdgeCases:
    def test_track_clear_message_clears_every_sector(self):
        tl = _tl([_msg(10, flag="YELLOW", scope="Sector", sector=3), _msg(11, flag="YELLOW", scope="Sector", sector=5),
                  _msg(20, flag="CLEAR", scope="Track")])  # fmt: skip
        assert tl.state_at(15).sector_flags == {3: "YELLOW", 5: "YELLOW"}
        assert tl.state_at(21).sector_flags == {}

    def test_green_clears_a_sector_like_clear(self):
        tl = _tl([_msg(10, flag="YELLOW", scope="Sector", sector=3), _msg(12, flag="GREEN", scope="Sector", sector=3)])
        assert tl.state_at(11).sector_flags == {3: "YELLOW"} and tl.state_at(13).sector_flags == {}

    def test_yellow_can_escalate_to_double_yellow_and_back(self):
        tl = _tl([_msg(10, flag="YELLOW", scope="Sector", sector=2),
                  _msg(12, flag="DOUBLE YELLOW", scope="Sector", sector=2),
                  _msg(14, flag="YELLOW", scope="Sector", sector=2)])  # fmt: skip
        assert [tl.state_at(t).sector_flags[2] for t in (11, 13, 15)] == ["YELLOW", "DOUBLE YELLOW", "YELLOW"]

    def test_sector_message_without_sector_number_is_ignored(self):
        tl = _tl([_msg(10, flag="YELLOW", scope="Sector", sector=None)])
        assert tl.state_at(11).sector_flags == {}

    def test_driver_scoped_flags_do_not_create_sector_flags(self):
        tl = _tl([_msg(10, flag="BLUE", scope="Driver", message="CAR 44 BLUE")])
        assert tl.state_at(11).is_clear

    @pytest.mark.parametrize("tz", ["UTC", "Europe/Budapest"])
    def test_timezone_aware_message_times_are_converted_to_naive_utc(self, tz):
        naive = _tl([_msg(10, flag="YELLOW", scope="Sector", sector=1)])
        aware = _tl([_msg(10, flag="YELLOW", scope="Sector", sector=1)])
        # build the same wall-clock instant expressed in `tz`
        row = _msg(10, flag="YELLOW", scope="Sector", sector=1)
        row["Time"] = (T0.tz_localize("UTC") + pd.to_timedelta(10, unit="s")).tz_convert(tz)
        aware = _tl([row])
        assert aware.state_at(9).is_clear and aware.state_at(11).sector_flags == {1: "YELLOW"}
        assert naive.state_at(11).sector_flags == aware.state_at(11).sector_flags

    def test_timezone_aware_t0(self):
        tl = _tl([_msg(10, flag="YELLOW", scope="Sector", sector=1)], t0=T0.tz_localize("UTC"))
        assert tl.state_at(11).sector_flags == {1: "YELLOW"}

    def test_nat_message_time_is_skipped_not_fatal(self):
        bad = _msg(5, flag="YELLOW", scope="Sector", sector=9)
        bad["Time"] = pd.NaT
        tl = _tl([bad, _msg(10, flag="YELLOW", scope="Sector", sector=1)])
        assert tl.state_at(20).sector_flags == {1: "YELLOW"}

    def test_drs_messages_recognised_by_text_even_without_category(self):
        tl = _tl(
            [_msg(10, category="Other", message="DRS ENABLED"), _msg(20, category="Other", message="DRS DISABLED")]
        )
        assert [tl.state_at(t).drs_enabled for t in (5, 15, 25)] == [False, True, False]

    def test_input_order_does_not_matter(self):
        rows = [_msg(30, flag="CLEAR", scope="Sector", sector=1), _msg(10, flag="YELLOW", scope="Sector", sector=1)]
        assert _tl(rows).state_at(20).sector_flags == {1: "YELLOW"} and _tl(rows).state_at(40).sector_flags == {}

    def test_status_is_applied_before_messages_at_the_same_instant(self):
        """All-clear and a *new* yellow in the same second: the yellow must survive."""
        tl = _tl(
            [_msg(10, flag="YELLOW", scope="Sector", sector=4)], status=[(0, "2", "Yellow"), (10, "1", "AllClear")]
        )
        assert tl.state_at(11).sector_flags == {4: "YELLOW"}

    def test_red_flag(self):
        st = _tl([], status=[(5, "5", "Red")]).state_at(6)
        assert st.is_red and not st.is_clear and not st.is_safety_car

    def test_all_clear_resets_sc_in_this_lap(self):
        tl = _tl([_msg(10, category="SafetyCar", message="SAFETY CAR IN THIS LAP")],
                 status=[(0, "4", "SCDeployed"), (20, "1", "AllClear")])  # fmt: skip
        assert tl.state_at(15).sc_in_this_lap and not tl.state_at(25).sc_in_this_lap

    def test_before_first_event_and_far_future(self):
        tl = _tl([_msg(10, flag="YELLOW", scope="Sector", sector=1)], status=[(10, "2", "Yellow")])
        assert tl.state_at(-1e9).time_s == float("-inf") and tl.state_at(1e9).track_status == "2"

    def test_no_feeds_at_all(self):
        st = RaceControlTimeline(None, None, None).state_at(5.0)
        assert st.is_clear and st.drs_enabled is None and st.sector_flags == {}

    def test_a_feed_without_any_drs_message_means_drs_was_never_enabled(self):
        """e.g. a wet race. Not 'unknown': that would depend on knowing the future of the feed."""
        tl = _tl([_msg(10, flag="YELLOW", scope="Sector", sector=1)])
        assert [tl.state_at(t).drs_enabled for t in (-5, 5, 50)] == [False, False, False]

    def test_drs_state_does_not_depend_on_later_messages(self):
        base = [_msg(10, flag="YELLOW", scope="Sector", sector=1)]
        later = base + [_msg(500, category="Drs", message="DRS ENABLED")]
        for t in (-1, 5, 20, 499):
            assert _tl(base).state_at(t).drs_enabled == _tl(later).state_at(t).drs_enabled

    def test_messages_without_t0_are_ignored(self):
        tl = _tl([_msg(10, flag="YELLOW", scope="Sector", sector=1)], t0=None)
        assert tl.state_at(20).is_clear and tl.state_at(20).drs_enabled is None  # unusable feed == no feed

    def test_from_session_survives_unloaded_data(self):
        class Unloaded:
            @property
            def track_status(self):
                raise RuntimeError("DataNotLoadedError")

            race_control_messages = property(lambda self: (_ for _ in ()).throw(RuntimeError("not loaded")))
            t0_date = property(lambda self: (_ for _ in ()).throw(RuntimeError("not loaded")))

        assert RaceControlTimeline.from_session(Unloaded()).state_at(1.0).is_clear

    def test_from_session_reads_all_three_sources(self):
        sess = SimpleNamespace(
            track_status=pd.DataFrame({"Time": pd.to_timedelta([5], unit="s"), "Status": ["4"], "Message": ["SC"]}),
            race_control_messages=pd.DataFrame([_msg(1, flag="YELLOW", scope="Sector", sector=2)], columns=MSG_COLS),
            t0_date=T0,
        )
        st = RaceControlTimeline.from_session(sess).state_at(6)
        assert st.is_safety_car and st.sector_flags == {2: "YELLOW"}

    def test_to_frame_lists_every_state_change(self):
        tl = _tl([_msg(10, flag="YELLOW", scope="Sector", sector=1)], status=[(5, "2", "Yellow")])
        df = tl.to_frame()
        assert len(df) == 2 and list(df.columns) == ["time_s", "track_status", "message", "sector_flags", "drs_enabled", "sc_in_this_lap"]  # fmt: skip
        assert RaceControlTimeline(None, None, None).to_frame().empty

    def test_states_are_immutable_snapshots(self):
        tl = _tl([_msg(10, flag="YELLOW", scope="Sector", sector=1), _msg(20, flag="CLEAR", scope="Sector", sector=1)])
        early = tl.state_at(15)
        _ = tl.state_at(25)
        assert early.sector_flags == {1: "YELLOW"}  # later replay steps must not mutate earlier states
        with pytest.raises(AttributeError):
            early.track_status = "5"  # frozen dataclass
