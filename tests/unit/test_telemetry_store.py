"""TelemetryStore: interpolation, projection, progress and gaps against ground truth."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from f1strategy.telemetry.store import TelemetryStore
from f1strategy.track.model import TrackModel
from tests.helpers import clone_session, true_gap
from tests.synthetic import session as syn_mod


class TestTelemetry:
    TIMES = [31.0, 45.0, 77.3, 150.0, 183.9, 222.2, 300.0, 345.5]

    def test_position_speed_progress_against_truth(self, analyser, syn):
        tru, tel = syn.truth, analyser.telemetry
        for code in ("VER", "HAM", "ALO"):
            for t in self.TIMES:
                smp = tel.sample(code, t)
                S = float(tru["S"](code, t))
                assert abs(smp.x_m - float(tru["per"](tru["X"], S))) < 0.6
                assert abs(smp.y_m - float(tru["per"](tru["Y"], S))) < 0.6
                assert abs(smp.progress_m - S) < 1.5, (code, t)
                v_true = float(tru["per"](tru["v"], S)) * {d[0]: d[2] for d in syn_mod.DRIVERS}[code] * 3.6
                assert abs(smp.speed_kmh - v_true) < 3.0
                assert abs(smp.lateral_offset_m) < 1.0

    def test_heading_follows_track_while_moving(self, analyser):
        for t in self.TIMES:
            smp = analyser.telemetry.sample("NOR", t)
            assert smp.heading_valid
            assert abs(np.degrees(smp.heading_rel_track_rad)) < 3.0

    def test_stationary_car_has_no_velocity_heading_but_falls_back_to_track_tangent(self, analyser, syn):
        smp = analyser.telemetry.sample("SAI", syn.truth["t_end"]["SAI"] + 30)  # retired car
        assert smp.speed_kmh == 0 and not smp.heading_valid
        assert smp.heading_rad == pytest.approx(analyser.track.heading[smp.track_node])
        assert smp.progress_m == pytest.approx(syn.truth["retire_s"], abs=2.0)
        grid = analyser.telemetry.sample("VER", 10.0)  # on the grid before lights out
        assert not grid.heading_valid and -20 < grid.progress_m < 0

    def test_grid_slots_are_behind_the_line_in_order(self, analyser):
        prog = [analyser.telemetry.sample(d[0], 10.0).progress_m for d in syn_mod.DRIVERS]
        assert prog == sorted(prog, reverse=True) and all(-60 < p < 0 for p in prog)

    def test_zero_signal_samples_are_dropped(self, analyser):
        for code in analyser.telemetry.codes:
            p = analyser.telemetry._pos[code]
            assert not np.any((p["x"] == 0) & (p["y"] == 0))

    def test_time_gaps_match_reference(self, analyser, syn):
        tel = analyser.telemetry
        for t in (100.0, 184.3, 250.0):
            for behind, ahead in (("NOR", "VER"), ("HAM", "LEC"), ("SAI", "VER")):
                if t > 190 and "SAI" in (behind, ahead):
                    continue
                assert tel.gap_seconds(behind, ahead, t) == pytest.approx(
                    true_gap(syn.truth, behind, ahead, t), abs=0.03
                )

    def test_gap_sign_and_nan(self, analyser, syn):
        tel = analyser.telemetry
        assert tel.gap_seconds("VER", "NOR", 184.0) < 0  # VER is really ahead of NOR -> negative "behind" gap
        late = syn.truth["t_end"]["SAI"] + 30  # retired SAI never reached VER's current progress
        assert np.isnan(tel.gap_seconds("VER", "SAI", late))

    def test_drs_channel_only_active_when_enabled_and_in_zone(self, analyser, syn):
        tel, tr = analyser.telemetry, analyser.track
        opened, outside = 0, 0
        for t in np.arange(40.0, 370.0, 0.7):
            smp = tel.sample("VER", t)
            if smp.drs_raw in (10.0, 12.0, 14.0):
                opened += 1
                assert 100 <= t < 180 or t >= 250  # never while DRS is disabled (pre-lap-2, SC)
                assert tr.drs_zone[smp.track_node] > 0 or any(
                    abs(tr.circular_delta(tr.s[smp.track_node], tr.s[k])) < 30 for k in np.flatnonzero(tr.drs_zone)
                )
            elif tr.drs_zone[smp.track_node] > 0 and (100 <= t < 180 or t >= 250) and smp.speed_kmh > 0:
                outside += 1  # inside a zone while enabled but not open
        assert opened > 20 and outside < opened * 0.2


def _driver_map():
    return {d[1]: d[0] for d in syn_mod.DRIVERS}


class TestTelemetryEdgeCases:
    def test_driver_without_position_data_is_skipped(self, syn, track):
        sess = clone_session(syn.session, pos_data={k: v for k, v in syn.session.pos_data.items() if k != "44"})
        store = TelemetryStore(sess, track, _driver_map())
        assert "HAM" not in store.codes and len(store.codes) == 5

    def test_driver_with_a_single_sample_is_skipped(self, syn, track):
        pos = dict(syn.session.pos_data)
        pos["44"] = pos["44"].iloc[:3]  # first two rows are the zero artefacts -> 1 usable sample
        store = TelemetryStore(clone_session(syn.session, pos_data=pos), track, _driver_map())
        assert "HAM" not in store.codes

    def test_no_telemetry_at_all(self, syn, track):
        store = TelemetryStore(clone_session(syn.session, pos_data={}, car_data={}), track, _driver_map())
        assert store.codes == []

    def test_missing_car_data_falls_back_to_position_speed(self, syn, track):
        store = TelemetryStore(clone_session(syn.session, car_data={}), track, _driver_map())
        smp = store.sample("VER", 150.0)
        reference = TelemetryStore(syn.session, track, _driver_map()).sample("VER", 150.0)
        assert smp.speed_kmh == pytest.approx(reference.speed_kmh, abs=4.0)
        assert all(np.isnan(getattr(smp, f)) for f in ("gear", "rpm", "throttle", "brake", "drs_raw"))

    @pytest.mark.parametrize("t", [-100.0, 0.0, 1e6])
    def test_sampling_outside_the_recorded_range_clamps_instead_of_failing(self, analyser, t):
        smp = analyser.telemetry.sample("NOR", t)
        assert np.isfinite([smp.x_m, smp.y_m, smp.progress_m, smp.track_s_m]).all() and smp.track_node >= 0

    def test_progress_never_runs_backwards_for_a_normal_driver(self, analyser):
        _ = analyser.telemetry.progress_at("VER", 100.0)  # force the lazy computation
        prog = analyser.telemetry._progress["VER"]["progress"]
        assert np.diff(prog).min() > -1.5  # only position noise, no wrap glitches

    def test_time_at_progress_inverts_progress_at(self, analyser):
        tel = analyser.telemetry
        for t in (40.0, 90.0, 200.0, 330.0):
            p = tel.progress_at("LEC", t)
            assert tel.progress_at("LEC", tel.time_at_progress("LEC", p)) == pytest.approx(p, abs=1.0)
            assert tel.time_at_progress("LEC", p) == pytest.approx(t, abs=0.1)

    def test_time_at_progress_is_nan_outside_the_driven_range(self, analyser):
        assert np.isnan(analyser.telemetry.time_at_progress("VER", -5000.0))
        assert np.isnan(analyser.telemetry.time_at_progress("VER", 1e9))

    def test_gap_to_self_is_zero(self, analyser):
        assert analyser.telemetry.gap_seconds("VER", "VER", 150.0) == pytest.approx(0.0, abs=0.01)

    def test_gap_is_additive(self, analyser):
        tel = analyser.telemetry
        t = 150.0
        total = tel.gap_seconds("ALO", "VER", t)
        assert total == pytest.approx(tel.gap_seconds("ALO", "HAM", t) + tel.gap_seconds("HAM", "VER", t), abs=0.05)

    @settings(max_examples=40, deadline=None)
    @given(t=st.floats(0.0, 420.0), code=st.sampled_from(["VER", "NOR", "LEC", "HAM", "ALO", "SAI"]))
    def test_samples_are_always_finite_and_on_the_track(self, analyser, t, code):
        s = analyser.telemetry.sample(code, t)
        assert np.isfinite([s.x_m, s.y_m, s.z_m, s.speed_kmh, s.progress_m, s.track_s_m]).all()
        assert 0 <= s.track_node < analyser.track.n and 0 <= s.track_s_m < analyser.track.length_m
        assert abs(s.lateral_offset_m) < 3.0 and -np.pi <= s.heading_rad <= np.pi

    def test_samples_are_deterministic(self, analyser):
        assert analyser.telemetry.sample("HAM", 123.4) == analyser.telemetry.sample("HAM", 123.4)


class TestRaceStartDetection:
    def _bare(self, start, laps_start):
        laps = pd.DataFrame({"LapStartTime": pd.to_timedelta(laps_start, unit="s")})
        return SimpleNamespace(session_start_time=start, laps=laps)

    def test_uses_session_start_time(self):
        assert TelemetryStore._guess_race_start(self._bare(pd.to_timedelta(77.5, unit="s"), [10.0])) == 77.5

    def test_falls_back_to_first_lap_start_if_nat_or_missing(self):
        assert TelemetryStore._guess_race_start(self._bare(pd.NaT, [30.0, 90.0])) == 30.0
        sess = SimpleNamespace(laps=pd.DataFrame({"LapStartTime": pd.to_timedelta([42.0, 99.0], unit="s")}))
        assert TelemetryStore._guess_race_start(sess) == 42.0

    def test_zero_if_nothing_is_known(self):
        sess = SimpleNamespace(laps=pd.DataFrame({"LapStartTime": pd.to_timedelta([None], unit="s")}))
        assert TelemetryStore._guess_race_start(sess) == 0.0


class TestProgressAnchoring:
    """Progress must be 0 at the start/finish line at lights out, however the car got to the grid.
    Hand-made data on an exact circle, so the right answer is known analytically."""

    R = 300.0
    START = 100.0

    @pytest.fixture()
    def circle(self):
        th = np.linspace(0, 2 * np.pi, 2000)
        return TrackModel.from_polyline(self.R * np.cos(th), self.R * np.sin(th), np.zeros_like(th))

    def _store(self, circle, dist_of_t, race_start=None, times=None):
        t = times if times is not None else np.arange(0.0, 300.0, 0.2)
        S = dist_of_t(t)
        th = S / self.R  # the circle starts at (R, 0) and runs counter-clockwise like the track model
        pos = pd.DataFrame({
            "SessionTime": pd.to_timedelta(t, unit="s"),
            "X": np.round(self.R * np.cos(th) * 10), "Y": np.round(self.R * np.sin(th) * 10),
            "Z": np.full(len(t), 5.0), "Status": "OnTrack",
        })  # fmt: skip
        sess = SimpleNamespace(pos_data={"1": pos}, car_data={}, session_start_time=pd.to_timedelta(self.START, unit="s"),
                               laps=pd.DataFrame({"LapStartTime": pd.to_timedelta([self.START], unit="s")}))  # fmt: skip
        return TelemetryStore(sess, circle, {"1": "AAA"}, race_start_s=race_start)

    def test_formation_lap_before_the_race_does_not_offset_progress(self, circle):
        L = circle.length_m

        def dist(t):  # 0..50 s: 1.7 laps of formation lap ending 10 m before the line, then stand, then 20 m/s
            formation = np.minimum(t / 50.0, 1.0) * (2 * L - 10.0 - 0.3 * L) + 0.3 * L
            race = np.maximum(t - self.START, 0.0) * 20.0
            return np.where(t < self.START, formation, 2 * L - 10.0 + race)

        store = self._store(circle, dist)
        assert store.progress_at("AAA", self.START) == pytest.approx(-10.0, abs=2.5)
        assert store.progress_at("AAA", self.START + 25.0) == pytest.approx(500.0 - 10.0, abs=4.0)
        assert store.progress_at("AAA", self.START + 150.0) == pytest.approx(3000.0 - 10.0, abs=4.0)  # >1 lap on

    def test_regression_pole_sitter_crossing_the_line_between_two_samples(self, circle):
        """Old bug: interpolating the *wrapped* track position at lights out averaged L-0.5 and +1.0 into ~L/2."""
        L = circle.length_m

        def dist(t):
            return np.where(t < self.START, L - 0.5, L - 0.5 + (t - self.START) * 20.0)

        # lights out falls between the samples at 100.0 s (still on the grid) and 100.2 s (already past the line)
        store = self._store(circle, dist, race_start=self.START + 0.1)
        assert store.progress_at("AAA", self.START + 30.0) == pytest.approx(600.0 - 0.5, abs=4.0)
        assert store.progress_at("AAA", 10.0) == pytest.approx(-0.5, abs=2.0)

    def test_car_that_already_crossed_the_line_at_race_start_gets_positive_progress(self, circle):
        def dist(t):
            return np.where(t < self.START, 40.0, 40.0 + (t - self.START) * 20.0)

        store = self._store(circle, dist)
        assert store.progress_at("AAA", self.START) == pytest.approx(40.0, abs=2.5)

    def test_wrapping_the_line_many_times_stays_continuous(self, circle):
        L = circle.length_m
        store = self._store(circle, lambda t: np.maximum(t - self.START, 0.0) * 60.0 + (L - 5.0))
        prog = store.progress_at("AAA", self.START + 99.0)
        assert prog == pytest.approx(99.0 * 60.0 - 5.0, abs=5.0)  # > 3 laps, no lap lost at any crossing
