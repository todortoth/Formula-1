"""TrackModel: geometry, sectors, corners, marshal sectors, DRS zones, projection."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from f1strategy import SessionAnalyser
from f1strategy.track.model import TrackModel, _select_reference_lap
from f1strategy.utils import circular_runs
from tests.synthetic import session as syn_mod
from tests.synthetic.session import make_synthetic_session


class TestTrackModel:
    def test_spacing_is_ten_metres_and_ring_closes(self, analyser, syn):
        tr = analyser.track
        assert abs(tr.length_m - syn.truth["L"]) < 0.005 * syn.truth["L"]
        assert abs(tr.spacing_m - 10.0) < 0.1
        step = np.hypot(np.diff(np.append(tr.x, tr.x[0])), np.diff(np.append(tr.y, tr.y[0])))
        assert step.max() < 10.5 and step.min() > 7.0  # chord of a 10 m arc is never longer, hairpins make it shorter

    def test_nodes_lie_on_the_true_centre_line(self, analyser, syn):
        tr, tru = analyser.track, syn.truth
        d = np.hypot(tr.x[:, None] - tru["X"][None, :], tr.y[:, None] - tru["Y"][None, :]).min(axis=1)
        assert d.max() < 1.0

    def test_heading_and_curvature_match_truth(self, analyser, syn):
        tr, tru = analyser.track, syn.truth
        s_true = np.arange(len(tru["X"]), dtype=float)
        # reference tangent from the true dense curve, evaluated at the node's *own* arc length
        i = np.round(tr.s).astype(int) % len(s_true)
        dx = np.gradient(tru["X"])[i]
        dy = np.gradient(tru["Y"])[i]
        diff = np.degrees(np.abs(np.angle(np.exp(1j * (tr.heading - np.arctan2(dy, dx))))))
        assert np.median(diff) < 1.0 and diff.max() < 5.0
        k_true = tru["kappa"][i]
        assert np.corrcoef(np.abs(tr.curvature), k_true)[0, 1] > 0.9

    def test_curvature_sign_is_left_positive(self, analyser):
        tr = analyser.track
        # the synthetic circuit runs counter-clockwise (theta increases): net turning is +2*pi (left)
        assert abs(np.sum(tr.curvature) * tr.spacing_m - 2 * np.pi) < 0.3

    def test_sectors(self, analyser, syn):
        tr, L = analyser.track, syn.truth["L"]
        assert abs(tr.sector_end_s[0] - 0.33 * L) < 10 and abs(tr.sector_end_s[1] - 0.68 * L) < 10
        assert set(np.unique(tr.sector_idx)) == {0, 1, 2}
        assert tr.sector_idx[0] == 0 and tr.sector_idx[-1] == 2

    def test_drs_zones_inferred(self, analyser, syn):
        tr, L = analyser.track, syn.truth["L"]
        assert tr.drs_zone.max() == 2
        for zone, (a, b) in enumerate(syn_mod.DRS_ZONES_FRAC, start=1):
            nodes = np.flatnonzero(tr.drs_zone == zone)
            assert abs(tr.s[nodes.min()] - a * L) < 40 and abs(tr.s[nodes.max()] - b * L) < 40
        starts = tr.drs_zone_starts_s
        assert len(starts) == 2 and tr.is_drs_end.sum() == 2

    def test_drs_detection_points_from_offset_and_manual(self, analyser, syn):
        tr = analyser.track
        assert tr.is_drs_detection.sum() == 2
        for s_start, s_det in zip(tr.drs_zone_starts_s, sorted(tr.drs_detection_s)):
            assert abs(tr.circular_delta(s_start - 200.0, s_det)) < 1e-6
        # node lies within half a spacing of the exact point
        for s_exact in tr.drs_detection_s:
            k = int(tr.node_at_s(s_exact))
            assert tr.is_drs_detection[k] and abs(tr.circular_delta(tr.s[k], s_exact)) <= tr.spacing_m / 2 + 1e-9
        tr2 = TrackModel.from_polyline(tr.x, tr.y, tr.z)
        tr2.set_drs_detection_points([100.0, 4990.0])
        assert tr2.is_drs_detection.sum() == 2

    def test_marshal_sectors_wrap_over_start_line(self, analyser, syn):
        tr, L = analyser.track, syn.truth["L"]
        assert tr.marshal_sector[0] == 12  # before the first marker -> last sector
        assert tr.marshal_sector[int(tr.node_at_s(0.02 * L + 15))] == 1
        assert tr.marshal_sector[int(tr.node_at_s(0.02 * L + L / 12 + 15))] == 2
        assert set(np.unique(tr.marshal_sector)) == set(range(1, 13))

    def test_corners(self, analyser, syn):
        tr = analyser.track
        found = np.flatnonzero(tr.corner_number > 0)
        assert len(found) == len(syn.truth["corner_s"])
        for node, s_true in zip(found, syn.truth["corner_s"]):
            assert abs(tr.circular_delta(tr.s[node], s_true)) < 15

    def test_projection_round_trip_and_lateral_sign(self, analyser):
        tr = analyser.track
        k = 123
        left = (-np.sin(tr.heading[k]), np.cos(tr.heading[k]))
        p = tr.project([tr.x[k] + 3 * left[0]], [tr.y[k] + 3 * left[1]])
        assert p.node[0] == k and abs(p.lateral[0] - 3.0) < 0.2 and abs(p.s[0] - tr.s[k]) < 0.5
        p = tr.project([tr.x[k] - 2 * left[0]], [tr.y[k] - 2 * left[1]])
        assert abs(p.lateral[0] + 2.0) < 0.2
        nan = tr.project([np.nan, tr.x[k]], [0.0, tr.y[k]])
        assert nan.node[0] == -1 and nan.node[1] == k

    def test_drs_zone_that_wraps_over_the_finish_line(self, analyser):
        tr = TrackModel.from_polyline(analyser.track.x, analyser.track.y, analyser.track.z)
        s = np.concatenate([np.linspace(0.93, 0.999, 700), np.linspace(0.0, 0.03, 300)]) * tr.length_m
        xy = np.array([[tr.x[int(tr.node_at_s(v))], tr.y[int(tr.node_at_s(v))]] for v in s])
        assert tr.set_drs_zones_from_points(xy[:, 0], xy[:, 1]) == 1
        assert tr.drs_zone[0] == 1 and tr.drs_zone[-1] == 1 and tr.is_drs_start.sum() == 1

    def testcircular_runs(self):
        m = np.zeros(10, bool)
        m[[8, 9, 0, 1]] = True
        m[5] = True
        assert sorted(circular_runs(m)) == [(5, 5), (8, 1)]
        assert circular_runs(np.zeros(5, bool)) == []


def _circle(radius=500.0, n=3000, clockwise=False, closed=True):
    th = np.linspace(0, 2 * np.pi, n, endpoint=closed)
    if clockwise:
        th = -th
    return radius * np.cos(th), radius * np.sin(th), np.zeros_like(th)


class TestAnalyticGeometry:
    """A circle has an exactly known length and curvature - no synthetic-session machinery involved."""

    def test_length_spacing_and_node_count(self):
        tr = TrackModel.from_polyline(*_circle(500.0))
        assert tr.length_m == pytest.approx(2 * np.pi * 500, rel=1e-3)
        assert tr.n == round(tr.length_m / 10) and abs(tr.spacing_m - 10.0) < 0.05
        assert tr.s[0] == 0 and tr.s[-1] == pytest.approx(tr.length_m - tr.spacing_m)

    @pytest.mark.parametrize("clockwise, sign", [(False, 1.0), (True, -1.0)])
    def test_curvature_is_one_over_radius_and_signed_by_turn_direction(self, clockwise, sign):
        tr = TrackModel.from_polyline(*_circle(500.0, clockwise=clockwise))
        assert np.median(tr.curvature) == pytest.approx(sign / 500.0, rel=0.02)
        assert (np.sign(tr.curvature) == sign).mean() > 0.99

    def test_heading_is_tangent_and_advances_with_s(self):
        tr = TrackModel.from_polyline(*_circle(500.0))
        # counter-clockwise circle starting at (R, 0): tangent is +y, i.e. pi/2, at node 0
        assert tr.heading[0] == pytest.approx(np.pi / 2, abs=0.02)
        quarter = tr.n // 4
        assert tr.heading[quarter] == pytest.approx(np.pi, abs=0.05) or tr.heading[quarter] == pytest.approx(
            -np.pi, abs=0.05
        )

    def test_flat_track_has_zero_gradient_and_slope_is_recovered(self):
        x, y, z = _circle(800.0)
        assert np.abs(TrackModel.from_polyline(x, y, z).gradient).max() < 1e-6
        th = np.linspace(0, 2 * np.pi, 3000)
        z_wavy = 5.0 * np.sin(th)  # amplitude 5 m over 2*pi*800 m -> max slope 5 / 800
        tr = TrackModel.from_polyline(x, y, z_wavy)
        assert np.abs(tr.gradient).max() == pytest.approx(5.0 / 800.0, rel=0.05)

    def test_node_positions_lie_on_the_circle(self):
        tr = TrackModel.from_polyline(*_circle(500.0))
        assert np.abs(np.hypot(tr.x, tr.y) - 500.0).max() < 0.5

    def test_open_polyline_warns_but_still_closes(self, caplog):
        x, y, z = _circle(500.0, closed=False)
        x, y = x[: int(len(x) * 0.97)], y[: int(len(y) * 0.97)]  # leaves a ~95 m gap
        with caplog.at_level(logging.WARNING):
            tr = TrackModel.from_polyline(x, y, z[: len(x)])
        assert "does not close well" in caplog.text
        assert np.hypot(tr.x[0] - tr.x[-1], tr.y[0] - tr.y[-1]) < 20

    def test_tiny_track_has_a_minimum_node_count(self):
        tr = TrackModel.from_polyline(*_circle(5.0, n=50))
        assert tr.n >= 8

    def test_repr(self):
        assert "TrackModel(" in repr(TrackModel.from_polyline(*_circle(500.0))) and "nodes" in repr(
            TrackModel.from_polyline(*_circle(500.0))
        )

    def test_to_frame_has_one_row_per_node_and_all_layers(self):
        tr = TrackModel.from_polyline(*_circle(500.0))
        df = tr.to_frame()
        assert len(df) == tr.n
        assert {"s_m", "x_m", "y_m", "z_m", "curvature", "sector", "corner_number", "marshal_sector", "drs_zone",
                "is_drs_start", "is_drs_end", "is_drs_detection"} <= set(df.columns)  # fmt: skip


class TestLayoutSetters:
    @pytest.fixture()
    def tr(self):
        return TrackModel.from_polyline(*_circle(500.0))

    def test_unsorted_marshal_markers_are_handled(self, tr):
        L = tr.length_m
        tr.set_marshal_markers([3, 1, 2], [0.7 * L, 0.1 * L, 0.4 * L])
        assert tr.marshal_sector[int(tr.node_at_s(0.2 * L))] == 1
        assert tr.marshal_sector[int(tr.node_at_s(0.5 * L))] == 2
        assert tr.marshal_sector[int(tr.node_at_s(0.9 * L))] == 3
        assert tr.marshal_sector[0] == 3  # before the first marker -> wraps to the last sector

    def test_corner_collision_keeps_the_first_and_reset_clears(self, tr):
        tr.set_corners([1, 2], [100.0, 101.0])  # both map to the same 10 m node
        assert tr.corner_number[int(tr.node_at_s(100.0))] == 1 and (tr.corner_number > 0).sum() == 1
        tr.set_corners([7], [500.0])
        assert (tr.corner_number > 0).sum() == 1 and tr.corner_number.max() == 7

    def test_explicit_drs_detection_points_wrap_negative_values(self, tr):
        tr.set_drs_detection_points([-10.0, tr.length_m + 5.0])
        assert tr.is_drs_detection.sum() == 2
        assert all(0 <= s < tr.length_m for s in tr.drs_detection_s)

    def test_sector_boundaries(self, tr):
        tr.set_sector_boundaries(1000.0, 2000.0)
        assert tr.sector_idx[int(tr.node_at_s(500))] == 0 and tr.sector_idx[int(tr.node_at_s(1500))] == 1
        assert tr.sector_idx[int(tr.node_at_s(2500))] == 2

    def test_drs_zones_need_enough_data(self, tr):
        assert tr.set_drs_zones_from_points(np.array([1.0] * 10), np.array([1.0] * 10)) == 0
        assert tr.drs_zone.max() == 0

    def test_drs_zone_reset_on_recompute(self, tr):
        s = np.linspace(0.2, 0.3, 800) * tr.length_m
        k = tr.node_at_s(s)
        assert tr.set_drs_zones_from_points(tr.x[k], tr.y[k]) == 1
        s2 = np.linspace(0.6, 0.7, 800) * tr.length_m
        k2 = tr.node_at_s(s2)
        assert tr.set_drs_zones_from_points(tr.x[k2], tr.y[k2]) == 1
        assert tr.drs_zone[int(tr.node_at_s(0.25 * tr.length_m))] == 0  # old zone is gone

    def test_short_zones_are_dropped_and_small_gaps_merged(self, tr):
        L = tr.length_m
        short = tr.node_at_s(np.linspace(0.1, 0.1 + 50 / L, 400))  # 50 m < 100 m minimum
        assert tr.set_drs_zones_from_points(tr.x[short], tr.y[short]) == 0
        a = tr.node_at_s(np.linspace(0.30, 0.34, 600) * L)
        b = tr.node_at_s(np.linspace(0.34 + 30 / L, 0.40, 600) * L)  # 30 m hole between two bursts
        k = np.concatenate([a, b])
        assert tr.set_drs_zones_from_points(tr.x[k], tr.y[k]) == 1


class TestProjectionProperties:
    @settings(max_examples=60, deadline=None)
    @given(k=st.integers(0, 499), off=st.floats(-3.0, 3.0))
    def test_node_plus_lateral_offset_projects_back(self, track, k, off):
        left = np.array([-np.sin(track.heading[k]), np.cos(track.heading[k])])
        p = track.project([track.x[k] + off * left[0]], [track.y[k] + off * left[1]])
        assert abs(track.circular_delta(track.s[k], p.s[0])) < 1.5
        assert p.lateral[0] == pytest.approx(off, abs=0.4)
        assert abs(int(p.node[0]) - k) in (0, 1, track.n - 1)

    @given(a=st.floats(-1e5, 1e5), b=st.floats(-1e5, 1e5))
    def test_circular_delta_range_periodicity_and_antisymmetry(self, track, a, b):
        L = track.length_m
        d = float(track.circular_delta(a, b))
        assert -L / 2 - 1e-6 <= d <= L / 2 + 1e-6
        assert float(track.circular_delta(a + L, b)) == pytest.approx(d, abs=1e-6) or abs(abs(d) - L / 2) < 1e-6
        back = float(track.circular_delta(b, a))
        assert back == pytest.approx(-d, abs=1e-6) or abs(abs(d) - L / 2) < 1e-6

    def test_shape_is_preserved_and_empty_input_is_fine(self, track):
        p = track.project(np.zeros(0), np.zeros(0))
        assert p.node.shape == p.s.shape == p.lateral.shape == (0,)
        pts = track.project(track.x[:7], track.y[:7])
        assert pts.node.shape == (7,) and (pts.node == np.arange(7)).all()

    def test_node_at_s_wraps(self, track):
        assert int(track.node_at_s(-1.0)) == track.n - 0 or int(track.node_at_s(-1.0)) in (0, track.n - 1)
        assert int(track.node_at_s(track.length_m + 1.0)) == 0


class TestReferenceLapSelection:
    @staticmethod
    def _laps(rows):
        base = {"LapStartTime": pd.to_timedelta(0, unit="s"), "Time": pd.to_timedelta(90, unit="s"),
                "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "IsAccurate": True, "TrackStatus": "1",
                "LapNumber": 5.0, "LapTime": pd.to_timedelta(90, unit="s"), "DriverNumber": "1"}  # fmt: skip
        return pd.DataFrame([{**base, **r} for r in rows])

    @staticmethod
    def _t(seconds):
        return pd.to_timedelta(seconds, unit="s")

    def test_picks_the_fastest_clean_lap(self):
        laps = self._laps(
            [{"LapTime": self._t(91), "DriverNumber": "a"}, {"LapTime": self._t(89), "DriverNumber": "b"}]
        )
        assert _select_reference_lap(laps)["DriverNumber"] == "b"

    def test_pit_laps_and_untimed_laps_never_qualify(self):
        laps = self._laps([
            {"LapTime": self._t(80), "PitInTime": self._t(85), "DriverNumber": "pit_in"},
            {"LapTime": self._t(81), "PitOutTime": self._t(1), "DriverNumber": "pit_out"},
            {"LapTime": pd.NaT, "DriverNumber": "untimed"},
            {"LapTime": self._t(95), "DriverNumber": "clean"},
        ])  # fmt: skip
        assert _select_reference_lap(laps)["DriverNumber"] == "clean"

    def test_prefers_accurate_then_not_lap_one_then_green_flag_but_only_if_available(self):
        laps = self._laps([
            {"LapTime": self._t(80), "IsAccurate": False, "DriverNumber": "inaccurate"},
            {"LapTime": self._t(81), "LapNumber": 1.0, "DriverNumber": "lap1"},
            {"LapTime": self._t(82), "TrackStatus": "14", "DriverNumber": "sc"},
            {"LapTime": self._t(99), "DriverNumber": "good"},
        ])  # fmt: skip
        assert _select_reference_lap(laps)["DriverNumber"] == "good"
        # when nothing satisfies a preference the preference is dropped instead of failing
        only_sc = self._laps([{"LapTime": self._t(82), "TrackStatus": "14", "DriverNumber": "sc"}])
        assert _select_reference_lap(only_sc)["DriverNumber"] == "sc"

    def test_raises_when_no_lap_is_usable(self):
        laps = self._laps([{"PitInTime": self._t(80)}, {"LapTime": pd.NaT}])
        with pytest.raises(ValueError, match="No usable lap"):
            _select_reference_lap(laps)


class TestDegradedSessions:
    """FastF1 data is sometimes incomplete; each missing source must degrade gracefully, not crash."""

    def test_missing_sector_times_fall_back_to_equal_thirds(self, caplog):
        syn = make_synthetic_session(with_sector_times=False)
        with caplog.at_level(logging.WARNING):
            tr = TrackModel.from_session(syn.session)
        assert "equal thirds" in caplog.text
        assert tr.sector_end_s == pytest.approx((tr.length_m / 3, 2 * tr.length_m / 3))

    def test_circuit_info_outage_leaves_corners_and_marshal_sectors_unknown(self, caplog):
        syn = make_synthetic_session(with_circuit_info=False)
        with caplog.at_level(logging.WARNING):
            analyser = SessionAnalyser.from_session(syn.session)
        assert "Circuit info unavailable" in caplog.text
        assert (analyser.track.marshal_sector == -1).all() and (analyser.track.corner_number == 0).all()
        # sector yellow messages cannot be placed on the track, but the global state is still reported
        G = analyser.build_graph_at_time(121.0)
        assert G.graph["global"]["is_yellow"] and G.graph["global"]["n_yellow_sectors"] == 2
        assert not any(d["flag_yellow"] or d["flag_double_yellow"] for _, d in G.nodes(data=True) if d["node_type"] == "track")  # fmt: skip

    def test_circuit_info_returning_none(self, caplog):
        syn = make_synthetic_session()
        syn.session.get_circuit_info = lambda: None
        with caplog.at_level(logging.WARNING):
            tr = TrackModel.from_session(syn.session)
        assert (tr.marshal_sector == -1).all()

    def test_circuit_info_with_empty_tables(self):
        from types import SimpleNamespace

        syn = make_synthetic_session()
        empty = pd.DataFrame({"X": [], "Y": [], "Number": []})
        syn.session.get_circuit_info = lambda: SimpleNamespace(corners=empty, marshal_sectors=None)
        tr = TrackModel.from_session(syn.session)
        assert (tr.marshal_sector == -1).all() and (tr.corner_number == 0).all()

    def test_no_drs_samples_means_no_zones(self):
        tr = TrackModel.from_session(make_synthetic_session(with_drs=False).session, drs_detection_offset_m=100.0)
        assert tr.drs_zone.max() == 0 and not tr.is_drs_start.any()
        assert not tr.is_drs_detection.any()  # nothing to offset from

    def test_explicit_detection_points_win_over_offset(self, syn):
        tr = TrackModel.from_session(syn.session, drs_detection_points_m=[123.0], drs_detection_offset_m=300.0)
        assert tr.is_drs_detection.sum() == 1 and tr.drs_detection_s == [123.0]

    def test_reference_lap_is_recorded(self, analyser):
        meta = analyser.track.meta
        assert meta["reference_lap"] > 1 and meta["reference_lap_time_s"] > 0 and meta["reference_driver"]
