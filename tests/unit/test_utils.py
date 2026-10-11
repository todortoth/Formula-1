"""f1strategy.utils: unit conversion, position cleaning, circular runs, time formatting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from f1strategy.utils import (
    POS_UNIT_TO_M,
    circular_runs,
    format_time_mmssms,
    pos_arrays,
    pos_frame,
    run_length,
    td_seconds,
    wrap_angle,
)


def _pos(times, x, y, z, status=None):
    df = pd.DataFrame({"SessionTime": pd.to_timedelta(times, unit="s"), "X": x, "Y": y, "Z": z})
    if status is not None:
        df["Status"] = status
    return df


class TestTdSeconds:
    def test_converts_and_keeps_nat_as_nan(self):
        out = td_seconds(pd.Series([pd.to_timedelta(1.5, unit="s"), pd.NaT, pd.to_timedelta(120, unit="s")]))
        assert out[0] == 1.5 and np.isnan(out[1]) and out[2] == 120.0

    def test_empty(self):
        assert td_seconds(pd.Series([], dtype="timedelta64[ns]")).shape == (0,)


class TestWrapAngle:
    @given(st.floats(min_value=-1e5, max_value=1e5, allow_nan=False))
    def test_range_and_equivalence(self, a):
        w = float(wrap_angle(a))
        assert -np.pi <= w < np.pi
        assert np.sin(w) == pytest.approx(np.sin(a), abs=1e-6)
        assert np.cos(w) == pytest.approx(np.cos(a), abs=1e-6)

    def test_vectorised(self):
        out = wrap_angle(np.array([0.0, 2 * np.pi, -3 * np.pi / 2, np.pi]))
        assert out[0] == 0 and abs(out[1]) < 1e-12 and out[2] == pytest.approx(np.pi / 2) and out[3] == -np.pi


class TestPosFrame:
    def test_units_are_tenths_of_a_metre(self):
        f = pos_frame(_pos([1, 2], [100, 200], [50, 60], [10, 20], ["OnTrack", "OnTrack"]))
        assert POS_UNIT_TO_M == 0.1
        assert list(f["x"]) == [10.0, 20.0] and list(f["y"]) == [5.0, 6.0] and list(f["z"]) == [1.0, 2.0]

    def test_all_zero_samples_are_dropped_but_partial_zeros_kept(self):
        f = pos_frame(_pos([1, 2, 3], [0, 0, 5], [0, 7, 0], [0, 0, 0]))
        assert list(f["t"]) == [2.0, 3.0]  # only (0,0,0) is the 'no signal' artefact

    def test_sorted_and_deduplicated(self):
        f = pos_frame(_pos([3, 1, 2, 2], [30, 10, 20, 21], [1, 1, 1, 1], [1, 1, 1, 1]))
        assert list(f["t"]) == [1.0, 2.0, 3.0]
        assert f["x"][0] == 1.0 and f["x"][2] == 3.0

    def test_nat_times_dropped(self):
        df = _pos([1, 2], [10, 20], [1, 1], [1, 1])
        df.loc[0, "SessionTime"] = pd.NaT
        assert list(pos_frame(df)["t"]) == [2.0]

    def test_on_track_flag(self):
        f = pos_frame(_pos([1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], ["OnTrack", "OffTrack", "OnTrack"]))
        assert list(f["on"]) == [True, False, True]

    def test_missing_status_column_means_on_track(self):
        assert pos_frame(_pos([1, 2], [1, 2], [1, 2], [1, 2]))["on"].all()

    def test_on_flag_stays_aligned_after_filtering_and_sorting(self):
        # (t=1) is a zero artefact and is dropped; remaining rows are unsorted
        f = pos_frame(_pos([1, 4, 2], [0, 40, 20], [0, 4, 2], [0, 4, 2], ["OffTrack", "OffTrack", "OnTrack"]))
        assert list(f["t"]) == [2.0, 4.0] and list(f["on"]) == [True, False]

    def test_pos_arrays_matches_frame(self):
        df = _pos([1, 2], [10, 20], [1, 2], [3, 4])
        t, x, y, z = pos_arrays(df)
        f = pos_frame(df)
        assert all(np.array_equal(a, f[k]) for a, k in ((t, "t"), (x, "x"), (y, "y"), (z, "z")))


class TestCircularRuns:
    def test_wrapping_run_and_lengths(self):
        m = np.zeros(10, bool)
        m[[8, 9, 0, 1]] = True
        m[5] = True
        runs = circular_runs(m)
        assert sorted(runs) == [(5, 5), (8, 1)]
        assert sorted(run_length(r, 10) for r in runs) == [1, 4]

    def test_all_true_and_all_false(self):
        assert circular_runs(np.ones(6, bool)) == [(0, 5)]
        assert circular_runs(np.zeros(6, bool)) == []

    def test_single_element_array(self):
        assert circular_runs(np.array([True])) == [(0, 0)]

    @given(st.lists(st.booleans(), min_size=1, max_size=60).map(np.array))
    def test_runs_partition_exactly_the_true_elements(self, mask):
        n = len(mask)
        covered = np.zeros(n, bool)
        runs = circular_runs(mask)
        for a, b in runs:
            for k in range(run_length((a, b), n)):
                idx = (a + k) % n
                assert not covered[idx]  # runs never overlap
                covered[idx] = True
            if not mask.all():
                assert not mask[(b + 1) % n]  # maximal: next element is False
                assert not mask[(a - 1) % n]
        assert np.array_equal(covered, mask)


class TestFormatTime:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (83.456, "1:23:456"),
            (0.0, "0:00:000"),
            (59.999, "0:59:999"),
            (59.9996, "1:00:000"),  # rounding must carry, never print "0:59:1000"
            (119.9999, "2:00:000"),
            (pd.to_timedelta(90, unit="s"), "1:30:000"),
            (np.timedelta64(61, "s"), "1:01:000"),
        ],
    )
    def test_values(self, value, expected):
        assert format_time_mmssms(value) == expected

    @pytest.mark.parametrize("value", [float("nan"), pd.NaT, None, np.nan])
    def test_missing(self, value):
        assert format_time_mmssms(value) == "N/A"

    @given(st.floats(min_value=0, max_value=7200, allow_nan=False))
    def test_always_well_formed(self, s):
        m, sec, ms = format_time_mmssms(s).split(":")
        assert 0 <= int(sec) < 60 and 0 <= int(ms) < 1000 and int(m) >= 0
