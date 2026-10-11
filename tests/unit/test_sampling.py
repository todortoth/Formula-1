"""Time grids and lap windows: choosing when to sample the race."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1strategy import time_grid


class TestTimeGrid:
    def test_regular_and_inclusive_when_aligned(self):
        g = time_grid(10.0, 14.0, 1.0)
        assert g.tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]

    def test_end_is_excluded_when_not_on_the_grid(self):
        g = time_grid(0.0, 2.5, 1.0)
        assert g.tolist() == [0.0, 1.0, 2.0]

    def test_fractional_steps_do_not_lose_the_last_frame_to_rounding(self):
        g = time_grid(0.0, 1.0, 0.1)  # 0.1 is not exactly representable; 11 frames expected
        assert len(g) == 11 and g[-1] == pytest.approx(1.0)

    def test_single_frame(self):
        assert time_grid(5.0, 5.0, 1.0).tolist() == [5.0]

    @pytest.mark.parametrize("step", [0.0, -1.0, float("nan")])
    def test_invalid_step(self, step):
        with pytest.raises(ValueError, match="step_s"):
            time_grid(0.0, 1.0, step)

    def test_end_before_start(self):
        with pytest.raises(ValueError, match="before"):
            time_grid(5.0, 4.0)

    def test_strictly_increasing_with_constant_step(self):
        g = time_grid(123.4, 523.4, 0.25)
        assert np.allclose(np.diff(g), 0.25) and g[0] == 123.4


class TestLapWindow:
    def test_window_spans_the_leaders_laps(self, analyser, syn):
        laps = syn.session.laps
        ver = laps[laps.Driver == "VER"].set_index("LapNumber")
        start, end = analyser.graph_builder.lap_window(2, 3)
        assert start == pytest.approx(ver.loc[2, "LapStartTime"].total_seconds())
        assert end == pytest.approx(ver.loc[3, "Time"].total_seconds())

    def test_default_is_a_single_lap(self, analyser):
        s1, e1 = analyser.graph_builder.lap_window(4)
        s2, e2 = analyser.graph_builder.lap_window(4, 4)
        assert (s1, e1) == (s2, e2) and e1 - s1 == pytest.approx(57.6, abs=1.5)

    def test_consecutive_laps_are_contiguous(self, analyser):
        _, end3 = analyser.graph_builder.lap_window(3)
        start4, _ = analyser.graph_builder.lap_window(4)
        assert end3 == pytest.approx(start4, abs=0.01)

    def test_last_before_first_and_unknown_laps(self, analyser):
        with pytest.raises(ValueError, match="before"):
            analyser.graph_builder.lap_window(5, 3)
        with pytest.raises(ValueError, match="No data for lap"):
            analyser.graph_builder.lap_window(1, 99)

    def test_missing_end_time_falls_back_to_sector_3_then_fails(self, analyser, syn):
        from f1strategy import RaceGraphBuilder
        from tests.helpers import clone_session

        laps = syn.session.laps.copy()
        laps.loc[laps.LapNumber == 3, "Time"] = pd.NaT
        b = RaceGraphBuilder(clone_session(syn.session, laps=laps), analyser.track, analyser.telemetry,
                             analyser.timing, analyser.timeline, analyser._driver_map)  # fmt: skip
        assert b.lap_window(3)[1] == pytest.approx(analyser.graph_builder.lap_window(3)[1], abs=0.01)
        laps.loc[laps.LapNumber == 3, "Sector3SessionTime"] = pd.NaT
        with pytest.raises(ValueError, match="no start / end timestamp"):
            b.lap_window(3)
