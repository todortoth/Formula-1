"""Label logic of the fine-tuning tasks (numpy only)."""

import numpy as np

from f1strategy.finetune import labels as L


def _car(n=300, stop_frame=200, stop_lap=22, lap_len=10):
    """One synthetic car: lap k lasts ``lap_len`` frames (lap_number = frame // lap_len + 1), one tyre-change stop."""
    f = np.arange(n)
    lap = (f // lap_len + 1).astype(float)
    ps = (f >= stop_frame).astype(float)
    stint = (f >= stop_frame + 3).astype(float)  # the new stint starts shortly after the pit entry
    cmp = np.where(f >= stop_frame + 3, 2, 0).astype(np.int8)  # soft -> hard
    in_pit = ((f >= stop_frame) & (f < stop_frame + 5)).astype(float)
    return f, lap, ps, stint, cmp, in_pit


def test_event_found_with_compound_and_in_lap():
    f, lap, ps, stint, cmp, _ = _car()
    ev = L.tyre_change_events(ps, stint, cmp, lap)
    assert ev == [(200, 21, 2)]  # entry frame, lap number of the in-lap, compound received


def test_penalty_entry_is_not_a_tyre_change():
    f, lap, ps, stint, cmp, _ = _car()
    ev = L.tyre_change_events(ps, np.zeros_like(stint), cmp, lap)  # pit_stops rises but stint never does
    assert ev == []


def test_laps_to_pit_counts_lap_ends_like_a_strategist():
    # user's example: end of lap 10 and the stop in lap 22 -> 12 laps; at the end of lap 17 -> 5 laps
    n, lap_len = 400, 10
    f = np.arange(n)
    lap = (f // lap_len + 1).astype(float)
    ev = [(21 * lap_len + 5, 22, 1)]  # entry during lap 22
    laps, cmp, nomore, valid = L.car_labels(lap, np.zeros(n), np.zeros(n), ev, retired=False)
    f10 = 10 * lap_len  # first frame of lap 11 = lap 10 just completed
    assert laps[f10] == 12 and laps[17 * lap_len] == 5 and laps[21 * lap_len] == 1  # in lap 22: stop at this lap's end
    assert cmp[f10] == 1 and not nomore[f10] and valid[f10]


def test_after_the_last_stop_there_is_no_more_and_the_pit_lane_is_invalid():
    f, lap, ps, stint, cmp, in_pit = _car()
    ev = L.tyre_change_events(ps, stint, cmp, lap)
    laps, nxt, nomore, valid = L.car_labels(lap, in_pit, np.zeros_like(lap), ev, retired=False)
    assert np.isfinite(laps[199]) and np.isnan(laps[210]) and nomore[210] and nxt[210] == -1
    assert not valid[200:205].any()  # in the pit lane
    assert valid[210]


def test_retiring_car_without_next_stop_is_not_a_label():
    f, lap, ps, stint, cmp, in_pit = _car()
    ev = L.tyre_change_events(ps, stint, cmp, lap)
    _, _, nomore, valid = L.car_labels(lap, in_pit, np.zeros_like(lap), ev, retired=True)
    assert nomore[250] and not valid[250]  # "no more stops" would be an artefact of the retirement
    assert valid[50]  # before the stop the label is still fine


def test_finished_cars_are_masked():
    f, lap, ps, stint, cmp, in_pit = _car()
    fin = (f > 280).astype(float)
    _, _, _, valid = L.car_labels(lap, in_pit, fin, [], retired=False)
    assert valid[100] and not valid[290]


def test_ffill_and_compound_index():
    a = np.array([np.nan, 1.0, np.nan, np.nan, 2.0])
    out = L._ffill(a)
    assert np.isnan(out[0]) and list(out[1:]) == [1.0, 1.0, 1.0, 2.0]
    one = lambda k, n=3: np.eye(5)[k][None].repeat(n, 0)  # noqa: E731
    oh = one(2)
    c = L.compound_index(oh[:, 0], oh[:, 1], oh[:, 2], oh[:, 3], oh[:, 4])
    assert (c == 2).all()
    assert (L.compound_index(*[np.zeros(3)] * 5) == -1).all()
