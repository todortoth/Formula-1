"""End to end: continuous sampling -> dataset file -> training windows / model tensors -> GIF."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from f1strategy import GraphExporter, RaceAnimator, RaceSequence, suggest_pool_factor
from f1strategy.graph.exporter import DRIVER_FEATURES, GLOBAL_FEATURES, TRACK_FEATURES


def test_the_whole_workflow_from_session_to_model_tensors_and_gif(analyser, tmp_path):
    seq = analyser.build_sequence_for_laps(2, 3, step_s=1.0)
    assert seq.continuity_report().ok

    seq.save(tmp_path / "laps.npz")
    loaded = RaceSequence.load(tmp_path / "laps.npz")

    # training windows -> stacked, fixed-shape, finite model inputs (normalised, track pooled to fit BERT)
    factor = suggest_pool_factor(loaded.n_track, loaded.n_drivers, max_tokens=128)
    n_pooled = -(-loaded.n_track // factor)
    window_len = 8
    batches = []
    for w in loaded.windows(window_len, stride=20):
        frames = [f.pool_track(factor) for f in w.frames(normalise=True)]
        batches.append({k: np.stack([getattr(f, k) for f in frames]) for k in ("track_x", "driver_x", "global_x")})
    assert len(batches) >= 5
    for b in batches:
        assert b["track_x"].shape == (window_len, n_pooled, len(TRACK_FEATURES))
        assert b["driver_x"].shape == (window_len, 6, len(DRIVER_FEATURES))
        assert b["global_x"].shape == (window_len, 1, len(GLOBAL_FEATURES))
        assert all(np.isfinite(a).all() for a in b.values())

    out = RaceAnimator(loaded, colors=analyser.driver_colors()).save_gif(tmp_path / "laps.gif", every=5, fps=10)
    with Image.open(out) as im:
        assert im.n_frames == len(range(0, loaded.n_frames, 5))


def test_the_same_numbers_feed_the_model_and_the_animation(analyser, sequence):
    """The animator reads the sequence arrays; a model reads the frames reconstructed from them. Same source."""
    fr = sequence.frame(100)
    anim = RaceAnimator(sequence)
    anim.render(100)
    shown = anim._cars.get_offsets()
    codes = sequence.driver_codes
    for r, code in enumerate(fr.driver_codes):
        x, y = fr.driver_x[r, fr.driver_names.index("x_m")], fr.driver_x[r, fr.driver_names.index("y_m")]
        assert np.allclose(shown[codes.index(code)], [x, y], atol=1e-3)


def test_continuous_and_point_by_point_snapshots_agree_at_the_sector_crossing(analyser):
    """The old workflow (one snapshot at a sector end) is a special case of the new one."""
    G = analyser.build_session_graph(3, "Sector 2")
    ref = GraphExporter().export(G)
    seq = analyser.build_sequence(G.graph["time_s"] - 2.0, G.graph["time_s"] + 2.0, 1.0)
    got = seq.frame(2)
    assert got.time_s == pytest.approx(ref.time_s)
    for k, v in ref.to_dict().items():
        assert np.array_equal(v, got.to_dict()[k], equal_nan=True), k


@pytest.mark.parametrize("step", [0.25, 1.0, 4.0])
def test_other_step_sizes_give_consistent_data(analyser, step):
    seq = analyser.build_sequence(150.0, 190.0, step)
    assert seq.n_frames == int(40 / step) + 1 and seq.continuity_report().ok
    j = seq.driver_names.index("progress_m")
    assert (np.diff(seq.driver_x[:, :, j], axis=0) > -1.5).all()  # everyone keeps moving forward


def test_race_control_changes_are_visible_across_the_continuous_data(sequence):
    names = sequence.global_names
    sc = sequence.global_x[:, names.index("is_sc")]
    on = np.flatnonzero(sc > 0.5)
    assert sequence.times[on[0]] == pytest.approx(180.0, abs=1.0) and sequence.times[on[-1]] == pytest.approx(
        244.0, abs=1.0
    )
    assert (np.diff(on) == 1).all()  # one contiguous safety-car period, no flicker in the signal
    drs = sequence.global_x[:, names.index("drs_enabled")]
    assert (
        np.flatnonzero(np.diff(drs) != 0).size == 3
    )  # on at 100 (before the slice), off at 180, on at 250 -> 3 edges in 40..380 s


def test_pit_stop_and_retirement_are_visible_over_time(sequence):
    names = sequence.driver_names
    nor, sai = sequence.driver_codes.index("NOR"), sequence.driver_codes.index("SAI")
    in_pit = sequence.driver_x[:, nor, names.index("in_pit")]
    stops = sequence.driver_x[:, nor, names.index("pit_stops")]
    assert in_pit.sum() >= 5 and stops[0] == 0 and stops[-1] == 1 and (np.diff(stops) >= 0).all()
    out = sequence.driver_x[:, sai, names.index("finished_or_retired")]
    assert out[0] == 0 and out[-1] == 1 and (np.diff(out) >= 0).all()  # once out, always out
