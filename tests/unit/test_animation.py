"""RaceAnimator: frames are drawn from the sequence arrays; the GIF is valid, shared-palette and deterministic."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

from f1strategy import RaceAnimator, make_gif
from f1strategy.plotting import FLAG_COLORS


@pytest.fixture(scope="module")
def anim(sequence, analyser):
    return RaceAnimator(sequence, colors=analyser.driver_colors(), title="Synthetic GP")


def _idx(sequence, t):
    return int(np.argmin(abs(sequence.times - t)))


def _count_color(img, hex_color, tol=12):
    rgb = np.array([int(hex_color[k : k + 2], 16) for k in (1, 3, 5)])
    return int((np.abs(img.astype(int) - rgb).max(axis=2) <= tol).sum())


class TestRendering:
    def test_frame_is_an_rgb_uint8_image_of_the_configured_size(self, anim):
        img = anim.render(10)
        assert img.dtype == np.uint8 and img.shape == (450, 720, 3)

    def test_rendering_is_deterministic(self, anim):
        assert np.array_equal(anim.render(30), anim.render(30))

    def test_frames_differ_over_time(self, anim):
        assert not np.array_equal(anim.render(10), anim.render(40))

    def test_rendering_order_does_not_matter(self, anim):
        a = anim.render(20)
        anim.render(200)
        anim.render(5)
        assert np.array_equal(a, anim.render(20))  # no state leaks between frames (trail, labels, colours)

    def test_flag_state_is_visible_in_the_pixels(self, anim, sequence):
        green, yellow, sc = (anim.render(_idx(sequence, t)) for t in (60.0, 121.0, 190.0))
        assert _count_color(sc, FLAG_COLORS["SC"]) > 1000  # the whole track turns SC-coloured
        assert _count_color(green, FLAG_COLORS["SC"]) < 50
        assert (
            100 < _count_color(yellow, FLAG_COLORS["YELLOW"])
            and _count_color(yellow, FLAG_COLORS["DOUBLE YELLOW"]) > 100
        )
        assert _count_color(green, FLAG_COLORS["YELLOW"]) < 50

    def test_vsc_and_vsc_ending_have_their_own_colours(self, anim, sequence):
        assert _count_color(anim.render(_idx(sequence, 340.0)), FLAG_COLORS["VSC"]) > 1000
        assert _count_color(anim.render(_idx(sequence, 356.0)), FLAG_COLORS["VSC ENDING"]) > 1000

    def test_works_for_first_and_last_frame_and_without_trail(self, sequence):
        a = RaceAnimator(sequence, trail=0)
        assert a.render(0).shape == a.render(sequence.n_frames - 1).shape

    def test_retired_and_pit_cars_do_not_break_the_leaderboard(self, anim, sequence):
        for t in (95.0, 225.0, 300.0, 375.0):  # NOR in pit around 169-190 s, SAI out after ~227 s, flag at the end
            anim.render(_idx(sequence, t))
        texts = [r.get_text() for r in anim._rows]
        assert any("OUT" in x for x in texts)  # SAI retired by then

    def test_leaderboard_lists_cars_in_race_order_with_gaps(self, anim, sequence):
        anim.render(_idx(sequence, 100.0))
        rows = [r.get_text().split() for r in anim._rows if r.get_text()]
        assert [r[0] for r in rows] == ["1", "2", "3", "4", "5", "6"] and rows[0][2] == "LEADER"
        gaps = [float(r[2].strip("+s")) for r in rows[1:]]
        assert gaps == sorted(gaps) and all(g > 0 for g in gaps)

    def test_banner_title_and_drs_text(self, anim, sequence):
        anim.render(_idx(sequence, 190.0))
        assert anim._banner.get_text() == "SAFETY CAR" and anim._drs.get_text() == "DRS disabled"
        assert "Synthetic GP" in anim._title.get_text() and "Lap 3/6" in anim._title.get_text()
        anim.render(_idx(sequence, 121.0))
        assert anim._banner.get_text() == "YELLOW FLAG" and anim._drs.get_text() == "DRS enabled"
        anim.render(_idx(sequence, 60.0))
        assert anim._banner.get_text() == "GREEN FLAG"

    def test_lapped_cars_are_shown_as_laps_down(self, sequence, analyser):
        a = RaceAnimator(sequence)
        i = _idx(sequence, 376.0)  # SAI is >1 lap down by then
        a.render(i)
        assert any(
            t.get_text().strip().split()[1] == "SAI" and "L" in t.get_text().split()[2] for t in a._rows if t.get_text()
        )

    def test_empty_sequences_are_rejected(self, sequence):
        with pytest.raises(ValueError, match="empty"):
            RaceAnimator(sequence.slice(0, 0))

    def test_save_png(self, anim, tmp_path):
        anim.save_png(50, tmp_path / "f.png")
        with Image.open(tmp_path / "f.png") as im:
            assert im.size == (720, 450) and im.format == "PNG"


@pytest.fixture(scope="module")
def gif(tmp_path_factory, sequence, analyser):
    path = tmp_path_factory.mktemp("gif") / "race.gif"
    RaceAnimator(sequence.slice(60, 140), colors=analyser.driver_colors()).save_gif(path, every=4, fps=8)
    return path


class TestGif:
    def test_is_a_looping_animation_with_the_requested_frames_and_timing(self, gif):
        with Image.open(gif) as im:
            assert im.format == "GIF" and im.size == (720, 450) and im.n_frames == 20
            assert (
                im.info["loop"] == 0 and im.info["duration"] == 120
            )  # 1000 / 8 fps rounded to the 10 ms GIF resolution

    def test_static_content_does_not_flicker(self, gif, sequence, analyser):
        """Pixels that are identical in every source frame must decode to identical colours in every GIF frame
        (per-frame adaptive palettes would shift them slightly; the shared palette does not)."""
        anim = RaceAnimator(sequence.slice(60, 140), colors=analyser.driver_colors())
        source = np.stack([anim.render(j) for j in range(0, 80, 4)])
        static = (source == source[0]).all(axis=(0, 3))
        assert static.mean() > 0.5  # most of the picture is static (track, leaderboard frame, background)
        with Image.open(gif) as im:
            decoded = []
            for k in range(im.n_frames):
                im.seek(k)
                decoded.append(np.asarray(im.convert("RGB")).copy())
        decoded = np.stack(decoded)
        assert decoded.shape[0] == 20
        assert (decoded[:, static] == decoded[0][static]).all()

    def test_frames_differ_and_flag_colours_survive_quantisation(self, gif):
        with Image.open(gif) as im:
            first = np.asarray(im.convert("RGB")).copy()
            im.seek(im.n_frames - 1)
            last = np.asarray(im.convert("RGB")).copy()
        assert not np.array_equal(first, last)

    def test_file_size_is_reasonable(self, gif):
        assert 20_000 < gif.stat().st_size < 3_000_000

    def test_output_is_byte_for_byte_deterministic(self, sequence, tmp_path):
        part = sequence.slice(60, 90)
        h = []
        for k in range(2):
            RaceAnimator(part).save_gif(tmp_path / f"{k}.gif", every=3)
            h.append(hashlib.sha256((tmp_path / f"{k}.gif").read_bytes()).hexdigest())
        assert h[0] == h[1]

    def test_every_controls_the_frame_count(self, sequence, tmp_path):
        part = sequence.slice(0, 31)
        for every, expected in ((1, 31), (2, 16), (10, 4), (100, 1)):
            out = RaceAnimator(part).save_gif(tmp_path / f"e{every}.gif", every=every, fps=10)
            with Image.open(out) as im:
                assert im.n_frames == expected, every

    def test_single_frame_gif(self, sequence, tmp_path):
        out = RaceAnimator(sequence.slice(5, 6)).save_gif(tmp_path / "one.gif")
        with Image.open(out) as im:
            assert im.n_frames == 1

    def test_progress_callback(self, sequence, tmp_path):
        calls = []
        RaceAnimator(sequence.slice(0, 6)).save_gif(tmp_path / "p.gif", progress=lambda d, t: calls.append((d, t)))
        assert calls == [(k, 6) for k in range(1, 7)]

    @pytest.mark.parametrize(
        "kwargs", [{"every": 0}, {"fps": 0}, {"fps": -5}, {"palette_colors": 1}, {"palette_colors": 300}]
    )
    def test_invalid_arguments(self, sequence, tmp_path, kwargs):
        with pytest.raises(ValueError):
            RaceAnimator(sequence.slice(0, 5)).save_gif(tmp_path / "x.gif", **kwargs)

    def test_too_many_frames_are_refused_with_a_helpful_message(self, sequence, tmp_path):
        with pytest.raises(ValueError, match=r"max_frames=10.*every=\d+"):
            RaceAnimator(sequence.slice(0, 100)).save_gif(tmp_path / "x.gif", max_frames=10)
        assert not (tmp_path / "x.gif").exists()

    def test_make_gif_shortcut(self, sequence, tmp_path):
        out = make_gif(sequence.slice(0, 10), tmp_path / "m.gif", title="T", every=5, fps=5)
        with Image.open(out) as im:
            assert im.n_frames == 2 and im.info["duration"] == 200
