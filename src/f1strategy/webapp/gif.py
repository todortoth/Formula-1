"""Race replay GIF for the web app: the CLI's continuous mode (1 s sampling, at most 300 frames, 10 fps) with the chosen driver highlighted."""

from pathlib import Path

MAX_FRAMES = 300


def focus_animator(seq, focus, **kw):
    """A RaceAnimator that draws driver `focus` bigger with a black outline and a larger label (animation.py stays untouched)."""
    from f1strategy.animation import RaceAnimator

    class FocusAnimator(RaceAnimator):
        def update(self, i):
            super().update(i)
            codes = list(self.seq.driver_codes)
            if focus in codes:
                k, n = codes.index(focus), len(codes)
                self._cars.set_sizes([230 if s == k else 62 for s in range(n)])
                self._cars.set_edgecolors(["black" if s == k else "white" for s in range(n)])
                self._cars.set_linewidths([2.2 if s == k else 0.8 for s in range(n)])
                self._labels[k].set_fontsize(11)
                self._labels[k].set_zorder(9)

    return FocusAnimator(seq, **kw)


def make_gif(analyser, first_lap, last_lap, focus, title, out) -> Path:
    seq = analyser.build_sequence_for_laps(first_lap, last_lap, step_s=1.0)
    seq = seq.decimate(max(1, -(-seq.n_frames // MAX_FRAMES)))
    return focus_animator(seq, focus, colors=analyser.driver_colors(), title=title).save_gif(
        str(out), every=1, fps=10.0
    )
