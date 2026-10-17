"""
visualize_training.py - figures that explain the sequence model (architecture, graph, sequence, scales, masks), generated
from the real cache so that they can be put into a thesis chapter or sent to a supervisor.

    python -m f1strategy.seqbert.visualize --cache cache --out figures

Writes fig1_architecture.png, fig2_graph_to_vector.png, fig3_sequence.png, fig4_scales.png, fig5_task_masks.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from f1strategy.pretrain import schema as S  # noqa: E402
from f1strategy.seqbert import data as D  # noqa: E402
from f1strategy.seqbert.config import SCALES  # noqa: E402
from f1strategy.seqbert.masking import TASKS, SeqMasker  # noqa: E402

CMAP = plt.get_cmap("tab20")
RED, BLUE, GREY = "#d62728", "#1f77b4", "#9a9a9a"


def _window(races, scale_name, frac=0.35, offset=0):
    ix = D.WindowIndex(races, SCALES[scale_name])
    ds = D.WindowDataset(ix)
    k = int(len(ds) * frac) + offset
    return ix, ds, k, torch.utils.data.default_collate([ds[k]])


def _xy(raw, t, r):
    """car positions of frame t in the normalised track frame + mask of present cars."""
    drv = raw["drv"][0, t].numpy()
    pres = raw["present"][0, t].numpy()
    return drv[:, S.DI["x_m"]], drv[:, S.DI["y_m"]], pres & np.isfinite(drv[:, S.DI["x_m"]])


def _draw_track(ax, track, lw=5):
    x, y = track[:, S.TI["x_m"]], track[:, S.TI["y_m"]]
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], color="#cfcfcf", lw=lw, solid_capstyle="round", zorder=1)
    ax.set_aspect("equal")
    ax.axis("off")


def _frame(ax, raw, t, track, ghost=None, edges=True, title=None, s=26):
    _draw_track(ax, track)
    x, y, pres = _xy(raw, t, None)
    if edges:
        ee = raw["eexists"][0, t].numpy()
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                if ee[i, j] and pres[i] and pres[j]:
                    ax.plot([x[i], x[j]], [y[i], y[j]], color="#555", lw=0.8, zorder=2)
    for d in np.where(pres)[0]:
        c = RED if ghost is not None and ghost[d] else CMAP(d % 20)
        ax.scatter(x[d], y[d], s=s * (1.8 if c == RED else 1), color=c, zorder=3, edgecolor="k", linewidth=0.4)
    ax.plot(*track[0, [S.TI["x_m"], S.TI["y_m"]]], marker="s", color="k", ms=4, zorder=4)
    if title:
        ax.set_title(title, fontsize=8)


def _box(ax, xy, wh, text, fc="#f4f4f4", ec="k", fs=8, ls="-", hatch=None):
    p = FancyBboxPatch(
        xy, wh[0], wh[1], boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=1, ls=ls, hatch=hatch
    )
    ax.add_patch(p)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.2))


# -------------------------------------------------------------------------------------------------------------
def fig_architecture(out: Path, races):
    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 0.62)
    ax.axis("off")
    ix, ds, k, raw = _window(races, "short")
    track = raw["trk"][0].numpy()
    # sequence of graphs
    ax.text(0.07, 0.6, "0. Sequence of graphs", ha="center", fontsize=9, weight="bold")
    for i, t in enumerate([0, 7, 13, 19]):
        sub = fig.add_axes([0.012 + 0.036 * i * 1.0, 0.43 - 0.06 * i + 0.15, 0.065, 0.2])
        _frame(sub, raw, t, track, ghost=None, edges=False, s=9)
        sub.set_title(f"t={t}", fontsize=7, pad=0)
    ax.text(0.09, 0.07, "T frames (e.g. 20 x 0.5 s)\none graph per time step", ha="center", fontsize=8)
    _arrow(ax, (0.19, 0.3), (0.26, 0.3))
    # graph encoder
    ax.text(0.355, 0.57, "I. Graph -> vector\n(per frame)", ha="center", fontsize=9, weight="bold", va="center")
    for j, (name, sub) in enumerate(
        [
            ("GNN  (graph attention on cars +\ntrack nodes + proximity edges)", ""),
            ("Flatten  (all visible numbers\n-> one linear projection)", ""),
            ("Sector  (pool the ring arcs:\ncars, flags, mean features)", ""),
        ]
    ):
        _box(ax, (0.265, 0.42 - 0.15 * j), (0.17, 0.115), name, fc="#eaf2fb", fs=7.5)
    _arrow(ax, (0.44, 0.3), (0.5, 0.3))
    # vectors
    ax.text(0.585, 0.6, "T vectors + [CLS]", ha="center", fontsize=9, weight="bold")
    labels = ["[CLS]"] + [f"v{i}" for i in range(1, 4)] + ["..."] + ["v(T)"]
    for i, lab in enumerate(labels):
        masked = lab in ("v2", "...")
        _box(
            ax,
            (0.5, 0.5 - 0.075 * i),
            (0.16, 0.06),
            lab if not masked else lab + "   [-.1 +.1 -.1 ...]" if lab != "..." else "...",
            fc="#fde0dd" if lab == "v2" else ("#dde8fd" if lab == "[CLS]" else "white"),
            fs=7,
            hatch="//" if lab == "v2" else None,
        )
    ax.text(
        0.58,
        0.02,
        "task Z: a block of vectors is replaced by the\ndummy vector (0 or -0.1/+0.1) after the encoder",
        ha="center",
        fontsize=7.5,
        color=RED,
    )
    _arrow(ax, (0.67, 0.3), (0.74, 0.3))
    # transformer
    ax.text(0.83, 0.57, "II. Temporal BERT encoder", ha="center", fontsize=9, weight="bold")
    _box(
        ax,
        (0.745, 0.12),
        (0.17, 0.4),
        "torch nn.TransformerEncoder\n(bidirectional self-attention\nover TIME)\n\n+ learned time positions\n+ scale embedding\n\nL layers, GELU, pre-LN",
        fc="#fff4d6",
        fs=8,
    )
    _arrow(ax, (0.93, 0.3), (0.99, 0.3))
    # heads
    ax.text(1.2, 0.58, "III. Heads on the contextualised vectors", ha="center", fontsize=10, weight="bold")
    _box(
        ax,
        (1.0, 0.38),
        (0.4, 0.13),
        "1-layer linear head per time step ->  flattened frame\n[ drivers | global | track ]   (MSE / cosine on the masked cells)",
        fc="#e8f6e8",
        fs=8,
    )
    ax.text(
        1.2,
        0.335,
        "B ghost car  ·  T tyre set + age  ·  P pit-return window  ·  Z zone (35 %)",
        ha="center",
        fontsize=8,
    )
    _box(
        ax,
        (1.0, 0.17),
        (0.4, 0.1),
        "[CLS] -> projection -> contrastive loss\n(same circuit & race phase = positive, other circuits = negative)",
        fc="#f3e8f6",
        fs=8,
    )
    _arrow(ax, (1.2, 0.3), (1.2, 0.27))
    ax.text(
        0.8,
        0.0,
        "Fine-tuning later reads [CLS] (summary of the sequence) or the per-step vectors.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    fig.savefig(out / "fig1_architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_graph_to_vector(out: Path, races):
    ix, ds, k, raw = _window(races, "short", 0.5)
    track = raw["trk"][0].numpy()
    t = 10
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})
    _frame(
        axs[0],
        raw,
        t,
        track,
        title="(a) one race graph\nnodes = cars (colour = car slot), grey ring = track,\nlines = proximity edges (<5 s)",
    )
    # ring tokens
    M = track.shape[0]
    ax = axs[1]
    _frame(
        ax,
        raw,
        t,
        track,
        edges=False,
        title="(b) sector pooling: the ring is cut into 16 arcs;\nper arc: track features, flags, #cars, mean car features",
    )
    sec = np.arange(M) * 16 // M
    for s in range(16):
        idx = np.where(sec == s)[0]
        mx, my = track[idx, S.TI["x_m"]].mean(), track[idx, S.TI["y_m"]].mean()
        ax.text(mx, my, str(s + 1), fontsize=7, color=BLUE, ha="center", va="center", weight="bold", zorder=5)
    # GNN
    ax = axs[2]
    _frame(
        ax,
        raw,
        t,
        track,
        edges=True,
        title="(c) GNN: nodes = global + cars + 32 track nodes;\nedges = ring, car-on-track-node, car-car proximity",
    )
    for j in range(0, M, M // 32):
        ax.scatter(track[j, S.TI["x_m"]], track[j, S.TI["y_m"]], s=10, color="#444", marker="s", zorder=2)
    # flatten
    ax = axs[3]
    drv = raw["drv"][0, t].numpy()
    pres = raw["present"][0, t].numpy()
    mat = np.nan_to_num(drv) * pres[:, None]
    ax.imshow(np.clip(mat, -3, 3), aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_xlabel("driver features (42)")
    ax.set_ylabel("car slot")
    ax.set_title(
        "(d) flatten: this [22 x 42] matrix (+ edges, flags,\nglobal, track) is read as ONE long vector", fontsize=8
    )
    fig.suptitle("Step I - graph -> vector. All three variants return one H-dimensional vector per frame.", fontsize=11)
    fig.savefig(out / "fig2_graph_to_vector.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_sequence(out: Path, races):
    ix, ds, k, raw = _window(races, "short", 0.5)
    track = raw["trk"][0].numpy()
    mk = SeqMasker()
    mb = mk(raw, task=0, generator=torch.Generator().manual_seed(3))
    ghost = mb.slot_masked[0, 0].numpy()
    ts = np.linspace(0, 19, 8).astype(int)
    fig, axs = plt.subplots(2, 8, figsize=(17, 5.2), gridspec_kw={"height_ratios": [3, 1.3]})
    for i, t in enumerate(ts):
        _frame(axs[0, i], raw, t, track, ghost=ghost, edges=True, title=f"t = {t * 0.5:.1f} s", s=22)
    x_t = np.arange(20)
    gcar = int(np.where(ghost)[0][0])
    sp = raw["drv"][0, :, gcar, S.DI["speed_kmh"]].numpy() * 350
    gs = fig.add_subplot(2, 1, 2)
    for a in axs[1]:
        a.remove()
    gs.plot(x_t * 0.5, sp, color=RED, lw=2, label="ghost car speed (hidden in all frames - the target)")
    for d in range(22):
        if d != gcar and raw["present"][0, :, d].all():
            gs.plot(x_t * 0.5, raw["drv"][0, :, d, S.DI["speed_kmh"]].numpy() * 350, color="#ccc", lw=0.8)
    gs.set_xlabel("time in window [s]")
    gs.set_ylabel("speed [km/h]")
    gs.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "A sequence = 20 consecutive graphs (10 s). Task B: one car (red) is a ghost in every frame; the model must reconstruct it from the others, using time context.",
        fontsize=10,
    )
    fig.savefig(out / "fig3_sequence.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_scales(out: Path, races):
    r = races[0]
    a = r.a
    t = np.asarray(a["t"], dtype=float)
    spd = np.nanmean(np.asarray(a["drv"][:, :, S.DI["speed_kmh"]], dtype=float), axis=1) * 350
    lap = r.meta["norm"]["pace_ref_s"]
    fig, axs = plt.subplots(3, 1, figsize=(15, 8.5))
    cfg = [("short", "10 s - 20 tokens"), ("medium", "1 lap - 96 tokens"), ("long", "10 laps - 192 tokens")]
    c0 = int(len(t) * 0.45)
    for ax, (name, lab) in zip(axs, cfg):
        sc = SCALES[name]
        step = sc.step_frames(lap, r.meta["step_s"])
        span = (sc.n_tokens - 1) * step + 1
        lo, hi = c0 - span, c0 + 2 * span
        lo = max(lo, 0)
        xs = (np.arange(lo, min(hi, len(t))) - c0) * r.meta["step_s"]
        ax.plot(xs, spd[lo : min(hi, len(t))], color=GREY, lw=0.9)
        ax.axvspan(0, span * r.meta["step_s"], color="#cfe3f7", alpha=0.7)
        toks = np.arange(sc.n_tokens) * step * r.meta["step_s"]
        ax.vlines(toks, ax.get_ylim()[0], ax.get_ylim()[0] + 8, color=BLUE, lw=0.8)
        ax.set_title(
            f"{name}: {lab}; one token every {step * r.meta['step_s']:.1f} s; window = {span * r.meta['step_s']:.0f} s (median lap here = {lap:.0f} s)",
            fontsize=9,
        )
        ax.set_ylabel("mean field speed [km/h]")
    axs[-1].set_xlabel("seconds relative to window start")
    fig.suptitle(
        "The three sequence scales on the same race (blue band = one window, blue ticks = tokens)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out / "fig4_scales.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_task_masks(out: Path, races):
    ix, ds, k, raw = _window(races, "short", 0.62)
    mk = SeqMasker()
    names = {
        "B": "B  ghost car: one car, all 42 features, all frames",
        "T": "T  tyre set: compound + age + fresh + stint + pit stops of one car",
        "P": "P  pit-return window: cars 18-24 s behind a reference car (moves with time)",
        "Z": "Z  zone: whole graph vectors of the middle 35 % -> dummy vector",
    }
    fig, axs = plt.subplots(2, 2, figsize=(14, 7.5))
    for ax, (k_, tn) in zip(axs.ravel(), enumerate(TASKS)):
        mb = mk(raw, task=k_, generator=torch.Generator().manual_seed(11 + k_))
        pres = mb.present[0].numpy().T  # [D, T]
        m = np.zeros(pres.shape)
        m[pres] = 1
        if tn == "Z":
            m[:, mb.vec_mask[0].numpy()] = np.where(pres[:, mb.vec_mask[0].numpy()], 2, 0)
        else:
            m[mb.slot_masked[0].numpy().T] = 2
        if tn == "P":
            ref = np.where(mb.ref[0].numpy())[0]
            for r_ in ref:
                m[r_, :] = np.where(pres[r_], 3, 0)
        ax.set_title(names[tn], fontsize=9)
        ax.set_xlabel("time step in window")
        ax.set_ylabel("car slot")
    fig.suptitle(
        "What each pre-training task hides (blue = visible car, red = masked, green = visible reference car of task P)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "fig5_task_masks.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    races = D.open_races(a.cache)
    for f in (fig_architecture, fig_graph_to_vector, fig_sequence, fig_scales, fig_task_masks):
        f(out, races)
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
