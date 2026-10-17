"""
bench.py - what can this machine carry?  Measures one training step (masking + forward + backward + optimiser) and
peak memory for every (scale, graph encoder, batch size), and can probe the largest batch size that fits.

    python -m f1strategy.seqbert.bench --cache cache --scales short medium long --encoders flatten sector gnn \
        --batch-sizes 2 4 8 --preset small
    python -m f1strategy.seqbert.bench --cache cache --scales long --encoders sector --probe     # largest batch

Reading the table: ``frames/step`` = batch x tokens is what the graph encoder has to process; the temporal transformer
adds a T x T attention per sample. On CUDA the peak is torch's max allocated memory; on CPU the process RSS.
"""

from __future__ import annotations

import argparse
import resource
import time

import torch

from f1strategy.seqbert import data as D
from f1strategy.seqbert.config import PRESETS, SCALES
from f1strategy.seqbert.losses import build_targets, build_weights, reconstruction_loss
from f1strategy.seqbert.masking import SeqMasker
from f1strategy.seqbert.model import SeqBertConfig, SeqF1Bert


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run(races, scale, encoder, preset, bs, steps, device, amp):
    ix = D.WindowIndex(races, scale)
    if len(ix) == 0:
        return None
    ds = D.WindowDataset(ix)
    raw = torch.utils.data.default_collate([ds[i % len(ds)] for i in range(bs)])
    cfg = SeqBertConfig(graph_encoder=encoder, max_len=max(256, scale.n_tokens), **PRESETS[preset])
    model = SeqF1Bert(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    masker = SeqMasker(device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    times = []
    for i in range(steps + 1):
        t = time.time()
        mb = masker(raw)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(mb, scale.scale_id)
            loss, _ = reconstruction_loss(out["recon"], build_targets(mb), build_weights(mb), mb.task)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i > 0:  # first step = warm-up
            times.append(time.time() - t)
    mem = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else rss_mb()
    return sum(times) / len(times), mem, sum(p.numel() for p in model.parameters()) / 1e6


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--scales", nargs="+", default=list(SCALES))
    p.add_argument("--encoders", nargs="+", default=["flatten", "sector", "gnn"])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4])
    p.add_argument("--preset", default="small", choices=list(PRESETS))
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--probe", action="store_true", help="double the batch size until it no longer fits / 30 s per step")
    a = p.parse_args()
    if a.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(a.device)
    races = D.open_races(a.cache)
    print(f"device {device}, preset {a.preset}\n")
    print("| scale | tokens | encoder | batch | frames/step | params M | s/step | peak MB |")
    print("|---|---|---|---|---|---|---|---|")
    for sc in a.scales:
        scale = SCALES[sc]
        for enc in a.encoders:
            sizes = a.batch_sizes if not a.probe else [2**i for i in range(0, 9)]
            last_ok = None
            for bs in sizes:
                try:
                    r = run(races, scale, enc, a.preset, bs, a.steps, device, a.amp)
                except (torch.cuda.OutOfMemoryError, MemoryError, RuntimeError) as e:
                    print(
                        f"| {sc} | {scale.n_tokens} | {enc} | {bs} | {bs * scale.n_tokens} | - | OOM ({type(e).__name__}) | - |"
                    )
                    break
                if r is None:
                    print(f"| {sc} | - | {enc} | no windows in this cache | | | | |")
                    break
                last_ok = bs
                print(
                    f"| {sc} | {scale.n_tokens} | {enc} | {bs} | {bs * scale.n_tokens} | {r[2]:.2f} | {r[0]:.2f} | {r[1]:.0f} |",
                    flush=True,
                )
                if a.probe and r[0] > 30:
                    break
            if a.probe:
                print(f"\n=> largest working batch for {sc}/{enc}: {last_ok}\n")


if __name__ == "__main__":
    main()
