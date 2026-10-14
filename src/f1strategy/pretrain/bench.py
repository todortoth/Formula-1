"""Time one training step on the current device, split into data / mask / forward / backward+optimiser.

python -m f1strategy.pretrain.bench --cache cache --batch-size 64 --amp
python -m f1strategy.pretrain.bench --cache cache --batch-size 256 --amp --dropout 0.0
python -m f1strategy.pretrain.bench --cache cache --batch-size 64 --amp --profile   # + the most expensive ops
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from f1strategy.pretrain.data import PretrainDataset, open_races, split_races
from f1strategy.pretrain.losses import compute_losses
from f1strategy.pretrain.masking import Masker
from f1strategy.pretrain.model import F1Bert, F1BertConfig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--profile", action="store_true", help="also print the most expensive ops of one step")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    races = open_races(a.cache)
    ds = PretrainDataset(races, split_races(races).train)
    t = time.time()
    raw = next(iter(DataLoader(ds, batch_size=a.batch_size, shuffle=True)))
    t_data = time.time() - t
    masker = Masker(device=dev)
    model = F1Bert(F1BertConfig.preset(a.preset, dropout=a.dropout, n_track_tokens=races[0].meta["n_track_tokens"])).to(
        dev
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    amp = torch.autocast(dev.type, dtype=torch.bfloat16) if a.amp else torch.autocast(dev.type, enabled=False)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    acc = {"mask": 0.0, "forward": 0.0, "backward": 0.0}
    for i in range(a.iters + 3):
        sync()
        t0 = time.time()
        mb = masker(raw)
        sync()
        t1 = time.time()
        with amp:
            out = model(mb)
        sync()
        t2 = time.time()
        loss, _ = compute_losses(out, mb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sync()
        t3 = time.time()
        if i >= 3:  # skip warm-up
            acc["mask"] += t1 - t0
            acc["forward"] += t2 - t1
            acc["backward"] += t3 - t2
    n = a.iters
    print(f"device {dev} | batch {a.batch_size} | amp {a.amp} | dropout {a.dropout}")
    print(f"first batch load (one-off, single process): {t_data:.2f}s")
    print(
        "per step: "
        + " | ".join(f"{k} {v / n * 1000:.0f} ms" for k, v in acc.items())
        + f" | total {sum(acc.values()) / n * 1000:.0f} ms"
    )
    if a.profile:
        from torch.profiler import ProfilerActivity, profile

        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
        with profile(activities=acts) as prof:
            with amp:
                out = model(mb)
            loss, _ = compute_losses(out, mb)
            opt.zero_grad()
            loss.backward()
            sync()
        key = "self_cuda_time_total" if dev.type == "cuda" else "self_cpu_time_total"
        print(prof.key_averages().table(sort_by=key, row_limit=8, max_name_column_width=44))
    if dev.type == "cuda":
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()