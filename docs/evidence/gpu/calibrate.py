"""Does ioreg Device Utilization track a GPU load whose duty cycle we set and time ourselves?

Each phase runs a Nature-CNN-sized MPS forward with synchronize, then sleeps so that measured busy/wall hits a target.
Also: idle, a CPU-only control, and short-burst phases shaped like the real loop's inference (few ms per ~36 ms).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import statistics as st
import subprocess
import sys
import time

import torch
import torch.nn as nn

from gpu_sampler import read


def spin(stop_at: float) -> None:
    x = 0
    while time.time() < stop_at:
        x = (x * 31 + 7) % 1000003


def main() -> None:
    out = sys.argv[1]
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    dev = torch.device("mps")
    net = nn.Sequential(
        nn.Conv2d(3, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        nn.Flatten(), nn.Linear(1920, 512), nn.ReLU(), nn.Linear(512, 8),
    ).to(dev)
    big = torch.rand(1024, 3, 72, 80, device=dev)
    small = torch.rand(96, 3, 72, 80, device=dev)

    def work(x):
        with torch.no_grad():
            r = net(x)
        torch.mps.synchronize()
        return r

    # warm up
    for _ in range(20):
        work(big)
        work(small)
    t0 = time.perf_counter()
    for _ in range(20):
        work(small)
    small_ms = (time.perf_counter() - t0) / 20 * 1e3
    t0 = time.perf_counter()
    for _ in range(5):
        work(big)
    big_ms = (time.perf_counter() - t0) / 5 * 1e3
    print(f"work: batch96 {small_ms:.2f} ms, batch1024 {big_ms:.2f} ms", flush=True)

    samples: list[dict] = []
    fh = open(out, "a")

    def phase(name: str, duty: float | None, x, cpu_procs: int = 0, period_ms: float | None = None) -> None:
        t_end = time.time() + dur
        procs = []
        if cpu_procs:
            ctx = mp.get_context("fork")
            procs = [ctx.Process(target=spin, args=(t_end,)) for _ in range(cpu_procs)]
            for p in procs:
                p.start()
        busy = 0.0
        wall0 = time.perf_counter()
        vals, rend, tiler = [], [], []
        next_sample = time.time() + 1.0
        n = 0
        last = time.perf_counter()
        while time.time() < t_end:
            if duty is not None:
                a = time.perf_counter()
                work(x)
                d = time.perf_counter() - a
                busy += d
                n += 1
                target = d / duty - d if period_ms is None else max(period_ms / 1e3 - d, 0)
                if target > 0:
                    time.sleep(target)
            else:
                time.sleep(0.05)
            if time.time() >= next_sample:
                r = read()
                vals.append(r["Device"])
                rend.append(r["Renderer"])
                tiler.append(r["Tiler"])
                next_sample += 0.5
        wall = time.perf_counter() - wall0
        for p in procs:
            p.join()
        rec = dict(
            phase=name, timed_duty=busy / wall, n=n, ms_per_call=(busy / n * 1e3 if n else 0),
            device_mean=st.mean(vals), device_med=st.median(vals), device_min=min(vals), device_max=max(vals),
            renderer_mean=st.mean(rend), tiler_mean=st.mean(tiler), n_samples=len(vals), load1=os.getloadavg()[0],
        )
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)

    # ordering repeated idle to expose drift of the desktop baseline
    phase("idle_a", None, None)
    phase("duty100_big", 1.0, big)
    phase("duty75_big", 0.75, big)
    phase("duty50_big", 0.50, big)
    phase("duty25_big", 0.25, big)
    phase("duty10_big", 0.10, big)
    phase("idle_b", None, None)
    phase("burst_b96_36ms", 0.1, small, period_ms=36.0)
    phase("burst_b96_12ms", 0.1, small, period_ms=12.0)
    phase("burst_b96_tight", 1.0, small)
    phase("cpu12_no_gpu", None, None, cpu_procs=12)
    phase("cpu12_duty50", 0.5, big, cpu_procs=12)
    phase("idle_c", None, None)
    fh.close()


if __name__ == "__main__":
    main()
