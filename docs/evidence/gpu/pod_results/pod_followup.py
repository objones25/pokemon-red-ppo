#!/usr/bin/env python3
"""Follow-up to pod_benchmark.py on the same pod: the quota read 27.2 vCPUs, so W and k need a wider scan than W=V-1.

Reuses pod_benchmark's env, workers and loop unchanged; only the worker count and k differ. No GPU reference, so
capacity_frac is null here: compute it from pod.jsonl. Run next to pod_benchmark.py with the same --rom and --state.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import statistics as st
import tempfile
import time

import pod_benchmark as pb


def throttle():
    d = dict(line.split() for line in open("/sys/fs/cgroup/cpu.stat"))
    return {k: int(d[k]) for k in ("nr_periods", "nr_throttled", "throttled_usec", "usage_usec")}


def free_b(ctx, W, rom, state, window, emit):
    q, barrier = ctx.Queue(), ctx.Barrier(W + 1)
    ps = [ctx.Process(target=pb.free_worker, args=(i, rom, state, barrier, window, q, 2000), daemon=True) for i in range(W)]
    for p in ps:
        p.start()
    barrier.wait(timeout=300)
    t0 = throttle()
    res = [q.get(timeout=window + 120) for _ in range(W)]
    t1 = throttle()
    for p in ps:
        p.join(timeout=20)
    b = [r["wall"] / r["steps"] * 1e3 for r in res]
    emit(dict(kind="b_free", W=W, b_ms_mean=st.mean(b), b_ms_max=max(b), aggregate_sps=sum(r["steps"] / r["wall"] for r in res),
              cpu_over_wall=st.mean(r["cpu"] / r["wall"] for r in res), throttle={k: t1[k] - t0[k] for k in t0}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", default="followup.jsonl")
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=8.0)
    ap.add_argument("--cfg-max", type=float, default=100.0)
    ap.add_argument("--b-window", type=float, default=15.0)
    a = ap.parse_args()
    import torch

    tmp = tempfile.mkdtemp(prefix="podfollow_")
    rom = os.path.join(tmp, "red.gb")
    shutil.copyfile(a.rom, rom)
    a.rom = rom
    a.smoke = False
    state = open(a.state, "rb").read()
    fh = open(a.out, "a")
    start = time.time()
    deadline = start + a.minutes * 60

    def emit(rec):
        rec = {"t_elapsed_s": round(time.time() - start, 1), **rec}
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
        print(json.dumps(rec, default=str), flush=True)

    torch.set_num_threads(2)
    dev = torch.device("cuda")
    sampler = pb.UtilSampler("cuda")
    ctx = mp.get_context("spawn")
    for W in (27, 32):
        free_b(ctx, W, a.rom, state, a.b_window, emit)
    scan = [  # model, W, k, T
        ("nature", 26, 32, 64),  # repeat of nature_k32: how repeatable is one window?
        ("nature", 24, 32, 64),
        ("nature", 27, 32, 64),
        ("nature", 31, 32, 64),
        ("nature", 26, 16, 128),
        ("nature", 26, 64, 32),
        ("impala1", 24, 32, 64),
        ("impala1", 27, 32, 64),
    ]
    for name, W, k, T in scan:
        cfg = dict(label=f"{name}_W{W}_k{k}", model=name, W=W, k=k, T=T, epochs=3, mb=512)
        t0 = throttle()
        try:
            pb.full_loop(a, ctx, dev, sampler, cfg, state, None, emit, deadline)
        except Exception as e:
            emit(dict(kind="error", where=cfg["label"], error=f"{type(e).__name__}: {e}"[:400]))
        t1 = throttle()
        emit(dict(kind="throttle", label=cfg["label"], **{k2: t1[k2] - t0[k2] for k2 in t0}))
    sampler.close()
    shutil.rmtree(tmp, ignore_errors=True)
    emit(dict(kind="end", total_s=time.time() - start))
    fh.close()


if __name__ == "__main__":
    main()
