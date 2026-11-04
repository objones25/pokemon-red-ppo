#!/usr/bin/env python3
"""Does pinning each lockstep worker to its own physical core recover the free-running b? Same loop, W=26."""
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time

import pod_benchmark as pb


def pinned_worker(i, *a):
    core = int(os.environ["PIN_FIRST"]) + i
    os.sched_setaffinity(0, {core})
    real_lock_worker(i, *a)


real_lock_worker = pb.lock_worker


def main():
    import torch

    rom_in, state_in, out, node_first = sys.argv[1:5]
    os.environ["PIN_FIRST"] = node_first
    tmp = tempfile.mkdtemp(prefix="podpin_")
    rom = os.path.join(tmp, "red.gb")
    shutil.copyfile(rom_in, rom)
    state = open(state_in, "rb").read()
    fh = open(out, "a")
    start = time.time()

    def emit(rec):
        rec = {"t_elapsed_s": round(time.time() - start, 1), **rec}
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
        print(json.dumps(rec, default=str), flush=True)

    class A:
        pass

    a = A()
    a.rom, a.smoke, a.window, a.warmup, a.cfg_max = rom, False, 30.0, 8.0, 100.0
    torch.set_num_threads(2)
    dev = torch.device("cuda")
    sampler = pb.UtilSampler("cuda")
    ctx = mp.get_context("spawn")
    deadline = start + 360
    # Main process on the core after the workers, with torch's threads inheriting that plus one.
    os.sched_setaffinity(0, {int(node_first) + 26, int(node_first) + 27})
    import __main__

    pb.lock_worker = __main__.pinned_worker
    for k, T in ((8, 128), (32, 64), (1, 512)):
        cfg = dict(label=f"nature_pinned_k{k}", model="nature", W=26, k=k, T=T, epochs=3, mb=512)
        pb.full_loop(a, ctx, dev, sampler, cfg, state, None, emit, deadline)
    sampler.close()
    shutil.rmtree(tmp, ignore_errors=True)
    emit(dict(kind="end", total_s=time.time() - start))
    fh.close()


if __name__ == "__main__":
    main()
