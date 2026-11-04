"""Which pinned CPU sets give real emulator speed? Sysfs hides SMT inside this VM, so measure. Prints steps/s per process."""
import json, multiprocessing as mp, os, sys, time
from pathlib import Path

def run(core, seconds, q, fast):
    from pokered.config import EnvConfig
    from pokered.env import PokemonRedEnv
    if core is not None:
        os.sched_setaffinity(0, {core})
    env = PokemonRedEnv(EnvConfig(rom=Path("/workspace/rom/Pokemon Red.gb"), stagger_first_episode=False, fast_mode=fast))
    env.reset(seed=0)
    import numpy as np
    g = np.random.default_rng(core or 0)
    end = time.perf_counter() + seconds + 3
    while time.perf_counter() < end - seconds:
        env.step(int(g.integers(0, 7)))
    n, t0, c0 = 0, time.perf_counter(), time.process_time()
    while time.perf_counter() < end:
        env.step(int(g.integers(0, 7))); n += 1
    t = time.perf_counter() - t0
    q.put((core, n / t, (time.process_time() - c0) / t))

def trial(label, cores, seconds=10, fast=False):
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    ps = [ctx.Process(target=run, args=(c, seconds, q, fast)) for c in cores]
    [p.start() for p in ps]
    r = [q.get() for _ in ps]
    [p.join() for p in ps]
    sps = [x[1] for x in r]
    print(json.dumps({"label": label, "n": len(cores), "fast": fast, "total": round(sum(sps)), "mean": round(sum(sps)/len(sps), 1), "min": round(min(sps), 1), "max": round(max(sps), 1), "cpu_over_wall": round(sum(x[2] for x in r)/len(r), 3), "pairs_like": [x[1].__round__() for x in r][:6]}), flush=True)

if __name__ == "__main__":
    trial("cpu0 alone", [0])
    trial("cpu0+1", [0, 1])
    trial("cpu0+2", [0, 2])
    trial("cpu0+126", [0, 126])
    trial("26 at 0-25", list(range(26)))
    trial("26 at even 0-50", list(range(0, 52, 2)))
    trial("26 at odd 1-51", list(range(1, 53, 2)))
    trial("26 spread every 9", list(range(0, 234, 9)))
    trial("26 unpinned", [None] * 26)
    print("throttle", Path("/sys/fs/cgroup/cpu/cpu.stat").read_text().replace("\n", " "))
