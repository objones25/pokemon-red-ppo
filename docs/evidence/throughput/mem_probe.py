"""Claim D. One child process per case: measure RSS and USS at each stage of adding PyBoy envs.

Cases: bare (numpy + pyboy only), with gymnasium, with SB3 (which pulls in torch). Also cross-checks RSS with `ps`.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def probe(case, fast, q):
    import psutil

    me = psutil.Process()

    def mm():
        fi = me.memory_full_info()
        ps = int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout)
        return dict(rss=fi.rss / 1e6, uss=fi.uss / 1e6, ps_rss=ps * 1024 / 1e6)

    out = dict(case=case, fast=fast, python=mm())
    if case == "sb3":
        import stable_baselines3.common.vec_env.subproc_vec_env  # noqa: F401
    elif case == "gym":
        import gymnasium  # noqa: F401
    out["after_framework_import"] = mm()
    from env_core import CoreEnv

    out["after_env_import"] = mm()
    envs = []
    for n in range(1, 9):
        e = CoreEnv(fast_mode=fast)
        e.reset()
        for i in range(300):
            e.step(i % 7)
        envs.append(e)
        if n in (1, 2, 4, 8):
            out[f"envs_{n}"] = mm()
    for e in envs:
        e.close()
    q.put(out)


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    res = []
    for case, fast in (("bare", False), ("gym", False), ("sb3", False), ("bare", True)):
        q = ctx.Queue()
        p = ctx.Process(target=probe, args=(case, fast, q))
        p.start()
        r = q.get()
        p.join()
        res.append(r)
        print(json.dumps(r), flush=True)
        o0, o1, o8 = r["after_env_import"], r["envs_1"], r["envs_8"]
        print(
            f"{case:5s} fast={fast}: pre-env rss {o0['rss']:.0f} uss {o0['uss']:.0f} | 1 env rss {o1['rss']:.0f} uss {o1['uss']:.0f} "
            f"| 8 envs rss {o8['rss']:.0f} uss {o8['uss']:.0f} | first env +{o1['rss']-o0['rss']:.1f} MB rss, "
            f"each further env +{(o8['rss']-o1['rss'])/7:.1f} MB rss +{(o8['uss']-o1['uss'])/7:.1f} MB uss",
            flush=True,
        )
    with open(os.path.join(HERE, "results", "mem.json"), "w") as f:
        json.dump(res, f, indent=1)
