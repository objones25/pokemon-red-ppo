"""MPS microbench without emulators: inference latency vs batch, update time vs minibatch, encode-vs-wait split.

Fits T_inf(n) = L + c*n so the serial/latency part and the data-proportional part can be carried separately.
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time

import numpy as np
import torch

from models import Policy

pc = time.perf_counter
dev = torch.device("mps")


def timed(fn, reps: int) -> float:
    ts = []
    for _ in range(reps):
        t = pc()
        fn()
        ts.append(pc() - t)
    return st.median(ts)


def main() -> None:
    out = sys.argv[1]
    rng = np.random.default_rng(0)
    pool = rng.integers(0, 256, (2048, 3, 72, 80), dtype=np.uint8)
    res = []
    for name, ns, mbs in (
        ("nature", (12, 24, 48, 96, 192, 384, 768), (128, 512, 2048)),
        ("impala1", (12, 24, 48, 96, 192, 384, 768), (128, 512, 2048)),
        ("impala4", (12, 24, 48, 96, 192, 384), (64, 128, 512)),
    ):
        m = Policy(name).to(dev)
        opt = torch.optim.Adam(m.parameters(), 3e-4, eps=1e-5)
        inf = {}
        for n in ns:
            arr = pool[:n]

            def f():
                x = torch.from_numpy(arr).to(dev).float().div_(255.0)
                with torch.no_grad():
                    lo, v = m(x)
                    d = torch.distributions.Categorical(logits=lo)
                    a = d.sample()
                    lp = d.log_prob(a)
                torch.mps.synchronize()
                a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()

            for _ in range(3):
                f()
            inf[n] = timed(f, 25 if name != "impala4" else 10) * 1e3
        xs = np.array(list(inf)), np.array(list(inf.values()))
        c, L = np.polyfit(xs[0], xs[1], 1)
        upd = {}
        for mb in mbs:
            arr = pool[:mb]
            ac = torch.from_numpy(rng.integers(0, 7, mb)).to(dev)
            lpb = torch.zeros(mb, device=dev) - 1.9
            adv = torch.randn(mb, device=dev)
            ret = torch.randn(mb, device=dev)
            enc, tot = [], []
            reps = 12 if name != "impala4" else 5
            for i in range(reps + 2):
                t0 = pc()
                x = torch.from_numpy(arr).to(dev).float().div_(255.0)
                lo, v = m(x)
                d = torch.distributions.Categorical(logits=lo)
                r = (d.log_prob(ac) - lpb).exp()
                a2 = (adv - adv.mean()) / (adv.std() + 1e-8)
                pl = -torch.min(a2 * r, a2 * r.clamp(0.8, 1.2)).mean()
                loss = pl + 0.5 * (ret - v).pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
                opt.step()
                t1 = pc()
                torch.mps.synchronize()
                t2 = pc()
                if i >= 2:
                    enc.append(t1 - t0)
                    tot.append(t2 - t0)
            upd[mb] = dict(
                total_ms=st.median(tot) * 1e3, encode_ms=st.median(enc) * 1e3,
                us_per_sample=st.median(tot) / mb * 1e6,
            )
        rec = dict(
            model=name, params=sum(p.numel() for p in m.parameters()), infer_ms=inf,
            infer_L_ms=float(L), infer_c_us=float(c * 1e3), update=upd,
        )
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        del m, opt
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
