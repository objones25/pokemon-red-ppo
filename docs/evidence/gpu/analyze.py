"""Join runs with ioreg samples; capacity fraction from micro; projected busy fraction with unknowns as ranges."""
from __future__ import annotations

import json
import statistics as st

import numpy as np

runs = [json.loads(x) for x in open("results/runs.jsonl")]
io = [json.loads(x) for x in open("results/ioreg.jsonl")]
micro = {r["model"]: r for r in (json.loads(x) for x in open("results/micro.jsonl"))}
NAT_TIGHT_B96_MS = 1.08  # calibrate.py, forward+sync only, tight loop


def infer_cost_us(model: str, N: int) -> float:
    m = micro[model]
    pts = {int(k): v for k, v in m["infer_ms"].items()}
    if N in pts:
        return pts[N] / N * 1e3
    return (m["infer_L_ms"] + m["infer_c_us"] * N / 1e3) / N * 1e3


def upd_cost_us(model: str, mb: int) -> float:
    return micro[model]["update"][str(mb)]["us_per_sample"]


prev_end = None
base_all = []
print(f"{'label':11} {'model':8} {'N':>4} {'ep':>2} {'mb':>4} {'sps':>6} {'env%':>5} {'fwd%':>5} {'xfer+cpu%':>9} {'upd%':>5} {'gpucall%':>8} "
      f"{'ioreg':>5} {'sd':>4} {'base':>5} {'cap%':>5} {'emu_ms':>6} {'sync_ms':>7} {'load':>5}")
rows = {}
for r in runs:
    t0, t1 = r["window_start"], r["window_end"]
    s = [x for x in io if t0 <= x["t"] <= t1]
    dev = [x["Device"] for x in s]
    load = st.mean(x["load1"] for x in s) if s else float("nan")
    base = [x["Device"] for x in io if prev_end and prev_end + 2 <= x["t"] <= prev_end + 9]
    base_all += base
    prev_end = t1
    f = r["frac"]
    gpu_call = f["fwd"] + f["update"]
    if r["model"] in micro and r["frac"]["update"] > 0:
        G = infer_cost_us(r["model"], r["N"]) + r["epochs"] * upd_cost_us(r["model"], r["mb"])
        cap = r["sps"] * G * 1e-6
    else:
        G, cap = 0.0, 0.0
    ml = r["ms_per_lockstep"]
    row = dict(
        label=r["label"], sps=r["sps"], env=f["env"], fwd=f["fwd"], upd=f["update"], gpu_call=gpu_call,
        io=st.mean(dev), cap=cap, G_us=G, b_ms=(ml["busy"] / r["k"]), sync_ms=ml["sync_ipc"], load=load,
    )
    rows[r["label"]] = row
    print(f"{r['label']:11} {r['model']:8} {r['N']:>4} {r['epochs']:>2} {r['mb']:>4} {r['sps']:>6.0f} {f['env']*100:>5.1f} {f['fwd']*100:>5.1f} "
          f"{(f['xfer']+f['tocpu'])*100:>9.1f} {f['update']*100:>5.1f} {gpu_call*100:>8.1f} {st.mean(dev):>5.1f} {st.pstdev(dev):>4.0f} "
          f"{(st.mean(base) if base else float('nan')):>5.1f} {cap*100:>5.1f} {ml['emu']/r['k']:>6.2f} {ml['sync_ipc']:>7.2f} {load:>5.1f}")
print("idle-gap baseline Device%: mean", round(st.mean(base_all), 1), "p10/p90", np.percentile(base_all, [10, 90]), "n", len(base_all))

print("\nSaturated cost per env step on M4 (us): infer(b=96)/96 + 3*update(mb512)")
G = {}
for m in ("nature", "impala1", "impala4"):
    i = NAT_TIGHT_B96_MS / 96 * 1e3 if m == "nature" else infer_cost_us(m, 96)
    u = 3 * upd_cost_us(m, 512)
    G[m] = i + u
    print(f"  {m:8} params={micro[m]['params']:>8} infer={i:7.1f} update3={u:7.1f} total={G[m]:7.1f}")

print("\nProjected GPU busy fraction f = (G/g) / (b(1+s)/W + G/g + P), W=V-1, one main-process vCPU")
print("Corners: LOW f uses b=8ms s=0.35 P=P_hi; HIGH f uses b=2.1ms s=0 P=0; 'mid' b=5 s=0.2 P=P_hi/2. P_hi = 40us per step (CPU side)")
gs = (1, 2.9, 5, 10, 20)
for m in G:
    print(f"\n  {m}: rows V; columns g; cell low / mid / high %")
    print("    V   " + "".join(f"{('g=' + str(g)):>16}" for g in gs))
    for V in (8, 12, 16, 32, 64, 128):
        W = V - 1
        cells = []
        for g in gs:
            gg = G[m] / g

            def f(b, s, P):
                return 100 * gg / (b * 1e3 * (1 + s) / W + gg + P)

            cells.append(f"{f(8, .35, 40):4.0f}/{f(5, .2, 20):3.0f}/{f(2.1, 0, 0):3.0f}")
        print(f"  {V:>4} " + "".join(f"{c:>16}" for c in cells))

print("\nvCPUs where f=50% (ignoring P): V = 1 + b(1+s)*g/G")
for m in G:
    print(" ", m, {f"g={g}": [round(1 + b * 1e3 * (1 + s) * g / G[m]) for b, s in ((2.1, 0), (5, .2), (8, .35))] for g in gs})

print("\nSteps/s ceiling projections for a 32 vCPU pod, knob cost at V=12/32, mid case b=5, s=.2, P=20us:")
for label in ("nat_k8_a", "imp1_k8_a", "imp4_k8_a", "nat_ep10", "nat_ep30"):
    if label not in rows:
        continue
    g_us = rows[label]["G_us"]
    print(" ", label, {f"V={V},g={g}": round(1e6 / (5e3 * 1.2 / (V - 1) + g_us / g + 20)) for V in (12, 32) for g in (1, 2.9, 5)})
