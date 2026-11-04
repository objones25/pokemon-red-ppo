"""Gated benchmark runner. For each config: wait (bounded) until other processes use < 1 core, launch the benchmark,
sample system-wide busy cores minus own process tree every 2 s, crop to the benchmark's timed window, append to JSONL.

Flags a run contaminated when background > 1.0 core mean during the window. Contaminated runs are retried at the end
of their group while the deadline allows. All runs, including flagged ones, are kept in the log.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
BENCH = os.path.join(RES, "runs.jsonl")
LOG = os.path.join(RES, "gate.jsonl")

C = lambda *x: ["bench_custom.py", *x]  # noqa: E731
S = lambda *x: ["bench_sb3.py", *x]  # noqa: E731

GROUPS = {
    "core": {
        "S10": S("--N", "10"),
        "S12": S("--N", "12"),
        "S10dict2048": S("--N", "10", "--obs", "dict", "--T", "2048", "--mb", "8192"),
        "C12x1": C("--workers", "12", "--k", "1"),
        "C12x8": C("--workers", "12", "--k", "8", "--T", "64"),
        "C12x8F": C("--workers", "12", "--k", "8", "--T", "64", "--fast"),
        "S10F": S("--N", "10", "--fast"),
        "C12x1cpuinf": C("--workers", "12", "--k", "1", "--infer", "cpu"),
        "C12x8cpuinf": C("--workers", "12", "--k", "8", "--T", "64", "--infer", "cpu"),
    },
    "second": {
        "C12x8convcpu": C("--workers", "12", "--k", "8", "--T", "64", "--conv", "cpu"),
        "C12x1F": C("--workers", "12", "--k", "1", "--fast"),
        "S10cpu": S("--N", "10", "--device", "cpu", "--T", "256"),
    },
    "third": {
        "C12x1nopol": C("--workers", "12", "--k", "1", "--no-policy"),
        "C12x8nopol": C("--workers", "12", "--k", "8", "--T", "64", "--no-policy"),
        "S10noinst": S("--N", "10", "--no-instrument"),
        "C12x1T2048": C("--workers", "12", "--k", "1", "--T", "2048"),
        "C12x1conv": C("--workers", "12", "--k", "1", "--conv", "cpu"),
    },
}


def thermal() -> str:
    o = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    return " ".join(o.split())[-120:]


def gpu_util():
    o = subprocess.run(["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"], capture_output=True, text=True).stdout
    for tok in o.split(","):
        if "Device Utilization %" in tok:
            try:
                return int(tok.split("=")[-1].strip().strip('"}'))
            except ValueError:
                return None
    return None


class Monitor:
    """System busy cores minus self tree, every `dt` seconds."""

    def __init__(self, root_pid, dt=2.0):
        self.me = psutil.Process()
        self.root = root_pid
        self.dt = dt
        self.samples = []
        self._prev = {}
        self._stop = threading.Event()
        self._cpu = psutil.cpu_times()
        self._t = time.time()
        self._tree_cpu()

    def _tree_cpu(self):
        total = 0.0
        ps = [self.me]
        try:
            ps += self.me.children(recursive=True)
        except psutil.Error:
            pass
        for p in ps:
            try:
                c = p.cpu_times()
                v = c.user + c.system
            except psutil.Error:
                continue
            total += max(0.0, v - self._prev.get(p.pid, 0.0))
            self._prev[p.pid] = v
        return total

    def sample(self):
        c = psutil.cpu_times()
        t = time.time()
        busy = (sum(c) - c.idle) - (sum(self._cpu) - self._cpu.idle)
        mine = self._tree_cpu()
        dt = t - self._t
        self.samples.append((t, busy / dt, mine / dt, (busy - mine) / dt))
        self._cpu, self._t = c, t

    def run(self):
        while not self._stop.wait(self.dt):
            self.sample()

    def start(self):
        self._th = threading.Thread(target=self.run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join()


def bg_now(dur=4.0):
    m = Monitor(None)
    time.sleep(dur)
    m.sample()
    return m.samples[-1][3]


def top_others(n=5):
    mine = {psutil.Process().pid} | {p.pid for p in psutil.Process().children(recursive=True)}
    ps = []
    for p in psutil.process_iter(["pid", "name"]):
        if p.pid in mine:
            continue
        try:
            p.cpu_percent(None)
            ps.append(p)
        except psutil.Error:
            pass
    time.sleep(1.0)
    rows = []
    for p in ps:
        try:
            rows.append((p.cpu_percent(None), p.info["name"], p.pid))
        except psutil.Error:
            pass
    rows.sort(reverse=True)
    return [(round(a, 0), b, c) for a, b, c in rows[:n]]


def gate(max_wait, thr=1.0):
    t0 = time.time()
    while True:
        b = bg_now()
        if b < thr or time.time() - t0 > max_wait:
            return b, time.time() - t0
        time.sleep(4)


def run_one(name, cmd, rnd, args, group):
    bg0, waited = gate(args.max_wait)
    gpu0 = gpu_util()
    tops = top_others() if bg0 >= 1.0 else []
    la0 = os.getloadavg()
    th0 = thermal()
    full = [PY, os.path.join(HERE, cmd[0]), "--label", name, "--out", BENCH, "--window", str(args.window), *cmd[1:]]
    t0 = time.time()
    m = Monitor(None)
    m.start()
    p = subprocess.run(full, cwd=HERE, capture_output=True, text=True, timeout=args.run_timeout)
    m.stop()
    dur = time.time() - t0
    la1 = os.getloadavg()
    rec = dict(name=name, round=rnd, group=group, rc=p.returncode, duration=dur, gate_bg=bg0, gate_wait=waited,
               gpu_util_pre=gpu0, load_pre=la0, load_post=la1, thermal=th0, top_others_pre=tops, t0=t0)
    if p.returncode != 0:
        rec["stderr"] = p.stderr[-2000:]
        rec["contaminated"] = None
    else:
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")][-1]
        r = json.loads(line)
        s = [x for x in m.samples if r["window_start"] <= x[0] <= r["window_end"]]
        bg = [x[3] for x in s]
        rec.update(
            sps=r["sps"], bg_mean=sum(bg) / len(bg), bg_max=max(bg), own_cores=sum(x[2] for x in s) / len(s),
            busy_cores=sum(x[1] for x in s) / len(s), window=r["wall"], contaminated=(sum(bg) / len(bg)) > 1.0,
        )
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec.get(k) for k in ("name", "round", "rc", "sps", "bg_mean", "bg_max", "own_cores", "gate_wait", "load_pre", "load_post", "contaminated", "duration")}), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="core,second,third")
    ap.add_argument("--deadline-min", type=float, default=40)
    ap.add_argument("--window", type=float, default=60)
    ap.add_argument("--max-wait", type=float, default=60)
    ap.add_argument("--run-timeout", type=float, default=300)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    deadline = time.time() + a.deadline_min * 60
    only = set(filter(None, a.only.split(",")))
    for g in a.groups.split(","):
        flagged = []
        for rnd in range(1, a.rounds + 1):
            for name, cmd in GROUPS[g].items():
                if only and name not in only:
                    continue
                if time.time() > deadline:
                    print("deadline reached, stopping", flush=True)
                    return
                r = run_one(name, cmd, rnd, a, g)
                if r.get("contaminated") or r["rc"] != 0:
                    flagged.append((name, cmd, rnd))
        for name, cmd, rnd in flagged:
            if time.time() > deadline - 120:
                break
            run_one(name, cmd, rnd + 10, a, g)
    print("matrix complete", flush=True)


if __name__ == "__main__":
    main()
