#!/usr/bin/env python3
"""How much of a CUDA GPU can the Pokemon Red PPO loop use? One file, bounded time, run on a real pod.

Measures, in order of importance (later phases are skipped first if time runs out):
  1. b: emulator ms per step per process, at 1, V/4, V/2, V-1, V processes. V comes from cgroup + affinity, not nproc.
  2. GPU speed with no emulators: inference vs batch, update vs minibatch, sparse-burst inference, saturated utilization.
  3. The full synchronous PPO loop at each config: steps/s, time split, utilization, power, SM clock, capacity fraction.

Prints and writes JSONL (flushed per record, so a killed run keeps what it finished). Makes no Runpod API calls and does
not stop the pod: stop or terminate it yourself when this exits.

Setup on the pod (versions match the local runs; start.state is only valid for the PyBoy that wrote it):
    pip install pyboy==2.9.0 numpy          # torch comes from the image; check torch.cuda.is_available()
    python pod_benchmark.py --rom /workspace/red.gb --state /workspace/start.state --out /workspace/pod.jsonl

Utilization meaning (docs.nvidia.com/deploy/nvidia-smi): utilization.gpu is the percent of the sample period in which
at least one kernel ran. It is time-based, so it says nothing about how many SMs were busy. The capacity fraction
below (saturated cost x achieved steps/s) and power are the checks on it. power.draw is instantaneous on GA100:
power.draw.average is documented as unsupported there, so it is not queried.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import platform
import re
import shutil
import signal
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from multiprocessing import shared_memory as sm

import numpy as np  # torch is imported only in the parent, so spawned workers never load it

ROM_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"
BUTTONS = ("down", "left", "right", "up", "a", "b", "start")
N_ACTIONS = len(BUTTONS)
FRAMES_PER_ACTION, HOLD_FRAMES = 24, 8
SHAPE = (3, 72, 80)
FRAME_BYTES = int(np.prod(SHAPE))
MAX_STEPS = 10240
GAMMA, LAM, CLIP, VF, MAXGRAD, LR = 0.99, 0.95, 0.2, 0.5, 0.5, 3e-4
# Addresses from ram_map.py, kept here so that this file stands alone.
A_BADGES, A_MAP, A_Y, A_X, A_PARTY_N, A_PARTY, MON_SIZE, OFF_LEVEL = 0xD356, 0xD35E, 0xD361, 0xD362, 0xD163, 0xD16B, 44, 33
A_BATTLE, A_EVENTS, N_EVENTS = 0xD057, 0xD747, 320

# Local reference: M4 Pro 20-core, MPS, torch 2.15, tight loop (micro.py). us per sample per pass at mb=512; ms at b=96.
M4_REF = {
    "nature": {"update_us_per_sample": 23.6, "infer_b96_ms": 2.32},
    "impala1": {"update_us_per_sample": 161.8, "infer_b96_ms": 4.92},
    "impala4": {"update_us_per_sample": 977.5, "infer_b96_ms": 25.04},
}


class TimeUp(Exception):
    pass


class Env:
    """Same step as the training env: 8-frame hold in 24 frames, render last, 2x2 mean, 3-stack, full RAM read."""

    def __init__(self, rom: str, state: bytes) -> None:
        from pyboy import PyBoy

        self.py = PyBoy(rom, window="null", log_level="ERROR", sound_emulated=False)
        self.py.set_emulation_speed(0)
        self.state = state
        self.stack = np.zeros(SHAPE, np.uint8)
        self.seen: set[tuple[int, int, int]] = set()
        self.flags = bytes(N_EVENTS)
        self.t = 0
        self.resets = 0

    def _frame(self) -> np.ndarray:
        a = self.py.screen.ndarray[:, :, 0].astype(np.uint16)
        return ((a[::2, ::2] + a[1::2, ::2] + a[::2, 1::2] + a[1::2, 1::2]) >> 2).astype(np.uint8)

    def reset(self) -> np.ndarray:
        self.py.load_state(io.BytesIO(self.state))
        self.py.tick(3, True, False)  # a restored screen is blank for 2 frames
        self.stack[:] = self._frame()
        self.seen = set()
        self.t = 0
        self.resets += 1
        return self.stack.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        py, b = self.py, BUTTONS[int(action)]
        py.button_press(b)
        py.tick(HOLD_FRAMES, False, False)
        py.button_release(b)
        py.tick(FRAMES_PER_ACTION - HOLD_FRAMES - 1, False, False)
        py.tick(1, True, False)
        m = py.memory
        n = m[A_PARTY_N]
        key = (m[A_MAP], m[A_X], m[A_Y])
        flags = bytes(m[A_EVENTS + i] for i in range(N_EVENTS))
        lv = sum(m[A_PARTY + OFF_LEVEL + MON_SIZE * i] for i in range(min(n, 6)))
        r = 0.0
        if m[A_BATTLE] == 0 and key not in self.seen:
            self.seen.add(key)
            r += 0.005
        if flags != self.flags:
            self.flags = flags
        r += 0.0 * (lv + m[A_BADGES])
        self.stack[:2] = self.stack[1:]
        self.stack[2] = self._frame()
        self.t += 1
        return self.stack.copy(), r, self.t >= MAX_STEPS

    def close(self) -> None:
        self.py.stop(save=False)  # the default save=True writes a .ram next to the ROM


# ---------------------------------------------------------------- phase 1 worker
def free_worker(i, rom, state_bytes, barrier, window, q, seed):
    env = Env(rom, state_bytes)
    rng = np.random.default_rng(seed + i)
    env.reset()
    for _ in range(20):
        env.step(rng.integers(N_ACTIONS))
    barrier.wait(timeout=300)
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    n = 0
    while True:
        for _ in range(16):
            _, _, tr = env.step(rng.integers(N_ACTIONS))
            n += 1
            if tr:
                env.reset()
        if time.perf_counter() - t0 >= window:
            break
    wall = time.perf_counter() - t0
    q.put(dict(i=i, steps=n, wall=wall, cpu=time.process_time() - cpu0))
    env.close()


# ---------------------------------------------------------------- phase 3 shared memory + worker
def views(buf, N, W):
    off = 0
    obs = np.ndarray((N, *SHAPE), np.uint8, buf, off)
    off += N * FRAME_BYTES
    act = np.ndarray((N,), np.int64, buf, off)
    off += 8 * N
    rew = np.ndarray((N,), np.float32, buf, off)
    off += 4 * N
    done = np.ndarray((N,), np.uint8, buf, off)
    off += N + (-N % 8)
    tim = np.ndarray((W,), np.float64, buf, off)
    return obs, act, rew, done, tim


def shm_size(N, W):
    return N * FRAME_BYTES + 12 * N + N + (-N % 8) + 8 * W


def lock_worker(i, k, conn, name, N, W, rom, state_bytes):
    s = sm.SharedMemory(name=name)
    obs, act, rew, done, tim = views(s.buf, N, W)
    lo = i * k
    envs = [Env(rom, state_bytes) for _ in range(k)]
    for j, e in enumerate(envs):
        obs[lo + j] = e.reset()
    conn.send_bytes(b"r")
    busy = 0.0
    try:
        while True:
            try:
                msg = conn.recv_bytes()
            except EOFError:
                break
            if msg == b"q":
                break
            t0 = time.perf_counter()
            for j, e in enumerate(envs):
                o, r, tr = e.step(act[lo + j])
                if tr:
                    o = e.reset()
                obs[lo + j], rew[lo + j], done[lo + j] = o, r, tr
            busy += time.perf_counter() - t0
            tim[i] = busy
            conn.send_bytes(b"d")
    finally:
        for e in envs:
            e.close()
        del obs, act, rew, done, tim
        s.close()


# ---------------------------------------------------------------- utilization
SMI_FIELDS = "utilization.gpu,utilization.memory,power.draw,clocks.sm,memory.used,pstate"


def parse_smi_line(line: str) -> dict | None:
    """Parse one csv,noheader,nounits line. '[N/A]', '[Not Supported]' etc become None."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        return None

    def num(x):
        try:
            return float(x)
        except ValueError:
            return None

    return dict(util=num(parts[0]), mem_util=num(parts[1]), power=num(parts[2]), sm_mhz=num(parts[3]),
                vram_mb=num(parts[4]), pstate=parts[5])


class UtilSampler:
    """nvidia-smi loop on CUDA, ioreg AGX on macOS, nothing otherwise. Thread-safe append; slice by wall clock."""

    def __init__(self, device_type: str) -> None:
        self.kind = "none"
        self.rows: list[tuple[float, dict]] = []
        self.proc = None
        self.stop_flag = threading.Event()
        self.thread = None
        if device_type == "cuda" and shutil.which("nvidia-smi"):
            self.kind = "nvidia-smi"
            self.proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits", "-i", "0", "-lms", "500"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
            self.thread = threading.Thread(target=self._read_smi, daemon=True)
        elif device_type == "mps" and shutil.which("ioreg"):
            self.kind = "ioreg"
            self.thread = threading.Thread(target=self._read_ioreg, daemon=True)
        if self.thread:
            self.thread.start()

    def _read_smi(self):
        for line in self.proc.stdout:
            r = parse_smi_line(line)
            if r:
                self.rows.append((time.time(), r))

    def _read_ioreg(self):
        pat = re.compile(r'"Device Utilization %"=(\d+)')
        while not self.stop_flag.wait(0.5):
            s = subprocess.run(["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"], capture_output=True, text=True).stdout
            m = pat.search(s)
            if m:
                self.rows.append((time.time(), dict(util=float(m.group(1)))))

    def window(self, t0: float, t1: float) -> dict:
        rows = [r for t, r in list(self.rows) if t0 <= t <= t1]
        out: dict = dict(source=self.kind, n=len(rows))
        for key in ("util", "mem_util", "power", "sm_mhz", "vram_mb"):
            v = [r[key] for r in rows if r.get(key) is not None]
            if v:
                out[key + "_mean"] = round(st.mean(v), 2)
                out[key + "_p10"] = round(float(np.percentile(v, 10)), 2)
                out[key + "_p90"] = round(float(np.percentile(v, 90)), 2)
        ps = [r["pstate"] for r in rows if r.get("pstate")]
        if ps:
            out["pstates"] = sorted(set(ps))
        return out

    def close(self):
        self.stop_flag.set()
        if self.proc:
            self.proc.terminate()


def nvml_probe() -> dict:
    """torch.cuda.utilization needs nvidia-ml-py, which an image may lack. Record, never depend on it."""
    import torch

    if not torch.cuda.is_available():
        return dict(available=False)
    try:
        return dict(available=True, util=torch.cuda.utilization(0), mem=torch.cuda.memory_usage(0))
    except Exception as e:  # ModuleNotFoundError without pynvml, or NVML driver errors: both only lose a cross-check
        return dict(available=False, error=f"{type(e).__name__}: {e}"[:200])


# ---------------------------------------------------------------- system facts
def effective_vcpus() -> dict:
    """nproc in a container can report the host. Use the smallest of affinity and cgroup quota."""
    out: dict = dict(os_cpu_count=os.cpu_count())
    try:
        out["affinity"] = len(os.sched_getaffinity(0))
    except AttributeError:
        out["affinity"] = os.cpu_count()
    quota = None
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            quota = float(q) / float(p)
    except (OSError, ValueError):
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                quota = q / p
        except (OSError, ValueError):
            pass
    out["cgroup_quota_vcpus"] = quota
    out["usable"] = max(1, int(min(out["affinity"], quota if quota else out["affinity"])))
    return out


def sysinfo(device_type: str) -> dict:
    import importlib.metadata as md

    import torch

    cpu_model = platform.processor()
    threads_per_core = None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
        sib = open("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list").read().strip()
        threads_per_core = len(re.split("[,]", sib)) if "," in sib else (2 if "-" in sib else 1)
    except OSError:
        pass
    env = {k: v for k, v in os.environ.items() if k.startswith("RUNPOD_") and not re.search("KEY|SECRET|TOKEN|PASS", k)}
    info = dict(
        kind="sysinfo", date=time.strftime("%F %T %z"), platform=platform.platform(), python=sys.version.split()[0],
        cpu_model=cpu_model, threads_per_core=threads_per_core, vcpus=effective_vcpus(), loadavg=os.getloadavg(),
        versions={p: md.version(p) for p in ("torch", "numpy", "pyboy")}, runpod_env=env, device=device_type,
    )
    if device_type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_rt"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        info["nvml_probe"] = nvml_probe()
        if shutil.which("nvidia-smi"):
            q = "name,driver_version,power.limit,clocks.max.sm,pstate,memory.total"
            r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True)
            info["nvidia_smi"] = r.stdout.strip()
    return info


# ---------------------------------------------------------------- models
def build_models():
    import torch
    import torch.nn as nn

    class Res(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.a, self.b = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)

        def forward(self, x):
            return x + self.b(torch.relu(self.a(torch.relu(x))))

    class Policy(nn.Module):
        def __init__(self, name):
            super().__init__()
            if name == "nature":
                self.body = nn.Sequential(
                    nn.Conv2d(3, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1),
                    nn.ReLU(), nn.Flatten(),
                )
                h = 512
            else:
                w, ls, c0 = int(name.removeprefix("impala")), [], 3
                for c in (16 * w, 32 * w, 32 * w):
                    ls += [nn.Conv2d(c0, c, 3, padding=1), nn.MaxPool2d(3, 2, 1), Res(c), Res(c)]
                    c0 = c
                self.body = nn.Sequential(*ls, nn.ReLU(), nn.Flatten())
                h = 256
            with torch.no_grad():
                f = self.body(torch.zeros(1, *SHAPE)).shape[1]
            self.fc = nn.Sequential(nn.Linear(f, h), nn.ReLU())
            self.pi, self.vf = nn.Linear(h, N_ACTIONS), nn.Linear(h, 1)

        def forward(self, x):
            z = self.fc(self.body(x))
            return self.pi(z), self.vf(z).squeeze(1)

    return Policy


def make_sync(dev):
    import torch

    if dev.type == "cuda":
        return torch.cuda.synchronize
    if dev.type == "mps":
        return torch.mps.synchronize
    return lambda: None


def ppo_minibatch(model, opt, xb, ab, lpb, advb, retb):
    import torch
    import torch.nn as nn

    logits, v = model(xb)
    dist = torch.distributions.Categorical(logits=logits)
    adv = (advb - advb.mean()) / (advb.std() + 1e-8)
    r = (dist.log_prob(ab) - lpb).exp()
    loss = -torch.min(adv * r, adv * r.clamp(1 - CLIP, 1 + CLIP)).mean() + VF * (retb - v).pow(2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), MAXGRAD)
    opt.step()
    return loss


# ---------------------------------------------------------------- phase 1
def phase_b(args, ctx, V, state_bytes, emit, deadline):
    W_list = sorted({1, max(1, V // 4), max(1, V // 2), max(1, V - 1), V})
    if args.smoke:
        W_list = sorted({1, min(2, V)})
    for W in W_list:
        if time.time() + args.b_window + 15 > deadline:
            emit(dict(kind="b", W=W, skipped="deadline"))
            continue
        q, barrier = ctx.Queue(), ctx.Barrier(W + 1)
        ps = [ctx.Process(target=free_worker, args=(i, args.rom, state_bytes, barrier, args.b_window, q, 1000), daemon=True)
              for i in range(W)]
        for p in ps:
            p.start()
        barrier.wait(timeout=300)
        t0 = time.time()
        res = [q.get(timeout=args.b_window + 120) for _ in range(W)]
        for p in ps:
            p.join(timeout=20)
        wall = st.mean(r["wall"] for r in res)
        b_ms = [r["wall"] / r["steps"] * 1e3 for r in res]
        emit(dict(
            kind="b", W=W, window=wall, b_ms_mean=st.mean(b_ms), b_ms_max=max(b_ms), b_ms_min=min(b_ms),
            aggregate_sps=sum(r["steps"] / r["wall"] for r in res),
            cpu_over_wall=st.mean(r["cpu"] / r["wall"] for r in res),  # <1 under all-core load means throttling or steal
            loadavg=os.getloadavg(), t0=t0,
        ))


# ---------------------------------------------------------------- phase 2
def phase_gpu(args, dev, sampler, emit, deadline, batch_sizes, out):
    import torch

    Policy = build_models()
    sync = make_sync(dev)
    rng = np.random.default_rng(0)
    pool = rng.integers(0, 256, (max(batch_sizes + [2048]), *SHAPE), dtype=np.uint8)
    pc = time.perf_counter

    def to_x(a):
        return torch.from_numpy(a).to(dev).float().div_(255.0)

    for name in ("nature", "impala1", "impala4"):
        if time.time() + 20 > deadline:
            emit(dict(kind="gpu", model=name, skipped="deadline"))
            continue
        m = Policy(name).to(dev)
        opt = torch.optim.Adam(m.parameters(), LR, eps=1e-5)

        def infer(n, tf=None):
            t0 = pc()
            x = to_x(pool[:n])
            t1 = pc()
            with torch.no_grad():
                lo, v = m(x)
                d = torch.distributions.Categorical(logits=lo)
                a = d.sample()
                lp = d.log_prob(a)
            sync()
            t2 = pc()
            a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()
            t3 = pc()
            if tf is not None:
                tf.append((t1 - t0, t2 - t1, t3 - t2))

        rec: dict = dict(kind="gpu", model=name, params=sum(p.numel() for p in m.parameters()), infer={})
        per_point = 0.2 if args.smoke else 2.0
        for n in sorted(set(batch_sizes)):
            if name == "impala4" and n > 768:
                continue
            tf: list = []
            for _ in range(3):
                infer(n)
            t_end = time.time() + per_point
            while time.time() < t_end and len(tf) < 400:
                infer(n, tf)
            rec["infer"][n] = dict(
                total_ms=st.median(sum(x) for x in tf) * 1e3,
                to_device_ms=st.median(x[0] for x in tf) * 1e3,
                forward_sync_ms=st.median(x[1] for x in tf) * 1e3,
                to_cpu_ms=st.median(x[2] for x in tf) * 1e3, reps=len(tf),
            )
        # sparse bursts: does latency rise when the GPU idles between batches, as it does on MPS?
        n0 = 96
        tf = []
        t_end = time.time() + (0.3 if args.smoke else 4.0)
        while time.time() < t_end:
            infer(n0, tf)
            time.sleep(0.036)
        rec["infer_sparse_36ms_b96_ms"] = st.median(sum(x) for x in tf) * 1e3
        rec["update"] = {}
        for mb in ((512,) if args.smoke else (128, 512, 2048)):
            if name == "impala4" and mb > 512:
                continue
            xb = to_x(pool[:mb])
            ab = torch.from_numpy(rng.integers(0, N_ACTIONS, mb)).to(dev)
            lp, adv, ret = (torch.randn(mb, device=dev) for _ in range(3))
            ts: list[float] = []
            ppo_minibatch(m, opt, xb, ab, lp - 1.9, adv, ret)
            sync()
            t_end = time.time() + per_point
            while time.time() < t_end and len(ts) < 300:
                t0 = pc()
                # Fresh host-to-device copy per minibatch, as in the real loop.
                xb = to_x(pool[:mb])
                ppo_minibatch(m, opt, xb, ab, lp - 1.9, adv, ret)
                sync()
                ts.append(pc() - t0)
            rec["update"][mb] = dict(total_ms=st.median(ts) * 1e3, us_per_sample=st.median(ts) / mb * 1e6, reps=len(ts))
        # Saturated reading: tight update loop, then utilization, power and clock with the GPU known to be busy.
        xb = to_x(pool[:512])
        ab = torch.from_numpy(rng.integers(0, N_ACTIONS, 512)).to(dev)
        lp, adv, ret = (torch.randn(512, device=dev) for _ in range(3))
        t0_w = time.time()
        t_end = t0_w + (0.5 if args.smoke else args.sat_window)
        n_sat = 0
        while time.time() < t_end:
            ppo_minibatch(m, opt, xb, ab, lp - 1.9, adv, ret)
            sync()
            n_sat += 1
        rec["saturated_update"] = dict(
            minibatches_per_s=n_sat / (time.time() - t0_w), reading=sampler.window(t0_w, time.time()),
        )
        ref = M4_REF.get(name)
        if ref and 512 in rec["update"]:
            rec["g_update_vs_m4"] = ref["update_us_per_sample"] / rec["update"][512]["us_per_sample"]
        if ref and 96 in rec["infer"]:
            rec["g_infer_b96_vs_m4"] = ref["infer_b96_ms"] / rec["infer"][96]["total_ms"]
        if dev.type == "cuda":
            rec["vram_peak_gb"] = torch.cuda.max_memory_allocated(0) / 2**30
            torch.cuda.reset_peak_memory_stats(0)
        emit(rec)
        out[name] = rec
        del m, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        elif dev.type == "mps":
            torch.mps.empty_cache()


# ---------------------------------------------------------------- phase 3
def full_loop(args, ctx, dev, sampler, cfg, state_bytes, gpu_ref, emit, deadline):
    import torch

    label, name, W, k, T, epochs, mb = (cfg[x] for x in ("label", "model", "W", "k", "T", "epochs", "mb"))
    N, B = W * k, W * k * T
    limit = min(deadline, time.time() + args.cfg_max)
    if limit - time.time() < 20 and not args.smoke:
        emit(dict(kind="loop", **cfg, skipped="deadline"))
        return
    shm = sm.SharedMemory(create=True, size=shm_size(N, W))
    obs, act, rew, done, tim = views(shm.buf, N, W)
    pipes = [ctx.Pipe() for _ in range(W)]
    ps = [ctx.Process(target=lock_worker, args=(i, k, pipes[i][1], shm.name, N, W, args.rom, state_bytes), daemon=True)
          for i in range(W)]
    pc = time.perf_counter
    try:
        for p in ps:
            p.start()
        for x in pipes:
            x[1].close()  # so a dead worker gives the parent EOFError, not a hang until the global alarm
        conns = [x[0] for x in pipes]
        for c in conns:
            c.recv_bytes()
        Policy = build_models()
        m = Policy(name).to(dev)
        opt = torch.optim.Adam(m.parameters(), LR, eps=1e-5)
        sync = make_sync(dev)
        OB = np.zeros((T, N, *SHAPE), np.uint8)
        AC = np.zeros((T, N), np.int64)
        LP, VL, RW, ST = (np.zeros((T, N), np.float32) for _ in range(4))
        start = np.ones(N, np.float32)
        rng = np.random.default_rng(os.getpid())
        keys = ("to_device", "forward", "to_cpu", "buffer", "env", "gae", "update")
        tm = dict.fromkeys(keys, 0.0)
        t_begin = pc()
        win0 = None
        busy0 = None
        wall0 = 0.0
        steps = cycles = done_cycles = 0
        incomplete = False
        while True:
            for t in range(T):
                t0 = pc()
                x = torch.from_numpy(obs).to(dev).float().div_(255.0)
                t1 = pc()
                with torch.no_grad():
                    lo, v = m(x)
                    d = torch.distributions.Categorical(logits=lo)
                    a = d.sample()
                    lp = d.log_prob(a)
                sync()
                t2 = pc()
                a_np, lp_np, v_np = a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()
                t3 = pc()
                OB[t], AC[t], LP[t], VL[t], ST[t] = obs, a_np, lp_np, v_np, start
                t4 = pc()
                act[:] = a_np
                for c in conns:
                    c.send_bytes(b"s")
                for c in conns:
                    c.recv_bytes()
                t5 = pc()
                RW[t] = rew
                start = done.astype(np.float32)
                t6 = pc()
                tm["to_device"] += t1 - t0
                tm["forward"] += t2 - t1
                tm["to_cpu"] += t3 - t2
                tm["buffer"] += (t4 - t3) + (t6 - t5)
                tm["env"] += t5 - t4
            t0 = pc()
            with torch.no_grad():
                _, lv = m(torch.from_numpy(obs).to(dev).float().div_(255.0))
                lv = lv.cpu().numpy()
            ADV = np.zeros((T, N), np.float32)
            gae = np.zeros(N, np.float32)
            nv, nn_ = lv, 1.0 - start
            for t in reversed(range(T)):
                gae = RW[t] + GAMMA * nv * nn_ - VL[t] + GAMMA * LAM * nn_ * gae
                ADV[t] = gae
                nv, nn_ = VL[t], 1.0 - ST[t]
            RET = ADV + VL
            tm["gae"] += pc() - t0
            t0 = pc()
            fO, fA, fL, fD, fR = (OB.reshape(B, *SHAPE), AC.reshape(B), LP.reshape(B), ADV.reshape(B), RET.reshape(B))
            for _ in range(epochs):
                perm = rng.permutation(B)
                for s in range(0, B, mb):
                    j = perm[s : s + mb]
                    ppo_minibatch(
                        m, opt, torch.from_numpy(fO[j]).to(dev).float().div_(255.0), torch.from_numpy(fA[j]).to(dev),
                        torch.from_numpy(fL[j]).to(dev), torch.from_numpy(fD[j]).to(dev), torch.from_numpy(fR[j]).to(dev),
                    ).item()
                    if time.time() > limit + 30:
                        raise TimeUp
            sync()
            tm["update"] += pc() - t0
            cycles += 1
            now = pc()
            if win0 is None:
                if now - t_begin >= args.warmup and cycles >= 1:
                    tm = dict.fromkeys(keys, 0.0)
                    win0, busy0, wall0 = now, tim.copy(), time.time()
            else:
                steps += B
                done_cycles += 1
                if now - win0 >= args.window:
                    break
            if time.time() > limit:
                incomplete = win0 is None or done_cycles == 0
                break
        if win0 is None:
            emit(dict(kind="loop", **cfg, skipped=f"no window: {cycles} warmup cycles within limit"))
            return
        wall = pc() - win0
        if steps == 0:
            emit(dict(kind="loop", **cfg, incomplete=True, wall=wall))
            return
        lock = steps / N
        sps = steps / wall
        fr = {kk: tm[kk] / wall for kk in keys}
        b_ms = float((tim - busy0).mean()) / lock / k * 1e3
        # Capacity: saturated cost of the same work, per env step, times the achieved rate.
        cap = None
        g = gpu_ref.get(name) if gpu_ref else None
        if g:
            inf = g["infer"].get(N) or g["infer"].get(str(N))
            upd = g["update"].get(mb) or g["update"].get(512)
            if inf and upd:
                cap = sps * (inf["forward_sync_ms"] * 1e-3 / N + epochs * upd["us_per_sample"] * 1e-6)
        emit(dict(
            kind="loop", **cfg, N=N, cycles=done_cycles, wall=wall, sps=sps, frac=fr, b_ms_per_step_in_loop=b_ms,
            lockstep_ms=wall / lock * 1e3, gpu_call_frac=fr["forward"] + fr["update"], capacity_frac=cap,
            reading=sampler.window(wall0, time.time()), incomplete=incomplete, loadavg=os.getloadavg(),
        ))
    except TimeUp:
        emit(dict(kind="loop", **cfg, skipped="hit per-config time limit mid-update"))
    finally:
        for c in [x[0] for x in pipes]:
            try:
                c.send_bytes(b"q")
            except (BrokenPipeError, OSError):
                pass
        for p in ps:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        del obs, act, rew, done, tim
        shm.close()
        shm.unlink()


def selftest() -> None:
    assert parse_smi_line("37, 12, 118.51, 1410, 2048, P0") == dict(
        util=37.0, mem_util=12.0, power=118.51, sm_mhz=1410.0, vram_mb=2048.0, pstate="P0")
    r = parse_smi_line("[N/A], [N/A], [Not Supported], 1410, 2048, P0")
    assert r is not None and r["util"] is None and r["power"] is None and r["sm_mhz"] == 1410.0
    assert parse_smi_line("garbage") is None
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rom")
    ap.add_argument("--state")
    ap.add_argument("--out", default="pod_benchmark.jsonl")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--budget-min", type=float, default=18.0, help="hard limit for the whole run")
    ap.add_argument("--b-window", type=float, default=20.0)
    ap.add_argument("--sat-window", type=float, default=12.0)
    ap.add_argument("--warmup", type=float, default=8.0)
    ap.add_argument("--window", type=float, default=40.0)
    ap.add_argument("--cfg-max", type=float, default=150.0, help="time limit per loop config, seconds")
    ap.add_argument("--smoke", action="store_true", help="tiny windows and configs, to prove the code path runs")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--vcpus", type=int, help="override usable vCPUs when the allocation is not visible in cgroup or affinity")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.rom and args.state):
        ap.error("--rom and --state are required")
    if args.smoke:
        args.b_window, args.warmup, args.window, args.cfg_max = 1.5, 0.5, 1.0, 25.0

    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dev = torch.device(args.device)
    t_start = time.time()
    deadline = t_start + args.budget_min * 60
    torch.set_num_threads(2)

    # Work on a private ROM copy, so that nothing can ever appear next to the original.
    tmp = tempfile.mkdtemp(prefix="podbench_")
    sha = hashlib.sha1(open(args.rom, "rb").read()).hexdigest()
    rom_copy = os.path.join(tmp, "red.gb")
    shutil.copyfile(args.rom, rom_copy)
    args.rom = rom_copy
    state_bytes = open(args.state, "rb").read()

    fh = open(args.out, "a")

    def emit(rec: dict) -> None:
        rec = {"t_elapsed_s": round(time.time() - t_start, 1), **rec}
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()
        print(json.dumps(rec, default=str), flush=True)

    def on_alarm(signum, frame):
        raise TimeUp

    signal.signal(signal.SIGALRM, on_alarm)
    signal.signal(signal.SIGTERM, on_alarm)
    signal.alarm(int(args.budget_min * 60) + 60)  # backstop: ends the run even if a phase overruns its own checks

    sampler = UtilSampler(dev.type)
    ctx = mp.get_context("spawn")
    info = sysinfo(dev.type)
    info["rom_sha1_ok"] = sha == ROM_SHA1
    emit(info)
    V = args.vcpus or info["vcpus"]["usable"]
    emit(dict(kind="vcpus_used", V=V, override=bool(args.vcpus)))
    # One vCPU for the main process: W = V-1, floor 1, capped so a host-sized nproc cannot start hundreds of emulators.
    W_full = max(1, min(V - 1, 128))
    if args.smoke:
        W_full = 2

    cfgs = [
        dict(label="nature_k8", model="nature", k=8, T=128, epochs=3, mb=512),
        dict(label="impala1_k8", model="impala1", k=8, T=128, epochs=3, mb=512),
        dict(label="impala4_k8", model="impala4", k=8, T=64, epochs=3, mb=512),
        dict(label="nature_k1", model="nature", k=1, T=512, epochs=3, mb=512),
        dict(label="nature_k32", model="nature", k=32, T=64, epochs=3, mb=512),
        dict(label="nature_ep10", model="nature", k=8, T=128, epochs=10, mb=512),
        dict(label="impala1_k32", model="impala1", k=32, T=64, epochs=3, mb=512),
    ]
    for c in cfgs:
        c["W"] = W_full
        if args.smoke:
            c.update(k=min(c["k"], 2), T=4, mb=16, epochs=1)
    batch_sizes = sorted({12, 48, 96, 384, *(c["W"] * c["k"] for c in cfgs)})
    batch_sizes = [n for n in batch_sizes if n <= 4096]
    if args.smoke:
        batch_sizes = sorted({12, 96, *(c["W"] * c["k"] for c in cfgs)})

    gpu_ref: dict = {}

    def guarded(what, fn, *a):
        """A crash in one phase or config must not lose the later ones on a pod that is costing money."""
        try:
            fn(*a)
        except Exception as e:  # TimeUp is an Exception too, and has to end the run
            if isinstance(e, TimeUp):
                raise
            emit(dict(kind="error", where=what, error=f"{type(e).__name__}: {e}"[:500], tb=traceback.format_exc()[-1500:]))

    try:
        t = time.time()
        guarded("b", phase_b, args, ctx, min(V, 128), state_bytes, emit, deadline)
        emit(dict(kind="phase", name="b", seconds=time.time() - t))
        t = time.time()
        guarded("gpu", phase_gpu, args, dev, sampler, emit, deadline, batch_sizes, gpu_ref)
        emit(dict(kind="phase", name="gpu", seconds=time.time() - t))
        t = time.time()
        for c in cfgs:
            guarded(c["label"], full_loop, args, ctx, dev, sampler, c, state_bytes, gpu_ref, emit, deadline)
        emit(dict(kind="phase", name="loop", seconds=time.time() - t))
    except TimeUp:
        emit(dict(kind="abort", reason="global time limit or SIGTERM"))
    finally:
        signal.alarm(0)
        sampler.close()
        shutil.rmtree(tmp, ignore_errors=True)
        emit(dict(kind="end", total_s=time.time() - t_start, reminder="This script does not stop the pod. Stop it now."))
        fh.close()


if __name__ == "__main__":
    main()
