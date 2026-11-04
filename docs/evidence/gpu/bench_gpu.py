"""Custom synchronous PPO loop on env_core.CoreEnv. W worker processes x k envs per worker, shared memory for
obs/reward/done/action/timers, one-byte pipe messages for lockstep. Workers never import torch.

Measures whole update cycles only: warmup >= --warmup s and >= 2 cycles, window >= --window s, ends on a cycle boundary.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import shared_memory as sm

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from env_core import N_ACTIONS, SCREEN_SHAPE, CoreEnv  # noqa: E402

FRAME_BYTES = int(np.prod(SCREEN_SHAPE))
GAMMA, LAM, CLIP, VF, MAXGRAD, LR = 0.99, 0.95, 0.2, 0.5, 0.5, 3e-4


def views(buf, N, W):
    off = 0
    obs = np.ndarray((N, *SCREEN_SHAPE), np.uint8, buf, off)
    off += N * FRAME_BYTES
    act = np.ndarray((N,), np.int64, buf, off)
    off += 8 * N
    rew = np.ndarray((N,), np.float32, buf, off)
    off += 4 * N
    done = np.ndarray((N,), np.uint8, buf, off)
    off += N + (-N % 8)
    tim = np.ndarray((W, 6), np.float64, buf, off)
    return obs, act, rew, done, tim


def size_of(N, W):
    return N * FRAME_BYTES + 8 * N + 4 * N + N + (-N % 8) + 8 * 6 * W


def worker(i, k, conn, name, N, W, fast, null_env):
    s = sm.SharedMemory(name=name)
    obs, act, rew, done, tim = views(s.buf, N, W)
    lo = i * k
    envs = [CoreEnv(fast_mode=fast) for _ in range(k)]
    for j, e in enumerate(envs):
        obs[lo + j] = e.reset()
    conn.send_bytes(b"r")
    busy = 0.0
    try:
        while True:
            msg = conn.recv_bytes()
            if msg == b"q":
                break
            t0 = time.perf_counter()
            for j, e in enumerate(envs):
                o, r, tr = e.step(act[lo + j])
                if tr:
                    o = e.reset()
                obs[lo + j] = o
                rew[lo + j] = r
                done[lo + j] = tr
            busy += time.perf_counter() - t0
            ts = [e.timers() for e in envs]
            tim[i] = (
                sum(t[0] for t in ts),
                sum(t[1] for t in ts),
                sum(t[2] for t in ts),
                sum(t[3] for t in ts),
                sum(t[4] for t in ts),
                busy,
            )
            conn.send_bytes(b"d")
    finally:
        for e in envs:
            e.close()
        del obs, act, rew, done, tim
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="nature")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--infer", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--update", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--conv", default="device", choices=["device", "cpu"], help="where uint8->float happens")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-policy", action="store_true", help="lockstep only: random actions, no net, no update")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--window", type=float, default=60.0)
    a = ap.parse_args()

    import torch
    import torch.nn as nn

    from common import emit, worker_mem

    torch.set_num_threads(a.threads)
    W, k = a.workers, a.k
    N, T = W * k, a.T
    B = N * T
    ctx = mp.get_context("spawn")
    shm = sm.SharedMemory(create=True, size=size_of(N, W))
    obs_s, act_s, rew_s, done_s, tim_s = views(shm.buf, N, W)
    pipes = [ctx.Pipe() for _ in range(W)]
    procs = [
        ctx.Process(target=worker, args=(i, k, pipes[i][1], shm.name, N, W, a.fast, False), daemon=True)
        for i in range(W)
    ]
    for p in procs:
        p.start()
    conns = [p[0] for p in pipes]
    for c in conns:
        c.recv_bytes()

    from models import Policy

    def Net():
        return Policy(a.model, N_ACTIONS, SCREEN_SHAPE)

    upd, inf = torch.device(a.update), torch.device(a.infer)
    model = Net().to(upd)
    infer_m = model if a.infer == a.update else Net().to(inf)
    if infer_m is not model:
        infer_m.load_state_dict(model.state_dict())
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), LR, eps=1e-5)

    def dev_sync(d):
        if d.type == "mps":
            torch.mps.synchronize()

    OB = np.zeros((T, N, *SCREEN_SHAPE), np.uint8)
    AC = np.zeros((T, N), np.int64)
    LP = np.zeros((T, N), np.float32)
    VL = np.zeros((T, N), np.float32)
    RW = np.zeros((T, N), np.float32)
    ST = np.zeros((T, N), np.float32)
    start_flag = np.ones(N, np.float32)
    keys = ("env", "xfer", "fwd", "tocpu", "buf", "gae", "update", "wsync", "other")
    tm = dict.fromkeys(keys, 0.0)
    rng = np.random.default_rng(os.getpid())
    pc = time.perf_counter

    def env_step(acts):
        act_s[:] = acts
        for c in conns:
            c.send_bytes(b"s")
        for c in conns:
            c.recv_bytes()

    t_begin = pc()
    cycles = 0
    window_start = None
    win_steps = 0
    snap_workers = None
    epoch_start = None
    cycles_in_window = 0

    while True:
        cyc0 = pc()
        tsum0 = sum(tm.values())
        for t in range(T):
            if a.no_policy:
                t0 = pc()
                env_step(rng.integers(0, N_ACTIONS, N))
                tm["env"] += pc() - t0
                continue
            t0 = pc()
            if a.conv == "device" or a.infer == "cpu":
                x = torch.from_numpy(obs_s).to(inf).float().div_(255.0)
            else:
                x = torch.from_numpy(obs_s).float().div_(255.0).to(inf)
            t1 = pc()
            with torch.no_grad():
                logits, v = infer_m(x)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                lp = dist.log_prob(acts)
                dev_sync(inf)
            t2 = pc()
            acts_np = acts.cpu().numpy()
            lp_np = lp.cpu().numpy()
            v_np = v.cpu().numpy()
            t3 = pc()
            OB[t] = obs_s
            AC[t] = acts_np
            LP[t] = lp_np
            VL[t] = v_np
            ST[t] = start_flag
            t4 = pc()
            env_step(acts_np)
            t5 = pc()
            RW[t] = rew_s
            start_flag = done_s.astype(np.float32)
            t6 = pc()
            tm["xfer"] += t1 - t0
            tm["fwd"] += t2 - t1
            tm["tocpu"] += t3 - t2
            tm["buf"] += (t4 - t3) + (t6 - t5)
            tm["env"] += t5 - t4

        if not a.no_policy:
            t0 = pc()
            with torch.no_grad():
                if a.conv == "device" or a.infer == "cpu":
                    x = torch.from_numpy(obs_s).to(inf).float().div_(255.0)
                else:
                    x = torch.from_numpy(obs_s).float().div_(255.0).to(inf)
                _, last_v = infer_m(x)
                last_v = last_v.cpu().numpy()
            ADV = np.zeros((T, N), np.float32)
            gae = np.zeros(N, np.float32)
            nextv, nextnonterm = last_v, 1.0 - start_flag
            for t in reversed(range(T)):
                delta = RW[t] + GAMMA * nextv * nextnonterm - VL[t]
                gae = delta + GAMMA * LAM * nextnonterm * gae
                ADV[t] = gae
                nextv, nextnonterm = VL[t], 1.0 - ST[t]
            RET = ADV + VL
            tm["gae"] += pc() - t0

            t0 = pc()
            fOB = OB.reshape(B, *SCREEN_SHAPE)
            fAC, fLP, fADV, fRET = AC.reshape(B), LP.reshape(B), ADV.reshape(B), RET.reshape(B)
            for _ in range(a.epochs):
                perm = rng.permutation(B)
                for s in range(0, B, a.mb):
                    j = perm[s : s + a.mb]
                    xb = torch.from_numpy(fOB[j]).to(upd).float().div_(255.0)
                    ab = torch.from_numpy(fAC[j]).to(upd)
                    lpb = torch.from_numpy(fLP[j]).to(upd)
                    advb = torch.from_numpy(fADV[j]).to(upd)
                    retb = torch.from_numpy(fRET[j]).to(upd)
                    logits, v = model(xb)
                    dist = torch.distributions.Categorical(logits=logits)
                    nlp = dist.log_prob(ab)
                    advb = (advb - advb.mean()) / (advb.std() + 1e-8)
                    ratio = (nlp - lpb).exp()
                    pl = -torch.min(advb * ratio, advb * ratio.clamp(1 - CLIP, 1 + CLIP)).mean()
                    vl = (retb - v).pow(2).mean()
                    loss = pl + VF * vl
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), MAXGRAD)
                    opt.step()
                    loss.item()
            dev_sync(upd)
            tm["update"] += pc() - t0
            if infer_m is not model:
                t0 = pc()
                infer_m.load_state_dict(model.state_dict())
                tm["wsync"] += pc() - t0

        cycles += 1
        now_ = pc()
        if window_start is None:
            if now_ - t_begin >= a.warmup and cycles >= (1 if T >= 1024 else 2):
                tm = dict.fromkeys(keys, 0.0)
                snap_workers = tim_s.copy()
                window_start = now_
                epoch_start = time.time()
        else:
            win_steps += B
            cycles_in_window += 1
            if now_ - window_start >= a.window:
                break

    wall = pc() - window_start
    epoch_end = time.time()
    dw = tim_s - snap_workers
    lock_steps = win_steps / N
    emu_per_lock = dw[:, 0].mean() / lock_steps
    obs_per_lock = dw[:, 1].mean() / lock_steps
    rew_per_lock = dw[:, 2].mean() / lock_steps
    busy_per_lock = dw[:, 5].mean() / lock_steps
    env_per_lock = tm["env"] / lock_steps
    tm["other"] = wall - sum(v for kk, v in tm.items() if kk != "other")
    mem = worker_mem([p.pid for p in procs])
    rec = dict(
        label=a.label, model=a.model, loop="custom", W=W, k=k, N=N, T=T, epochs=a.epochs, mb=a.mb, infer=a.infer, update=a.update,
        conv=a.conv, fast=a.fast, obs="image", threads=a.threads, params=params,
        steps=win_steps, wall=wall, sps=win_steps / wall, cycles=cycles_in_window,
        frac={kk: v / wall for kk, v in tm.items()},
        ms_per_lockstep=dict(
            env=env_per_lock * 1e3, emu=emu_per_lock * 1e3, obs=obs_per_lock * 1e3, rew=rew_per_lock * 1e3,
            busy=busy_per_lock * 1e3, sync_ipc=(env_per_lock - busy_per_lock) * 1e3,
        ),
        us_per_env_step=dict(emu=emu_per_lock / k * 1e6, obs=obs_per_lock / k * 1e6, rew=rew_per_lock / k * 1e6),
        resets=float(dw[:, 4].sum()),
        window_start=epoch_start, window_end=epoch_end, mem=mem,
    )
    emit(a.out, rec)

    for c in conns:
        c.send_bytes(b"q")
    for p in procs:
        p.join(timeout=10)
    del obs_s, act_s, rew_s, done_s, tim_s
    shm.close()
    shm.unlink()


if __name__ == "__main__":
    main()
