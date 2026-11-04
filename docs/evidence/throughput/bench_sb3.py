"""SB3 PPO + SubprocVecEnv on env_core.CoreEnv. Timing is by wrapping, not by rewriting SB3's loop.

Wrapped: VecEnv.step_async/step_wait (env), obs_as_tensor (xfer), policy.forward + mps.synchronize (fwd),
rollout_buffer.add (buf), compute_returns_and_advantage (gae), train (update). The rest of collect_rollouts
(.cpu().numpy() of actions/values/log_probs, callbacks, _update_info_buffer, last-value predict) cannot be separated
without rewriting it and is reported as 'collect_rest'; learn()'s own bookkeeping is 'other'.
--no-instrument keeps only the cycle-boundary timer, to measure the cost of the wrappers themselves.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from env_core import N_ACTIONS, SCREEN_SHAPE, STATE_DIM, VISIT_SHAPE, CoreEnv  # noqa: E402


class GymEnv(gym.Env):
    def __init__(self, fast=False, obs_mode="image"):
        self.core = CoreEnv(fast_mode=fast, obs_mode=obs_mode)
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        scr = gym.spaces.Box(0, 255, SCREEN_SHAPE, np.uint8)
        if obs_mode == "image":
            self.observation_space = scr
        else:
            self.observation_space = gym.spaces.Dict(
                screen=scr,
                visited=gym.spaces.Box(0, 255, VISIT_SHAPE, np.uint8),
                state=gym.spaces.Box(0.0, 1.0, (STATE_DIM,), np.float32),
            )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.core.reset(), {}

    def step(self, action):
        o, r, tr = self.core.step(int(action))
        return o, r, False, tr, {}

    def get_timers(self):
        return self.core.timers()

    def close(self):
        self.core.close()


def make(fast, obs_mode):
    def _f():
        return GymEnv(fast, obs_mode)

    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--obs", default="image", choices=["image", "dict"])
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--no-instrument", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--window", type=float, default=60.0)
    a = ap.parse_args()

    import torch
    import stable_baselines3.common.on_policy_algorithm as opa
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from common import emit, worker_mem

    torch.set_num_threads(a.threads)
    pc = time.perf_counter
    keys = ("env", "xfer", "fwd", "buf", "gae", "update", "collect")
    tm = dict.fromkeys(keys, 0.0)
    inst = not a.no_instrument

    class TimedVec(SubprocVecEnv):
        def step_async(self, actions):
            t = pc()
            super().step_async(actions)
            tm["env"] += pc() - t

        def step_wait(self):
            t = pc()
            r = super().step_wait()
            tm["env"] += pc() - t
            return r

    state = dict(t0=None, window_start=None, epoch_start=None, snap=None, steps0=0, cycles=0, win_cycles=0,
                 stop=False, result=None, paused=0.0)

    def timers(venv):
        return np.array(venv.env_method("get_timers"), dtype=np.float64)

    class TimedPPO(PPO):
        def collect_rollouts(self, *args, **kw):
            if state["stop"]:
                return False
            t = pc()
            r = super().collect_rollouts(*args, **kw)
            tm["collect"] += pc() - t
            return r

        def train(self):
            t = pc()
            super().train()
            if self.device.type == "mps":
                torch.mps.synchronize()
            tm["update"] += pc() - t
            state["cycles"] += 1
            now_ = pc()
            if state["window_start"] is None:
                if now_ - state["t0"] >= a.warmup and state["cycles"] >= (1 if a.T >= 1024 else 2):
                    state["snap"] = timers(self.env) if inst else None
                    for k in keys:
                        tm[k] = 0.0
                    state["steps0"] = self.num_timesteps
                    state["window_start"] = pc()
                    state["epoch_start"] = time.time()
            else:
                state["win_cycles"] += 1
                if now_ - state["window_start"] >= a.window:
                    state["stop"] = True
                    wall = now_ - state["window_start"]
                    state["result"] = (wall, self.num_timesteps - state["steps0"], dict(tm), time.time())

    venv = TimedVec([make(a.fast, a.obs) for _ in range(a.N)]) if inst else SubprocVecEnv(
        [make(a.fast, a.obs) for _ in range(a.N)]
    )
    policy = "CnnPolicy" if a.obs == "image" else "MultiInputPolicy"
    model = TimedPPO(
        policy, venv, device=a.device, n_steps=a.T, batch_size=a.mb, n_epochs=a.epochs, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, verbose=0,
    )
    params = sum(p.numel() for p in model.policy.parameters())

    if inst:
        orig_ot = opa.obs_as_tensor

        def t_ot(o, d):
            t = pc()
            r = orig_ot(o, d)
            tm["xfer"] += pc() - t
            return r

        opa.obs_as_tensor = t_ot
        orig_fwd = model.policy.forward

        def t_fwd(*x, **kw):
            t = pc()
            r = orig_fwd(*x, **kw)
            if model.device.type == "mps":
                torch.mps.synchronize()
            tm["fwd"] += pc() - t
            return r

        model.policy.forward = t_fwd
        rb = model.rollout_buffer
        orig_add, orig_gae = rb.add, rb.compute_returns_and_advantage

        def t_add(*x, **kw):
            t = pc()
            orig_add(*x, **kw)
            tm["buf"] += pc() - t

        def t_gae(*x, **kw):
            t = pc()
            orig_gae(*x, **kw)
            tm["gae"] += pc() - t

        rb.add, rb.compute_returns_and_advantage = t_add, t_gae

    state["t0"] = pc()
    model.learn(total_timesteps=10**12, log_interval=None)
    wall, steps, t, epoch_end = state["result"]
    lock_steps = steps / a.N
    rec = dict(
        label=a.label, loop="sb3", W=a.N, k=1, N=a.N, T=a.T, epochs=a.epochs, mb=a.mb, infer=a.device, update=a.device,
        conv="device", fast=a.fast, obs=a.obs, threads=a.threads, params=params, instrumented=inst,
        steps=steps, wall=wall, sps=steps / wall, cycles=state["win_cycles"], window_start=state["epoch_start"],
        window_end=epoch_end,
    )
    if inst:
        d = timers(venv) - state["snap"]
        emu, ob, rw = d[:, 0].mean() / lock_steps, d[:, 1].mean() / lock_steps, d[:, 2].mean() / lock_steps
        busy = emu + ob + rw
        env_per = t["env"] / lock_steps
        known = t["env"] + t["xfer"] + t["fwd"] + t["buf"] + t["gae"]
        rec["frac"] = dict(
            env=t["env"] / wall, xfer=t["xfer"] / wall, fwd=t["fwd"] / wall, buf=t["buf"] / wall, gae=t["gae"] / wall,
            update=t["update"] / wall, collect_rest=(t["collect"] - known) / wall,
            other=(wall - t["collect"] - t["update"]) / wall,
        )
        rec["ms_per_lockstep"] = dict(
            env=env_per * 1e3, emu=emu * 1e3, obs=ob * 1e3, rew=rw * 1e3, busy=busy * 1e3,
            sync_ipc=(env_per - busy) * 1e3,
        )
        rec["us_per_env_step"] = dict(emu=emu * 1e6, obs=ob * 1e6, rew=rw * 1e6)
        rec["resets"] = float(d[:, 4].sum())
    rec["mem"] = worker_mem([p.pid for p in venv.processes])
    emit(a.out, rec)
    venv.close()


if __name__ == "__main__":
    main()
