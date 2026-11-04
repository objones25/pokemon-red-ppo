"""Where does a lockstep rollout step's time go? Same parts as collect_rollout, timed separately."""
import json, sys, time
from functools import partial
from pathlib import Path
import numpy as np, torch
from pokered.affinity import pin_process, plan_for_this_machine
from pokered.config import EnvConfig
from pokered.env import PokemonRedEnv
from pokered.model import ActorCritic
from pokered.observations import OBSERVATION_SHAPES
from pokered.ppo import autocast, _to_device
from pokered.vec import VecEnv

if __name__ == "__main__":
    workers, k, threads, pin, steps, fast = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "pin", int(sys.argv[5]), sys.argv[6] == "fast"
    torch.set_num_threads(threads)
    d = torch.device("cuda")
    plan = plan_for_this_machine(workers, 0) if pin else None
    env = EnvConfig(rom=Path("/workspace/rom/Pokemon Red.gb"), fast_mode=fast)
    v = VecEnv(partial(PokemonRedEnv, env), OBSERVATION_SHAPES, workers=workers, envs_per_worker=k, seed=0, worker_cores=None if plan is None else plan.workers)
    if plan: pin_process(plan.main)
    v.reset()
    m = ActorCritic().to(d)
    t = dict(to_device=0.0, infer=0.0, sample=0.0, store=0.0, step=0.0)
    n = workers * k
    store = {key: torch.zeros((n, *s), dtype=torch.uint8, device=d) for key, s in OBSERVATION_SHAPES.items()}
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(steps):
            if i == 20:
                t = {key: 0.0 for key in t}; start = time.perf_counter()
            a = time.perf_counter(); obs = _to_device(v.obs, d); b = time.perf_counter()
            with autocast(d): logits, values = m(obs)
            c = time.perf_counter()
            dist = torch.distributions.Categorical(logits=logits.float()); actions = dist.sample()
            for key, value in obs.items(): store[key][:] = value
            e = time.perf_counter()
            np_actions = actions.cpu().numpy()
            f = time.perf_counter()
            v.step(np_actions)
            g = time.perf_counter()
            t["to_device"] += b - a; t["infer"] += c - b; t["store"] += e - c; t["sample"] += f - e; t["step"] += g - f
    total = time.perf_counter() - start
    used = steps - 20
    print(json.dumps({"workers": workers, "k": k, "threads": threads, "pin": pin, "fast": fast, "sps": round(n * used / total), "ms_per_step": round(1000 * total / used, 2), **{key: round(1000 * val / used, 2) for key, val in t.items()}}), flush=True)
    v.close()
