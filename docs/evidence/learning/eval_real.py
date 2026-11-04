"""One-off fixed comparison: 32 full episodes from the start state, budget 2048, trained policy (sampled) vs uniform random."""
import sys, numpy as np, torch
from functools import partial
from pathlib import Path
from pokered.checkpoints import CheckpointStore
from pokered.config import EnvConfig
from pokered.env import PokemonRedEnv
from pokered.model import ActorCritic
from pokered.observations import OBSERVATION_SHAPES
from pokered.train import pick_device
from pokered.vec import VecEnv
if __name__=="__main__":
    ck=Path(sys.argv[1]); d=pick_device(); env=EnvConfig(rom=Path("Pokemon Red.gb"),base_budget=2048,budget_per_event=512,stagger_first_episode=False)
    p=CheckpointStore(ck).load_latest(); m=ActorCritic(); m.load_state_dict(p["model"]); m.to(d); m.eval(); print("checkpoint steps",p["env_steps"])
    for name in ("random","trained"):
        v=VecEnv(partial(PokemonRedEnv,env),OBSERVATION_SHAPES,workers=8,envs_per_worker=4,seed=1000); v.reset(); E=[]; g=np.random.default_rng(0); torch.manual_seed(0)
        for _ in range(60000):
            if len(E)>=32: break
            if name=="random": a=g.integers(0,7,32)
            else:
                with torch.inference_mode(): a=torch.distributions.Categorical(logits=m({k:torch.from_numpy(x).to(d) for k,x in v.obs.items()})[0]).sample().cpu().numpy()
            E+=v.step(a).episodes
        v.close(); E=E[:32]
        for k in ("tiles","events","badges","level_sum","length"):
            x=np.array([e[k] for e in E]); print(name,k,"mean %.1f min %.0f median %.0f max %.0f"%(x.mean(),x.min(),np.median(x),x.max()))
