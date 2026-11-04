"""Local learning check on the real game. Same train() as production, with a JSONL logger."""
import json, sys, torch
from functools import partial
from pathlib import Path
from pokered.checkpoints import CheckpointStore
from pokered.config import EnvConfig, PPOConfig, TrainConfig
from pokered.env import PokemonRedEnv
from pokered.model import ActorCritic
from pokered.observations import OBSERVATION_SHAPES
from pokered.train import pick_device, train
from pokered.vec import VecEnv
class L:
    def __init__(s,p): s.f=open(p,"a")
    def log(s,m,*,step): s.f.write(json.dumps({"step":step,**m})+"\n"); s.f.flush()
    def finish(s): s.f.close()
if __name__=="__main__":
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
    env=EnvConfig(rom=Path("Pokemon Red.gb"),base_budget=2048,budget_per_event=512)
    c=TrainConfig(env=env,ppo=PPOConfig(rollout_steps=256,minibatch_size=2048),workers=8,envs_per_worker=8,total_steps=3_000_000,run_dir=out,run_name="learn",wandb_mode="disabled",checkpoint_every_steps=500_000)
    torch.manual_seed(0); torch.set_num_threads(2)
    v=VecEnv(partial(PokemonRedEnv,env),OBSERVATION_SHAPES,workers=8,envs_per_worker=8,seed=0); v.reset()
    try: train(c,ActorCritic(),v,L(out/"log.jsonl"),CheckpointStore(out/"checkpoints"),pick_device())
    finally: v.close()
