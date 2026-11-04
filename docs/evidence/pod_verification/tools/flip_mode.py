"""Copy the newest checkpoint with env.fast_mode set as asked, so the real pokered-evaluate runs it in that mode."""
import sys
from pathlib import Path
import torch
from pokered.checkpoints import CheckpointStore
source, destination, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "fast"
payload = CheckpointStore(source).load_latest()
payload["config"]["env"]["fast_mode"] = mode
CheckpointStore(destination, keep=1).save(payload, env_steps=payload["env_steps"])
print("wrote", destination, "steps", payload["env_steps"], "fast_mode", mode)
