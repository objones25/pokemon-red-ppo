"""Replay saved S.S. Anne rival fight starts with the frozen policy, as saved and with the party fully healed.

rival_fight_test.py ROM CKPT_DIR STATES_DIR TRIALS OUT_JSON. Analysis tool. Each trial restores the state, optionally
heals every party Pokemon (HP to max, status cleared, PP to base) and the active battle Pokemon, then lets the policy act
with sampled actions until the battle ends. Win = the battle ended without the lost-battle marker.
"""
import json, sys
from pathlib import Path

import torch

from pokered import ram_map
from pokered.checkpoints import CheckpointStore
from pokered.env import PokemonRedEnv
from pokered.evaluate import env_config_from_checkpoint
from pokered.model import ActorCritic

ROM, CKPT, STATES, TRIALS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
MOVES = ram_map.RomTables(ROM).moves
MAX_STEPS = 3000


def heal(mem):
    for slot in range(mem[ram_map.PARTY_COUNT]):
        a = ram_map.PARTY_MONS + slot * ram_map.MON_SIZE
        mx = mem[a + ram_map.OFF_MAX_HP] << 8 | mem[a + ram_map.OFF_MAX_HP + 1]
        mem[a + ram_map.OFF_HP], mem[a + ram_map.OFF_HP + 1] = mx >> 8, mx & 0xFF
        mem[a + ram_map.OFF_STATUS] = 0
        for i in range(4):
            move = mem[a + ram_map.OFF_MOVES + i]
            if move:
                mem[a + ram_map.OFF_PP + i] = MOVES[move].pp
        if slot == 0 and mem[0xD014] == mem[a]:  # the active battle copy, if the first Pokemon is out
            mem[0xD015], mem[0xD016] = mx >> 8, mx & 0xFF
            mem[0xD018] = 0
            for i in range(4):
                move = mem[0xD01C + i]
                if move:
                    mem[0xD02D + i] = MOVES[move].pp


payload = CheckpointStore(CKPT).load_latest()
model = ActorCritic(); model.load_state_dict(payload["model"]); model.eval(); torch.set_num_threads(1)
results = []
for path in sorted(STATES.glob("seed*_start_RIVAL2_2.state")):
    cfg = env_config_from_checkpoint(payload["config"]["env"], rom=ROM, state=path)
    env = PokemonRedEnv(cfg)
    for healed in (False, True):
        wins = 0
        for trial in range(TRIALS):
            gen = torch.Generator().manual_seed(trial)
            obs, _ = env.reset(seed=trial)
            mem = env.pyboy.memory
            if healed:
                heal(mem)
            for step in range(MAX_STEPS):
                with torch.inference_mode():
                    logits, _ = model({k: torch.from_numpy(v[None]) for k, v in obs.items()})
                obs, _, term, trunc, _ = env.step(int(torch.multinomial(logits.float().softmax(1), 1, generator=gen).item()))
                bt = mem[ram_map.BATTLE_TYPE]
                if bt == ram_map.BATTLE_LOST:
                    break
                if bt == ram_map.BATTLE_NONE and step > 5:
                    # The lost marker can arrive after the battle byte clears; a blackout also warps off the ship.
                    lost = False
                    for _ in range(60):
                        obs, _, term, trunc, _ = env.step(0)
                        if mem[ram_map.BATTLE_TYPE] == ram_map.BATTLE_LOST or mem[0xD35E] != 0x60:
                            lost = True
                            break
                    wins += not lost
                    break
                if term or trunc:
                    break
        results.append({"state": path.name, "healed": healed, "wins": wins, "trials": TRIALS})
        print(path.name, "healed" if healed else "as saved", f"{wins}/{TRIALS}", flush=True)
    env.close()
OUT.write_text(json.dumps(results, indent=1))
