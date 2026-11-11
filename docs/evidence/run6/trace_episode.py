"""Play one episode with a checkpoint and record what happened: trace_episode.py ROM CKPT_DIR PRET_DIR OUT_DIR SEED [--frames].

Analysis tool, not project code. Sampled actions (the emulator is deterministic). Records map changes, every fight with
the party before and after, event flags gained by name, reward payments, blackouts, respawn and bag changes, and saves a
.state at the start and end of every trainer fight on the S.S. Anne and every rival fight.
"""
import io, json, sys
from pathlib import Path

import numpy as np
import torch

from pokered import ram_map
from pokered.checkpoints import CheckpointStore
from pokered.env import PokemonRedEnv
from pokered.evaluate import env_config_from_checkpoint
from pokered.frame import read_frame
from pokered.model import ActorCritic
from pokered.start_state import START_STATE
from pokered.telemetry import opponent_label

ROM, CKPT, PRET, OUT, SEED = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
FRAMES = "--frames" in sys.argv
MAX_STEPS = 200_000


def consts(path, macro="const"):
    names, i = {}, 0
    for line in open(path):
        t = line.split(";")[0].split()
        if not t:
            continue
        if t[0] == "const_def":
            i = int(t[1].replace("$", "0x"), 0) if len(t) > 1 else 0
        elif t[0] in (macro, "const"):
            names[i] = t[1].rstrip(","); i += 1
        elif t[0] == "const_skip":
            i += int(t[1]) if len(t) > 1 else 1
        elif t[0] == "const_next":
            i = int(t[1].replace("$", "0x"), 0)
    return names


SPECIES = consts(PRET / "pokemon_constants.asm")
MOVES = consts(PRET / "move_constants.asm")
ITEMS = consts(PRET / "item_constants.asm")
EVENTS = consts(PRET / "event_constants.asm")
MAPS = {}
for i, l in enumerate([l.split()[1].rstrip(",") for l in open(PRET / "map_constants.asm") if l.split()[:1] == ["map_const"]]):
    MAPS[i] = l


def mon(p):
    n = sum(1 for m in p.moves if m)
    return {"species": SPECIES.get(p.species, p.species), "level": p.level, "hp": p.hp, "max_hp": p.max_hp,
            "moves": [MOVES.get(m, m) for m in p.moves if m], "pp": list(p.pp)[:n]}


def party(st):
    return [mon(p) for p in st.party]


payload = CheckpointStore(CKPT).load_latest()
env_config = env_config_from_checkpoint(payload["config"]["env"], rom=ROM, state=START_STATE)
env = PokemonRedEnv(env_config)
model = ActorCritic()
model.load_state_dict(payload["model"])
model.eval()
torch.set_num_threads(1)
gen = torch.Generator().manual_seed(SEED)

obs, _ = env.reset(seed=SEED)
mem = env.pyboy.memory
st = ram_map.read_state(mem)
fr = read_frame(mem)
log = {"seed": SEED, "env_steps": payload["env_steps"], "events": [], "start_party": party(st)}
ev = log["events"]
flags_prev = int.from_bytes(fr.event_flags, "little")
in_battle, fight, prev_map, prev_resp = False, None, st.map_id, mem[0xD719]
prev_bag = dict(st.bag)
totals = {}
frames = []
OUT.mkdir(parents=True, exist_ok=True)


def save_state(tag):
    buf = io.BytesIO(); env.pyboy.save_state(buf)
    (OUT / f"seed{SEED}_{tag}.state").write_bytes(buf.getvalue())


for step in range(1, MAX_STEPS + 1):
    with torch.inference_mode():
        logits, value = model({k: torch.from_numpy(v[None]) for k, v in obs.items()})
    action = int(torch.multinomial(logits.float().softmax(1), 1, generator=gen).item())
    obs, reward, terminated, truncated, info = env.step(action)
    if FRAMES:
        frames.append(env.pyboy.screen.ndarray[:, :, :3].copy())
    st = ram_map.read_state(mem)
    fr = read_frame(mem)
    for k, v in info["reward_terms"].items():
        if v:
            totals[k] = totals.get(k, 0.0) + v
            if k not in ("explore",):
                ev.append({"step": step, "kind": "reward", "term": k, "value": round(v, 4), "map": MAPS.get(st.map_id)})
    flags = int.from_bytes(fr.event_flags, "little")
    new = flags & ~flags_prev
    if new:
        for bit in range(len(fr.event_flags) * 8):
            if new >> bit & 1:
                ev.append({"step": step, "kind": "flag", "flag": EVENTS.get(bit, hex(bit)), "map": MAPS.get(st.map_id)})
    flags_prev |= flags
    if st.map_id != prev_map:
        ev.append({"step": step, "kind": "map", "from": MAPS.get(prev_map), "to": MAPS.get(st.map_id), "xy": [st.x, st.y]})
        prev_map = st.map_id
    if mem[0xD719] != prev_resp:
        ev.append({"step": step, "kind": "respawn", "to": MAPS.get(mem[0xD719])})
        prev_resp = mem[0xD719]
    bag = dict(st.bag)
    if bag != prev_bag:
        diff = {ITEMS.get(i, i): bag.get(i, 0) - prev_bag.get(i, 0) for i in set(bag) | set(prev_bag) if bag.get(i, 0) != prev_bag.get(i, 0)}
        ev.append({"step": step, "kind": "bag", "change": diff, "money": st.money, "map": MAPS.get(st.map_id)})
        prev_bag = bag
    battling = fr.battle_type not in (ram_map.BATTLE_NONE, ram_map.BATTLE_LOST)
    if battling and not in_battle:
        label = opponent_label(fr.battle_type, fr.opponent, fr.trainer_number)
        enemy = st.battle.enemy if st.battle else None
        fight = {"step": step, "kind": "fight", "opponent": label, "map": MAPS.get(st.map_id), "xy": [st.x, st.y],
                 "party_before": party(st), "first_enemy": None if enemy is None else
                 {"species": SPECIES.get(enemy.species), "level": enemy.level}, "turns": 0, "enemies_seen": []}
        ev.append(fight)
        if label.startswith("RIVAL") or (st.map_id >= 95 and st.map_id <= 105 and not label.startswith("wild")):
            save_state(f"{step}_start_{label.replace('#', '_')}")
    if battling and st.battle:
        e = (SPECIES.get(st.battle.enemy.species), st.battle.enemy.level)
        if e not in [tuple(x) for x in fight["enemies_seen"]]:
            fight["enemies_seen"].append(list(e))
        fight["turns"] = max(fight["turns"], st.battle.turn)
        fight["active"] = SPECIES.get(st.battle.player.species)
    if in_battle and not battling:
        fight["end_step"] = step
        fight["party_after"] = party(st)
        fight["blackout"] = fr.battle_type == ram_map.BATTLE_LOST or mem[0xD057] == 0xFF
        if fight["opponent"].startswith("RIVAL") or (95 <= st.map_id <= 105 and not fight["opponent"].startswith("wild")):
            save_state(f"{step}_end_{fight['opponent'].replace('#', '_')}")
    if fr.battle_type == ram_map.BATTLE_LOST and fight is not None and "blackout_marked" not in fight:
        fight["blackout"] = True; fight["blackout_marked"] = True
        ev.append({"step": step, "kind": "blackout", "after": fight["opponent"]})
    in_battle = battling
    if terminated or truncated:
        break

log.update({"steps": step, "terminated": terminated, "truncated": truncated, "end_map": MAPS.get(st.map_id),
            "end_xy": [st.x, st.y], "end_party": party(st), "end_bag": {ITEMS.get(i, i): n for i, n in st.bag},
            "money": st.money, "reward_totals": totals, "respawn": MAPS.get(mem[0xD719])})
save_state("end")
(OUT / f"seed{SEED}.json").write_text(json.dumps(log, indent=1))
if FRAMES:
    np.save(OUT / f"seed{SEED}_frames.npy", np.stack(frames))
env.close()
print(SEED, step, "terminated" if terminated else "truncated", MAPS.get(st.map_id), {k: round(v, 1) for k, v in totals.items()})
