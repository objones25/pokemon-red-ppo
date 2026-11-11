"""Read RAM from run6 stall states: read_stall_states.py ROM RUNS_DIR PRET_DIR, with RUNS_DIR = runs/analysis."""
import io, json, re, sys
from pathlib import Path

from pyboy import PyBoy
from pokered import ram_map

S = Path(__file__).resolve().parent
P = Path(sys.argv[3])  # directory holding pret/pokered constants and charmap.asm
ROM = sys.argv[1]


def consts(path, start=0):
    names, i = {}, start
    for line in open(path):
        t = line.split(";")[0].split()
        if not t:
            continue
        if t[0] == "const_def":
            i = int(t[1].replace("$", "0x"), 0) if len(t) > 1 else 0
        elif t[0] == "const":
            names[i] = t[1]; i += 1
        elif t[0] == "const_skip":
            i += int(t[1]) if len(t) > 1 else 1
        elif t[0] == "const_next":
            i = int(t[1].replace("$", "0x"), 0)
    return names


SPECIES = consts(P / "pokemon_constants.asm")
MOVES = consts(P / "move_constants.asm")
ITEMS = consts(P / "item_constants.asm")
CHAR = {}
for line in open(P / "charmap.asm"):
    m = re.match(r'\s*charmap\s+"(.*?)",\s+\$([0-9a-fA-F]+)', line)
    if m and int(m.group(2), 16) not in CHAR:
        CHAR[int(m.group(2), 16)] = m.group(1)
CHAR[0x7F] = " "


def screen_text(mem):
    rows = []
    for r in range(18):
        s = "".join(CHAR.get(mem[ram_map.TILEMAP + 20 * r + c], "·") if mem[ram_map.TILEMAP + 20 * r + c] >= 0x7F or mem[ram_map.TILEMAP + 20 * r + c] in CHAR else "·" for c in range(20))
        rows.append(s)
    return rows


def mon(p):
    return {
        "species": SPECIES.get(p.species, p.species), "level": p.level, "hp": p.hp, "max_hp": p.max_hp,
        "moves": [MOVES.get(m, m) for m in p.moves if m], "pp": list(p.pp)[: sum(1 for m in p.moves if m)], "exp": p.exp,
    }




RUNS = Path(sys.argv[2])
out = []
pyboy = PyBoy(ROM, window="null", log_level="ERROR", sound_emulated=False)
for run in ("run6",):
    meta = {}
    for rp in (RUNS / run / "reports").glob("*.json"):
        for snap in json.load(open(rp))["snapshots"]:
            meta[snap["file"]] = snap
    for path in sorted((RUNS / run / "stuck").glob("*.state")):
        pyboy.load_state(io.BytesIO(path.read_bytes()))
        mem = pyboy.memory
        st = ram_map.read_state(mem)
        b = st.battle
        m = meta[path.stem]
        out.append({
            "run": run, "file": path.stem, "train_step": int(path.stem.split("_")[0]), "ep_step": m["step"],
            "map": st.map_id, "x": st.x, "y": st.y, "money": st.money, "badges": bin(st.badges).count("1"),
            "respawn": mem[0xD719], "facing": mem[0xC109], "d771": mem[0xD771], "joy_ignore": mem[0xCD6B], "last_map": mem[0xD365], "used_center": bool(mem[0xD72E] & 4),
            "party": [mon(p) for p in st.party],
            "bag": [(ITEMS.get(i, i), n) for i, n in st.bag],
            "battle": None if b is None else {
                "trainer_class": mem[ram_map.TRAINER_CLASS], "enemy": SPECIES.get(b.enemy.species), "enemy_level": b.enemy.level,
                "player": SPECIES.get(b.player.species), "player_hp": b.player.hp, "enemy_hp": b.enemy.hp,
            },
            "screen": screen_text(mem),
        })
pyboy.stop(save=False)
json.dump(out, open(RUNS / "states_run6.json", "w"), indent=1)
print(len(out))
