"""Accurate vs fast_mode fidelity. Subcommands: scenarios, policy, interchange. Pod-only evidence code; no ROM or state is written out.

Every step compares WRAM + HRAM (RNG lives in HRAM at FFD3/FFD4), map/x/y, the battle byte, event flags, OAM, VRAM and the screen.
--resync loads the accurate state into fast after any WRAM/HRAM mismatch, so one divergence does not hide later ones.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from lockstep import MAP_ID, RNG, Pair, as_line, load, mem, place, save, write_png, WRAM, HRAM
from pokered.buttons import BUTTONS, FRAMES_PER_ACTION, HOLD_FRAMES
from pokered.checkpoints import CheckpointStore
from pokered.emulator import open_emulator
from pokered.model import ActorCritic
from pokered.start_state import START_STATE

A, B, START, DOWN, LEFT, RIGHT, UP = 4, 5, 6, 0, 1, 2, 3
FLY_MAP, FLAGS_D732, NUM_WARPS, WARPS = 0xD71A, 0xD732, 0xD3AE, 0xD3AF
OPPONENT, ENEMY_LEVEL, TRAINER_NO, BATTLE = 0xD059, 0xD127, 0xD05D, 0xD057
PARTY_HP0, PARTY_MAXHP0 = 0xD16C, 0xD18D
CITIES = {"pallet": 0, "viridian": 1, "pewter": 2, "cerulean": 3, "lavender": 4, "vermilion": 5, "celadon": 6, "fuchsia": 7, "cinnabar": 8, "indigo": 9, "saffron": 10}
# (name, map id, warp index), from pret/pokered map constants.
DUNGEONS = [("viridian_forest", 0x33, 1), ("mt_moon", 0x3B, 0), ("rock_tunnel", 0x52, 0), ("pewter_gym", 0x36, 0), ("ss_anne", 0x5F, 0),
            ("pokemon_tower", 0x8E, 0), ("safari_center", 0xDC, 0), ("silph_co", 0xB5, 0), ("mansion", 0xA5, 0), ("victory_road", 0x6C, 0),
            ("seafoam", 0xC0, 0), ("rocket_hideout", 0xC7, 0), ("power_plant", 0x53, 0), ("celadon_mart", 0x7A, 0)]
WILD = [0xA5, 0x24, 0x7B, 0x70, 0x54, 0x03, 0x05, 0x6D, 0x60, 0x19]
TRAINERS = [(201, 1), (202, 1), (203, 1), (204, 1), (225, 1), (234, 1), (205, 2), (206, 1)]


class Frames:
    """Replays one decision frame by frame, rendering all of them, to tell what kind of screen difference this is."""

    def __init__(self, rom: Path) -> None:
        self.a, self.b = open_emulator(rom), open_emulator(rom, fast_mode=True)

    def run(self, emu, state: bytes, action: int) -> list[np.ndarray]:
        load(emu, state)
        frames = []
        emu.button_press(BUTTONS[action])
        for i in range(FRAMES_PER_ACTION + 2):
            if i == HOLD_FRAMES:
                emu.button_release(BUTTONS[action])
            emu.tick(1, True, False)
            frames.append(emu.screen.ndarray[:, :, 0].copy())
        return frames

    def classify(self, previous: tuple[bytes, bytes], action: int) -> dict:
        fa, fb = self.run(self.a, previous[0], action), self.run(self.b, previous[1], action)
        last = FRAMES_PER_ACTION - 1
        same = lambda x, y: bool((x == y).all())  # noqa: E731
        return {
            "fast_final_eq_acc_prev": same(fb[last], fa[last - 1]),
            "fast_final_eq_acc_final": same(fb[last], fa[last]),
            "fast_final_eq_acc_next": same(fb[last], fa[last + 1]),
            "fast_final_eq_acc_next2": same(fb[last], fa[last + 2]),
            "acc_final_eq_fast_next": same(fa[last], fb[last + 1]),
            "acc_next_pixels": int((fb[last] != fa[last + 1]).sum()),
            "frames_differing_of_24": sum(not same(x, y) for x, y in zip(fa[:24], fb[:24], strict=True)),
        }

    def close(self) -> None:
        self.a.stop(save=False)
        self.b.stop(save=False)


class Log:
    def __init__(self, pair: Pair, *, resync: bool, frames: Frames | None, screenshots: Path | None, model=None, name: str = "") -> None:
        self.pair, self.resync, self.frames, self.shots, self.model, self.name = pair, resync, frames, screenshots, model, name
        self.steps = self.battle = self.window = 0
        self.first: dict[str, dict] = {}
        self.divergences: list[dict] = []
        self.non_rng = 0
        self.diff_addr: collections.Counter = collections.Counter()
        self.pixels: list[int] = []
        self.bbox_full = 0
        self.in_sprites: list[float] = []
        self.classes: collections.Counter = collections.Counter()
        self.obs_screen = self.obs_other = self.place_diff = self.events_diff = self.vram_steps = self.oam_steps = 0
        self.tv: list[float] = []
        self.argmax_flip = 0
        self.value_gap: list[float] = []
        self.step_kinds: collections.Counter = collections.Counter()
        self.pix_in = collections.Counter()
        self.maps: set[int] = set()
        self.map_changes = 0
        self.previous: tuple[bytes, bytes] | None = None
        self.neighbour_pixels: list[int] = []

    def act(self, action: int) -> dict:
        a, b = self.pair.emus
        if self.frames is not None:
            self.previous = (save(a), save(b))
        before = a.memory[MAP_ID]
        self.pair.step(action)
        report = self.pair.compare()
        self.steps += 1
        now = a.memory[MAP_ID]
        self.maps.add(now)
        self.map_changes += now != before
        in_battle = a.memory[BATTLE] != 0
        self.battle += in_battle
        windowed = bool(a.memory[0xFF40] & 0x20) and a.memory[0xFF4A] < 144
        self.window += windowed
        kind = "battle" if in_battle else ("window" if windowed else "overworld")
        self.step_kinds[kind] += 1
        self.place_diff += not report["place_equal"]
        self.events_diff += not report["events_equal"]
        self.vram_steps += report["vram"] > 0
        self.oam_steps += report["oam"] > 0
        self.obs_screen += report["obs_screen_differs"]
        self.obs_other += report["obs_other_differs"]
        for name, key in (("place", not report["place_equal"]), ("events", not report["events_equal"])):
            if key and name not in self.first:
                self.first[name] = {"step": self.steps, "a": report["place_a"], "b": report["place_b"]}
        if report["pixels"]:
            self.pixels.append(report["pixels"])
            self.pix_in[kind] += 1
            self.bbox_full += report["pixels"] > 0.5 * 144 * 160
            self.in_sprites.append(report["screen"]["in_sprites"])
            if self.frames is not None and self.previous is not None and sum(self.classes.values()) < 300:
                verdict = self.frames.classify(self.previous, action)
                if verdict["fast_final_eq_acc_final"]:
                    self.classes["replay_equal"] += 1
                elif verdict["fast_final_eq_acc_next"]:
                    self.classes["fast_equals_acc_one_frame_later"] += 1
                elif verdict["fast_final_eq_acc_next2"]:
                    self.classes["fast_equals_acc_two_frames_later"] += 1
                elif verdict["fast_final_eq_acc_prev"]:
                    self.classes["fast_equals_acc_one_frame_earlier"] += 1
                else:
                    self.classes["neither_neighbour"] += 1
                    self.neighbour_pixels.append(verdict["acc_next_pixels"])
            if self.shots is not None and len(self.pixels) <= 6:
                write_png(self.shots / f"{self.name}_{len(self.pixels)}_acc.png", a.screen.ndarray[:, :, 0])
                write_png(self.shots / f"{self.name}_{len(self.pixels)}_fast.png", b.screen.ndarray[:, :, 0])
        if self.model is not None:
            self._compare_policy()
        addresses = report["wram"] + report["hram"]
        if addresses:
            self.diff_addr.update(addresses)
            rng_only = set(addresses) <= RNG
            self.non_rng += not rng_only
            for name, present in (("any", True), ("rng", bool(set(addresses) & RNG)), ("non_rng", not rng_only)):
                if present and name not in self.first:
                    self.first[name] = self._describe(report, addresses)
            if len(self.divergences) < 200:
                self.divergences.append(self._describe(report, addresses))
            if self.resync:
                load(b, save(a))
        return report

    def _describe(self, report: dict, addresses: list[int]) -> dict:
        a, b = self.pair.emus
        return {"step": self.steps, "count": len(addresses), "addresses": [hex(x) for x in addresses[:24]],
                "a": [a.memory[x] for x in addresses[:24]], "b": [b.memory[x] for x in addresses[:24]],
                "place_a": report["place_a"], "place_b": report["place_b"], "ly": report["ly"], "name": self.name}

    def _compare_policy(self) -> None:
        obs_a, obs_b = self.pair.obs_a, self.pair.obs_b
        same = all((obs_a[k] == obs_b[k]).all() for k in obs_a)
        if same:
            return
        with torch.inference_mode():
            la, va = self.model({k: torch.from_numpy(v[None]) for k, v in obs_a.items()})
            lb, vb = self.model({k: torch.from_numpy(v[None]) for k, v in obs_b.items()})
        pa, pb = la.softmax(1)[0], lb.softmax(1)[0]
        self.tv.append(float(0.5 * (pa - pb).abs().sum()))
        self.argmax_flip += int(pa.argmax() != pb.argmax())
        self.value_gap.append(float((va - vb).abs()))

    def summary(self) -> dict:
        a, b = self.pair.emus
        pix = np.array(self.pixels or [0])
        tv = np.array(self.tv or [0.0])
        return {
            "name": self.name, "resync": self.resync, "steps": self.steps, "battle_steps": self.battle, "window_steps": self.window,
            "step_kinds": dict(self.step_kinds), "maps_visited": sorted(self.maps), "map_changes": self.map_changes,
            "divergence_steps": len(self.divergences) if self.resync else None, "non_rng_divergence_steps": self.non_rng,
            "first": self.first, "first_divergences": self.divergences[:5],
            "top_diff_addresses": [[hex(k), v] for k, v in self.diff_addr.most_common(25)],
            "place_diff_steps": self.place_diff, "event_diff_steps": self.events_diff, "vram_diff_steps": self.vram_steps, "oam_diff_steps": self.oam_steps,
            "screen": {"diff_steps": len(self.pixels), "share": len(self.pixels) / max(self.steps, 1), "by_kind": dict(self.pix_in),
                       "pixels_median": float(np.median(pix)), "pixels_p95": float(np.percentile(pix, 95)), "pixels_max": int(pix.max()),
                       "steps_over_half_frame": self.bbox_full, "mean_share_inside_sprites": float(np.mean(self.in_sprites or [0.0])),
                       "classes": dict(self.classes), "neither_pixels_vs_acc_next_median": float(np.median(self.neighbour_pixels or [0]))},
            "obs": {"screen_diff_steps": self.obs_screen, "other_diff_steps": self.obs_other},
            "policy": {"compared_steps": len(self.tv), "tv_mean": float(tv.mean()), "tv_median": float(np.median(tv)), "tv_p95": float(np.percentile(tv, 95)),
                       "tv_max": float(tv.max()), "argmax_flips": self.argmax_flip, "value_gap_mean": float(np.mean(self.value_gap or [0.0]))},
            "reward": [self.pair.reward_a, self.pair.reward_b],
            "tiles": [self.pair.a._tiles.gained, self.pair.b._tiles.gained], "events": [self.pair.a._events.gained, self.pair.b._events.gained],
            "final_place": [place(a), place(b)],
            "final_wram_equal": mem(a, WRAM) == mem(b, WRAM), "final_hram_equal": mem(a, HRAM) == mem(b, HRAM),
        }


def wait_overworld(log: Log, cap: int = 40) -> None:
    a = log.pair.a.pyboy
    for _ in range(cap):
        if a.memory[BATTLE] == 0:
            return
        log.act(A)


def fly(log: Log, destination: int) -> bool:
    a = log.pair.a.pyboy
    wait_overworld(log, 300)
    for _ in range(8):
        log.act(B)
    log.pair.write(FLY_MAP, destination)
    log.pair.write(FLAGS_D732, a.memory[FLAGS_D732] | 0x08)
    for _ in range(60):
        log.act(B)
        if a.memory[FLAGS_D732] & 0x18 == 0 and a.memory[MAP_ID] == destination:
            return True
    return False


def enter_dungeon(log: Log, map_id: int, warp: int) -> bool:
    """Fly to Viridian, point every warp on the map at the target, and walk through the Pokemon Center door."""
    a = log.pair.a.pyboy
    for _ in range(3):
        if not fly(log, CITIES["viridian"]):
            continue
        for i in range(a.memory[NUM_WARPS]):
            log.pair.write(WARPS + 4 * i + 2, warp)
            log.pair.write(WARPS + 4 * i + 3, map_id)
        for _ in range(20):
            log.act(UP)
            if a.memory[MAP_ID] == map_id:
                return True
    return False


def mash(log: Log, g: np.random.Generator, steps: int, weights=(0.17, 0.17, 0.17, 0.17, 0.2, 0.1, 0.02)) -> None:
    for _ in range(steps):
        log.act(int(g.choice(7, p=weights)))


def battle_until_over(log: Log, g: np.random.Generator, cap: int) -> int:
    a = log.pair.a.pyboy
    for used in range(cap):
        if a.memory[BATTLE] == 0 and used > 12:
            return used
        log.act(int(g.choice(7, p=(0.1, 0.03, 0.03, 0.04, 0.65, 0.13, 0.02))))
    return cap


def scenario_wild(log, g, n):
    a = log.pair.a.pyboy
    log.fly_ok = fly(log, CITIES["viridian"])
    done = 0
    for i in range(n):
        wait_overworld(log)
        log.pair.write(OPPONENT, WILD[i % len(WILD)])
        log.pair.write(ENEMY_LEVEL, int(g.integers(2, 8)) if i % 7 else 40)
        for _ in range(12):
            log.act(int(g.choice([B, LEFT, RIGHT])))
            if a.memory[BATTLE]:
                done += 1
                break
        battle_until_over(log, g, 500)
    log.started = done


def scenario_trainer(log, g, n):
    a = log.pair.a.pyboy
    log.fly_ok = fly(log, CITIES["viridian"])
    done = 0
    for i in range(n):
        wait_overworld(log)
        cls, number = TRAINERS[i % len(TRAINERS)]
        log.pair.write(OPPONENT, cls)
        log.pair.write(TRAINER_NO, number)
        log.pair.write(ENEMY_LEVEL, cls)
        for _ in range(12):
            log.act(int(g.choice([B, LEFT, RIGHT])))
            if a.memory[BATTLE] == 2:
                done += 1
                break
        battle_until_over(log, g, 900)
    log.started = done


def scenario_menus(log, g, n):
    """Open the Start menu and enter each item: Pokedex, Pokemon, Item, trainer card, Save, Option, Exit."""
    for i in range(n):
        log.act(START)
        for _ in range(i % 7):
            log.act(DOWN)
        log.act(A)
        for _ in range(int(g.integers(4, 14))):
            log.act(int(g.choice([UP, DOWN, LEFT, RIGHT, A, B])))
        for _ in range(4):
            log.act(B)
        mash(log, g, 8)


def scenario_text(log, g, n):
    """Walk around Oak's lab pressing A at people, books and the Pokeball table."""
    mash(log, g, n, weights=(0.15, 0.15, 0.15, 0.15, 0.3, 0.09, 0.01))


def scenario_cities(log, g, n):
    log.flights = {}
    for name, destination in CITIES.items():
        log.flights[name] = fly(log, destination)
        mash(log, g, n)


def scenario_heal(log, g, n):
    """Hurt the party, walk to the Viridian Center nurse and heal; the outcome must match in both modes."""
    a, b = log.pair.emus
    log.heals = []
    for attempt in range(n):
        fly(log, CITIES["viridian"])
        for _ in range(20):
            log.act(UP)
            if a.memory[MAP_ID] == 0x29:
                break
        hp_max = (a.memory[PARTY_MAXHP0] << 8) | a.memory[PARTY_MAXHP0 + 1]
        log.pair.write(PARTY_HP0, 0)
        log.pair.write(PARTY_HP0 + 1, 1)
        walk = 0
        while a.memory[0xD361] > 3 and walk < 24:  # up to the row in front of the counter
            log.act(UP)
            walk += 1
        for _ in range(40):
            log.act(A)
        for _ in range(25):
            log.act(B)
        log.heals.append({"attempt": attempt, "map": a.memory[MAP_ID], "yx": (a.memory[0xD361], a.memory[0xD362]), "hp_max": hp_max,
                          "hp_acc": (a.memory[PARTY_HP0] << 8) | a.memory[PARTY_HP0 + 1], "hp_fast": (b.memory[PARTY_HP0] << 8) | b.memory[PARTY_HP0 + 1]})
        for _ in range(20):
            log.act(DOWN)
            if a.memory[MAP_ID] == 1:
                break


def scenario_dungeons(log, g, n):
    log.entered = {}
    for name, map_id, warp in DUNGEONS:
        log.entered[name] = enter_dungeon(log, map_id, warp)
        mash(log, g, n)


SCENARIOS = {"wild": (scenario_wild, 60), "trainer": (scenario_trainer, 24), "menus": (scenario_menus, 120), "text": (scenario_text, 6000),
             "cities": (scenario_cities, 600), "heal": (scenario_heal, 6), "dungeons": (scenario_dungeons, 800)}


def cmd_scenarios(args) -> None:
    rom = Path(args.rom)
    out = open(args.out, "a")
    shots = Path(args.screens) if args.screens else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    frames = Frames(rom) if args.frames else None
    for name in args.names:
        function, default = SCENARIOS[name]
        n = int(default * args.scale)
        for resync in ((True, False) if args.both else (args.resync,)):
            torch.manual_seed(args.seed)
            pair = Pair(rom, START_STATE, fast=not args.control)
            log = Log(pair, resync=resync, frames=frames if resync else None, screenshots=shots if resync else None, name=name)
            started = time.time()
            function(log, np.random.default_rng(args.seed), n)
            record = log.summary()
            record["seconds"] = round(time.time() - started, 1)
            record["control"] = args.control
            for extra in ("heals", "started", "flights", "entered", "fly_ok"):
                if hasattr(log, extra):
                    record[extra] = getattr(log, extra)
            out.write(as_line(record) + "\n")
            out.flush()
            print(name, "resync" if resync else "open", "steps", record["steps"], "s", record["seconds"], "non_rng", record["non_rng_divergence_steps"], "first", {k: v.get("step") for k, v in record["first"].items()}, flush=True)
            pair.close()
    if frames:
        frames.close()


def cmd_policy(args) -> None:
    """A trained policy samples from accurate observations; the same actions drive fast mode."""
    rom = Path(args.rom)
    payload = CheckpointStore(Path(args.checkpoints)).load_latest()
    model = ActorCritic()
    model.load_state_dict(payload["model"])
    model.eval()
    torch.set_num_threads(1)
    out = open(args.out, "a")
    frames = Frames(rom) if args.frames else None
    for seed in range(args.seed, args.seed + args.trajectories):
        generator = torch.Generator().manual_seed(seed)
        pair = Pair(rom, START_STATE, fast=not args.control)
        log = Log(pair, resync=args.resync, frames=frames, screenshots=None, model=model, name=f"policy_seed{seed}")
        started = time.time()
        for _ in range(args.steps):
            with torch.inference_mode():
                logits, _ = model({k: torch.from_numpy(v[None]) for k, v in pair.obs_a.items()})
            log.act(int(torch.multinomial(logits.softmax(1)[0], 1, generator=generator)))
        record = log.summary()
        record.update(seconds=round(time.time() - started, 1), env_steps=payload["env_steps"], control=args.control)
        out.write(as_line(record) + "\n")
        out.flush()
        print(record["name"], "steps", record["steps"], "s", record["seconds"], "maps", record["maps_visited"], "first", {k: v.get("step") for k, v in record["first"].items()}, "tiles", record["tiles"], "events", record["events"], flush=True)
        pair.close()
    if frames:
        frames.close()


def cmd_interchange(args) -> None:
    """Are save states interchangeable? Saved in one mode, loaded in the other, then run next to a same-mode continuation."""
    rom = Path(args.rom)
    out = open(args.out, "a")
    for source_fast, fly_to, label in ((False, None, "lab"), (True, None, "lab"), (False, "viridian", "viridian"), (True, "viridian", "viridian")):
        g = np.random.default_rng(args.seed)
        source = open_emulator(rom, fast_mode=source_fast)
        same = open_emulator(rom, fast_mode=source_fast)
        other = open_emulator(rom, fast_mode=not source_fast)
        load(source, START_STATE.read_bytes())
        if fly_to:
            source.memory[FLY_MAP] = CITIES[fly_to]
            source.memory[FLAGS_D732] = source.memory[FLAGS_D732] | 0x08
        for _ in range(args.warmup):
            action = BUTTONS[int(g.integers(0, 7))]
            source.button_press(action)
            source.tick(8, False, False)
            source.button_release(action)
            source.tick(16, True, False)
        state = save(source)
        load(same, state)
        load(other, state)
        first_same = first_other = None
        pixel_steps = 0
        for s in range(args.steps):
            action = BUTTONS[int(g.integers(0, 7))]
            for emu in (source, same, other):
                emu.button_press(action)
                emu.tick(8, False, False)
                emu.button_release(action)
                emu.tick(16, True, False)
            memory = lambda e: mem(e, WRAM) + mem(e, HRAM)  # noqa: E731
            if first_same is None and memory(source) != memory(same):
                first_same = s
            if first_other is None and memory(source) != memory(other):
                first_other = s
            pixel_steps += bool((source.screen.ndarray != other.screen.ndarray).any())
        record = {"saved_in": "fast" if source_fast else "accurate", "where": label, "state_bytes": len(state), "steps": args.steps,
                  "first_divergence_same_mode": first_same, "first_divergence_other_mode": first_other, "screen_diff_steps_other_mode": pixel_steps,
                  "resaved_bytes_equal": save(other) == save(source), "final_place": [place(source), place(other)]}
        out.write(as_line(record) + "\n")
        out.flush()
        print(record, flush=True)
        for emu in (source, same, other):
            emu.stop(save=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scenarios")
    s.add_argument("--names", nargs="+", default=list(SCENARIOS))
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--resync", action="store_true")
    s.add_argument("--both", action="store_true")
    s.add_argument("--frames", action="store_true")
    s.add_argument("--control", action="store_true")
    s.add_argument("--screens", default=None)
    p = sub.add_parser("policy")
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--trajectories", type=int, default=1)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--resync", action="store_true")
    p.add_argument("--frames", action="store_true")
    p.add_argument("--control", action="store_true")
    i = sub.add_parser("interchange")
    i.add_argument("--steps", type=int, default=3000)
    i.add_argument("--warmup", type=int, default=200)
    for sp in (s, p, i):
        sp.add_argument("--rom", required=True)
        sp.add_argument("--out", required=True)
        sp.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"scenarios": cmd_scenarios, "policy": cmd_policy, "interchange": cmd_interchange}[args.command](args)


if __name__ == "__main__":
    main()
