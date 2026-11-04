"""The one environment used by BOTH loops. Imports neither torch nor stable_baselines3.

Step: press button (8-frame hold in 24 frames, render last only), integer 2x2 block-mean greyscale to 72x80,
3-frame stack, read RAM via pokered.frame.read_frame (320 event-flag bytes etc.), reward via pokered.rewards.
Auto-truncates at MAX_STEPS. Timers split each step into emulator / observation / reward (seconds, cumulative).
"""
from __future__ import annotations

import io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from pyboy import PyBoy  # noqa: E402

import ram_map  # noqa: E402
from pokered.buttons import BUTTONS, press  # noqa: E402
from pokered.frame import read_frame  # noqa: E402
from pokered.rewards import BadgeGain, EpisodeReward, EventFlagGain, LevelGain, NewTile  # noqa: E402

ROM = os.path.join(HERE, "game", "red.gb")
STATE = os.path.join(HERE, "game", "start.state")
MAX_STEPS = 10240
N_ACTIONS = len(BUTTONS)
SCREEN_SHAPE = (3, 72, 80)
VISIT_SHAPE = (1, 48, 48)
STATE_DIM = 2 * ram_map.MAX_PARTY + 8 * ram_map.EVENT_FLAGS_BYTES


class CoreEnv:
    def __init__(self, fast_mode: bool = False, obs_mode: str = "image") -> None:
        assert obs_mode in ("image", "dict")
        self.obs_mode = obs_mode
        kw = {"fast_mode": True} if fast_mode else {"sound_emulated": False}
        self.pyboy = PyBoy(ROM, window="null", log_level="ERROR", **kw)
        self.pyboy.set_emulation_speed(0)
        with open(STATE, "rb") as f:
            self._state = f.read()
        self._reward = EpisodeReward(
            [NewTile(0.005), EventFlagGain(2.0), BadgeGain(5.0), LevelGain(0.5)]
        )
        self._stack = np.zeros(SCREEN_SHAPE, np.uint8)
        self._visited: dict[int, np.ndarray] = {}
        self._t = 0
        self.t_emu = 0.0
        self.t_obs = 0.0
        self.t_rew = 0.0
        self.n_steps = 0
        self.n_resets = 0

    def _frame(self) -> np.ndarray:
        a = self.pyboy.screen.ndarray[:, :, 0].astype(np.uint16)
        return ((a[::2, ::2] + a[1::2, ::2] + a[::2, 1::2] + a[1::2, 1::2]) >> 2).astype(np.uint8)

    def _visit_crop(self, f) -> np.ndarray:
        m = self._visited.get(f.map_id)
        if m is None:
            m = self._visited[f.map_id] = np.zeros((256 + 48, 256 + 48), np.uint8)
        y, x = f.y + 24, f.x + 24
        m[y, x] = 255
        return m[y - 24 : y + 24, x - 24 : x + 24].reshape(VISIT_SHAPE).copy()

    def _state_vec(self, f) -> np.ndarray:
        v = np.zeros(STATE_DIM, np.float32)
        mem = self.pyboy.memory
        for i in range(len(f.party_levels)):
            base = ram_map.PARTY_MONS + ram_map.MON_SIZE * i
            hp = (mem[base + ram_map.OFF_HP] << 8) | mem[base + ram_map.OFF_HP + 1]
            mx = (mem[base + ram_map.OFF_MAX_HP] << 8) | mem[base + ram_map.OFF_MAX_HP + 1]
            v[2 * i] = hp / max(mx, 1)
            v[2 * i + 1] = f.party_levels[i] / 100.0
        v[2 * ram_map.MAX_PARTY :] = np.unpackbits(np.frombuffer(f.event_flags, np.uint8))
        return v

    def _obs(self, f, first: bool):
        s = self._frame()
        if first:
            self._stack[:] = s
        else:
            self._stack[:2] = self._stack[1:]
            self._stack[2] = s
        if self.obs_mode == "image":
            return self._stack.copy()
        return {
            "screen": self._stack.copy(),
            "visited": self._visit_crop(f),
            "state": self._state_vec(f),
        }

    def reset(self):
        self.pyboy.load_state(io.BytesIO(self._state))
        self.pyboy.tick(1, True, False)
        f = read_frame(self.pyboy.memory)
        self._reward.reset(f)
        self._visited = {}
        self._t = 0
        self.n_resets += 1
        return self._obs(f, True)

    def step(self, action: int):
        t0 = time.perf_counter()
        press(self.pyboy, int(action))
        t1 = time.perf_counter()
        f = read_frame(self.pyboy.memory)
        terms = self._reward.step(f)
        r = terms["explore"] + terms["event"] + terms["badge"] + terms["level"]
        t2 = time.perf_counter()
        obs = self._obs(f, False)
        t3 = time.perf_counter()
        self.t_emu += t1 - t0
        self.t_rew += t2 - t1
        self.t_obs += t3 - t2
        self.n_steps += 1
        self._t += 1
        return obs, float(r), self._t >= MAX_STEPS

    def timers(self) -> tuple[float, float, float, int, int]:
        return (self.t_emu, self.t_obs, self.t_rew, self.n_steps, self.n_resets)

    def close(self) -> None:
        self.pyboy.stop(save=False)
