"""Gymnasium environment: one Pokemon Red game from the start state."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from pokered.buttons import BUTTONS, press
from pokered.config import EnvConfig
from pokered.emulator import open_emulator, restore
from pokered.frame import GameFrame, read_frame
from pokered.milestones import MilestoneTracker
from pokered.negative_space import require
from pokered.observations import OBSERVATION_SHAPES, FrameStack, VisitedMap, downscale_screen, party_and_badges
from pokered.rewards import BadgeGain, EpisodeReward, EventFlagGain, LevelGain, NewTile


class PokemonRedEnv(gym.Env):
    """Time is the only way an episode ends, so it always ends as truncated: the value target must bootstrap."""

    def __init__(self, config: EnvConfig) -> None:
        super().__init__()
        self._config = config
        self.observation_space = gym.spaces.Dict(
            {key: gym.spaces.Box(0, 255, shape, np.uint8) for key, shape in OBSERVATION_SHAPES.items()}
        )
        self.action_space = gym.spaces.Discrete(len(BUTTONS))
        self.pyboy = open_emulator(config.rom, fast_mode=config.fast_mode)
        self._state = config.state.read_bytes()
        self._tiles = NewTile(config.tile_weight)
        self._events = EventFlagGain(config.event_weight)
        self._reward = EpisodeReward([self._tiles, self._events, BadgeGain(config.badge_weight), LevelGain(config.level_weight)])
        self._milestones = MilestoneTracker()
        self._stack = FrameStack()
        self._visited = VisitedMap()
        self._base_budget = config.base_budget
        self._first_episode = True
        self._steps = 0
        self._totals: dict[str, float] = {}

    def budget_for(self, events_gained: int) -> int:
        require(events_gained >= 0, f"events gained cannot be negative, got {events_gained}")
        return self._base_budget + self._config.budget_per_event * events_gained

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        # A shortened first episode keeps envs from all resetting together, including after a resume.
        stagger = self._first_episode and self._config.stagger_first_episode
        self._base_budget = int(self.np_random.integers(1, self._config.base_budget + 1)) if stagger else self._config.base_budget
        self._first_episode = False
        restore(self.pyboy, self._state, battle_animations=self._config.battle_animations)
        frame = read_frame(self.pyboy.memory)
        self._reward.reset(frame)
        self._milestones.reset(frame)
        self._visited.reset()
        self._stack.reset(self._screen())
        self._steps = 0
        self._totals = {}
        budget = self.budget_for(0)
        require(budget >= 1, f"an episode needs at least one step, got a budget of {budget}")
        return self._observation(frame), {"budget": budget}

    def step(self, action: int):
        require(self.action_space.contains(int(action)), f"action {action} is outside the action space")
        press(self.pyboy, int(action))
        frame = read_frame(self.pyboy.memory)
        terms = self._reward.step(frame)
        self._stack.push(self._screen())
        self._steps += 1
        self._milestones.update(frame, step=self._steps)
        for name, value in terms.items():
            self._totals[name] = self._totals.get(name, 0.0) + value
        truncated = self._steps >= self.budget_for(self._events.gained)
        info: dict[str, Any] = {"reward_terms": terms}
        if truncated:
            info["episode"] = self._summary(frame)
        require(self._steps <= self.budget_for(self._events.gained), "stepped past the budget without a reset")
        return self._observation(frame), sum(terms.values()), False, truncated, info

    def close(self) -> None:
        self.pyboy.stop(save=False)

    def _screen(self) -> np.ndarray:
        return downscale_screen(self.pyboy.screen.ndarray[:, :, 0])

    def _observation(self, frame: GameFrame) -> dict[str, np.ndarray]:  # nsp: ignore NSP001 every piece checks its own shape
        if not frame.in_battle:
            self._visited.visit(frame.map_id, frame.x, frame.y)
        return {
            "screens": self._stack.array.copy(),
            "visited": self._visited.crop(frame.map_id, frame.x, frame.y),
            "party": party_and_badges(frame),
            "events": np.frombuffer(frame.event_flags, np.uint8).copy(),
        }

    def _summary(self, frame: GameFrame) -> dict[str, float]:
        return {
            **self._totals,
            "length": self._steps,
            "tiles": self._tiles.gained,
            "events": self._events.gained,
            "badges": frame.badges.bit_count(),
            "level_sum": sum(frame.party_levels),
            **self._milestones.summary(),
        }
