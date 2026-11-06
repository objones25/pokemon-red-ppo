"""Gymnasium environment: one Pokemon Red game from the start state."""

from __future__ import annotations

import io
import math
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
from pokered.rewards import BadgeGain, EpisodeReward, EventFlagGain, LevelGain, NewSpecies, NewTile
from pokered.telemetry import EpisodeDetail, EpisodeReport, EpisodeTelemetry, Snapshot


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
        self._reward = EpisodeReward(
            [self._tiles, self._events, BadgeGain(config.badge_weight), LevelGain(config.level_weight), NewSpecies(config.species_weight)]
        )
        self._milestones = MilestoneTracker()
        self._telemetry = EpisodeTelemetry(config.stall_steps)
        self._last_stall: Snapshot | None = None
        self._stack = FrameStack()
        self._visited = VisitedMap()
        self._budget_fraction = 1.0
        self._first_episode = True
        self._steps = 0

    def budget_for(self, events_gained: int) -> int:
        require(events_gained >= 0, f"events gained cannot be negative, got {events_gained}")
        full = self._config.base_budget + self._config.budget_per_event * events_gained
        budget = max(1, math.ceil(self._budget_fraction * full))
        require(budget <= full, f"budget {budget} exceeds the full budget {full}")
        return budget

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        # The first episode keeps a random fraction of its whole budget, event extensions included, so first episodes
        # end anywhere in a full one and envs never reach the same part of the game together, even after a resume or
        # --init-from. Shortening only the base budget spread them over 10,240 of a trained agent's ~80,000 steps.
        stagger = self._first_episode and self._config.stagger_first_episode
        self._budget_fraction = 1.0 - float(self.np_random.uniform()) if stagger else 1.0
        require(0.0 < self._budget_fraction <= 1.0, f"budget fraction {self._budget_fraction} is outside (0, 1]")
        self._first_episode = False
        restore(self.pyboy, self._state, battle_animations=self._config.battle_animations)
        frame = read_frame(self.pyboy.memory)
        self._reward.reset(frame)
        self._milestones.reset(frame)
        self._telemetry.reset(frame)
        self._last_stall = None
        self._visited.reset()
        self._stack.reset(self._screen())
        self._steps = 0
        budget = self.budget_for(0)
        require(budget >= 1, f"an episode needs at least one step, got a budget of {budget}")
        return self._observation(frame), {"budget": budget, "map_id": frame.map_id, "in_battle": frame.in_battle}

    def step(self, action: int):
        require(self.action_space.contains(int(action)), f"action {action} is outside the action space")
        press(self.pyboy, int(action))
        frame = read_frame(self.pyboy.memory)
        terms = self._reward.step(frame)
        self._stack.push(self._screen())
        self._steps += 1
        self._milestones.update(frame, step=self._steps)
        if self._telemetry.update(frame, action=int(action), terms=terms):
            self._last_stall = self._snapshot("stall", frame)
        truncated = self._steps >= self.budget_for(self._events.gained)
        info: dict[str, Any] = {"reward_terms": terms, "map_id": frame.map_id, "in_battle": frame.in_battle}
        if truncated:
            info["episode"] = self._report(frame)
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

    def _snapshot(self, kind: str, frame: GameFrame) -> Snapshot:  # nsp: ignore NSP001 Snapshot only holds what these readers checked
        state = io.BytesIO()
        self.pyboy.save_state(state)
        return Snapshot(
            kind=kind,
            step=self._steps,
            dry_steps=self._telemetry.dry_steps,
            map_id=frame.map_id,
            x=frame.x,
            y=frame.y,
            in_battle=frame.in_battle,
            party_levels=frame.party_levels,
            screen=np.array(self.pyboy.screen.ndarray[:, :, 0], np.uint8),
            state=state.getvalue(),
        )

    def _report(self, frame: GameFrame) -> EpisodeReport:
        telemetry = self._telemetry
        require(telemetry.steps == self._steps, f"telemetry saw {telemetry.steps} of {self._steps} steps")
        summary: dict[str, float] = {
            "tiles": self._tiles.gained,
            "events": self._events.gained,
            "badges": frame.badges.bit_count(),
            "max_level": max(frame.party_levels, default=0),
            "party_size": len(frame.party_levels),
            "fights": len(telemetry.fights),
            "blackouts": telemetry.blackouts,
            "dry_steps": telemetry.dry_steps,
            **self._milestones.summary(),
        }
        detail = EpisodeDetail(
            length=self._steps,
            flag_steps=dict(telemetry.flag_steps),
            map_steps=dict(telemetry.map_steps),
            fights=tuple(telemetry.fights),
            phase_rewards=dict(telemetry.phase_rewards),
            action_counts=tuple(telemetry.action_counts),
            last_stall=self._last_stall,
            end=self._snapshot("end", frame),
        )
        return EpisodeReport(summary=summary, detail=detail)
