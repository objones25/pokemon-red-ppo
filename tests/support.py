"""Shared paths and emulator helpers for tests that need the user's ROM."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from pyboy import PyBoy

from pokered.emulator import open_emulator
from pokered.telemetry import EpisodeDetail, EpisodeReport, Snapshot

REPO = Path(__file__).resolve().parents[1]
ROM = REPO / "Pokemon Red.gb"

requires_rom = pytest.mark.skipif(not ROM.exists(), reason="the ROM is not distributed; place it at the repo root")


@contextmanager
def emulator() -> Iterator[PyBoy]:
    """A headless emulator that never writes a .ram file next to the ROM."""
    pyboy = open_emulator(ROM)
    try:
        yield pyboy
    finally:
        pyboy.stop(save=False)


def fake_episode(**summary: float) -> EpisodeReport:
    """One finished episode that spent its only step on map 3."""
    end = Snapshot("end", 1, 1, 3, 0, 0, False, (6,), np.zeros((144, 160), np.uint8), b"state")
    detail = EpisodeDetail(1, {}, {3: 1}, (), {}, (1, 0, 0, 0, 0, 0, 0), None, end)
    return EpisodeReport(summary=summary, detail=detail)


class CountingEnv:
    """A trivial env for vec tests: its observation counts steps, its reward echoes the action."""

    SHAPES = {"x": (2,)}  # noqa: RUF012

    def __init__(self, budget: int = 3, crash_at: int | None = None) -> None:
        self._budget = budget
        self._crash_at = crash_at
        self._steps = 0
        self._seed = 0

    def _obs(self):
        return {"x": np.array([self._seed, self._steps], np.uint8)}

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._seed = seed
        self._steps = 0
        return self._obs(), {"map_id": 7, "in_battle": False}

    def step(self, action):
        if self._crash_at is not None and self._steps == self._crash_at:
            raise ValueError("planted crash")
        self._steps += 1
        truncated = self._steps >= self._budget
        info: dict = {"map_id": 7, "in_battle": False}
        if truncated:
            info["episode"] = fake_episode(length=float(self._steps))
        return self._obs(), float(action), False, truncated, info

    def close(self) -> None:
        pass


class AffinityEnv(CountingEnv):
    """Reports the CPU its process is pinned to as the reward."""

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        if sys.platform != "linux":
            raise RuntimeError("AffinityEnv needs Linux")
        return obs, float(next(iter(os.sched_getaffinity(0)))), terminated, truncated, info
