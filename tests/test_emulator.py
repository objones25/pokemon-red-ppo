"""Opening an emulator and restoring the start state."""

from __future__ import annotations

import numpy as np
import pytest
from support import emulator, requires_rom

from pokered import ram_map
from pokered.emulator import restore
from pokered.start_state import START_STATE

pytestmark = requires_rom


def _state() -> bytes:
    return START_STATE.read_bytes()


def test_restore_turns_battle_animations_off_when_requested():
    with emulator() as pyboy:
        restore(pyboy, _state(), battle_animations=False)
        assert pyboy.memory[ram_map.OPTIONS] & ram_map.OPTION_NO_BATTLE_ANIMATION != 0


def test_restore_keeps_battle_animations_on_when_requested():
    with emulator() as pyboy:
        restore(pyboy, _state(), battle_animations=True)
        assert pyboy.memory[ram_map.OPTIONS] & ram_map.OPTION_NO_BATTLE_ANIMATION == 0


def test_restore_keeps_fast_text_setting():
    with emulator() as pyboy:
        restore(pyboy, _state(), battle_animations=False)
        assert pyboy.memory[ram_map.OPTIONS] & 0x7F == 0x01


def test_restore_leaves_a_valid_screen_for_the_first_observation():
    with emulator() as pyboy:
        restore(pyboy, _state(), battle_animations=False)
        assert len(np.unique(pyboy.screen.ndarray[:, :, 0])) == 4


@pytest.mark.parametrize("repeat", [1, 2])
def test_restore_keeps_player_position(repeat):
    with emulator() as pyboy:
        for _ in range(repeat):
            restore(pyboy, _state(), battle_animations=False)
        assert (pyboy.memory[ram_map.MAP_ID], pyboy.memory[ram_map.X_POS], pyboy.memory[ram_map.Y_POS]) == (40, 5, 3)
