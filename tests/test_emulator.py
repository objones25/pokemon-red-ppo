"""Opening an emulator and restoring the start state."""

from __future__ import annotations

import numpy as np
import pytest
from support import emulator, requires_rom

from pokered import ram_map
from pokered.buttons import BUTTONS, press
from pokered.emulator import HiddenEventWatch, restore
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


OAKS_LAB_EMAIL = (40, 3)  # pret data/events/hidden_events.asm: the fourth hidden event in Oak's Lab, at (1, 1)
WALK_TO_EMAIL = ("left",) * 4 + ("up",) * 3  # from (5, 3) to (1, 2), facing up


def _press_all(pyboy, buttons):
    for button in buttons:
        press(pyboy, BUTTONS.index(button))


def test_the_watch_records_the_hidden_event_the_player_reads():
    with emulator() as pyboy:
        watch = HiddenEventWatch(pyboy)
        restore(pyboy, _state(), battle_animations=False)
        _press_all(pyboy, (*WALK_TO_EMAIL, "a"))
        assert watch.drain() == (OAKS_LAB_EMAIL,)


def test_the_watch_records_nothing_without_a_hidden_event():
    with emulator() as pyboy:
        watch = HiddenEventWatch(pyboy)
        restore(pyboy, _state(), battle_animations=False)
        _press_all(pyboy, ("left", "a"))
        assert watch.drain() == ()


def test_draining_the_watch_empties_it():
    with emulator() as pyboy:
        watch = HiddenEventWatch(pyboy)
        restore(pyboy, _state(), battle_animations=False)
        _press_all(pyboy, (*WALK_TO_EMAIL, "a"))
        watch.drain()
        assert watch.drain() == ()
