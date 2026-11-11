"""Opening an emulator and restoring the start state."""

from __future__ import annotations

import numpy as np
import pytest
from support import REPO, emulator, requires_rom

from pokered import ram_map
from pokered.buttons import BUTTONS, press
from pokered.emulator import CutWatch, HiddenEventWatch, restore
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


TREE_STATE = REPO / "tests" / "states" / "vermilion_gym_tree.state"  # run6 save: (15, 17) in Vermilion, facing the tree


def give_cut(memory) -> None:
    """Test setup: the first party Pokemon learns Cut in its last move slot. The field menu lists Cut for any
    Pokemon that knows it; only teaching it from HM01 checks the species."""
    first = ram_map.PARTY_MONS
    memory[first + ram_map.OFF_MOVES + 3] = ram_map.CUT
    memory[first + ram_map.OFF_PP + 3] = 30


def use_cut_from_the_field_menu(pyboy) -> None:
    """START, POKEMON, the first Pokemon, its first field move (Cut), then clear the text. The start menu remembers
    its cursor, so steer by the cursor byte rather than a fixed count."""

    def to(item: int, button: str) -> None:
        for _ in range(8):
            if pyboy.memory[ram_map.MENU_CURSOR] == item:
                return
            press(pyboy, BUTTONS.index(button))

    press(pyboy, BUTTONS.index("start"))
    to(1, "down")
    _press_all(pyboy, ("a",))
    to(0, "up")
    _press_all(pyboy, ("a",))
    to(0, "up")
    _press_all(pyboy, ("a",) * 6)


@pytest.mark.slow
def test_the_cut_watch_records_the_vermilion_gym_tree():
    with emulator() as pyboy:
        watch = CutWatch(pyboy)
        restore(pyboy, TREE_STATE.read_bytes(), battle_animations=False)
        give_cut(pyboy.memory)
        use_cut_from_the_field_menu(pyboy)
        assert watch.drain() == ((5, 15, 17, 0, ram_map.CUT_TREE_TILE),)


@pytest.mark.slow
def test_the_cut_watch_records_nothing_with_nothing_to_cut():
    with emulator() as pyboy:
        watch = CutWatch(pyboy)
        restore(pyboy, TREE_STATE.read_bytes(), battle_animations=False)
        give_cut(pyboy.memory)
        _press_all(pyboy, ("up",))  # one step away from the tree, facing open ground
        use_cut_from_the_field_menu(pyboy)
        assert watch.drain() == ()


@pytest.mark.slow
def test_cutting_the_tree_opens_the_path_to_the_gym():
    with emulator() as pyboy:
        restore(pyboy, TREE_STATE.read_bytes(), battle_animations=False)
        give_cut(pyboy.memory)
        use_cut_from_the_field_menu(pyboy)
        _press_all(pyboy, ("down", "down"))
        assert (pyboy.memory[ram_map.X_POS], pyboy.memory[ram_map.Y_POS]) == (15, 19)
