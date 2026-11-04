"""One action is one button press inside a 24-frame decision."""

from __future__ import annotations

from support import emulator, requires_rom

from pokered import ram_map
from pokered.buttons import BUTTONS, press
from pokered.start_state import START_STATE


def test_buttons_are_the_seven_game_inputs_without_select():
    assert sorted(BUTTONS) == ["a", "b", "down", "left", "right", "start", "up"]


@requires_rom
def test_press_down_moves_player_one_tile_south():
    with emulator() as pyboy:
        with START_STATE.open("rb") as handle:
            pyboy.load_state(handle)
        before = pyboy.memory[ram_map.Y_POS]
        press(pyboy, BUTTONS.index("down"))
        assert pyboy.memory[ram_map.Y_POS] == before + 1
