"""The only module that imports PyBoy: opening an emulator and restoring a saved game."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Final

from pyboy import PyBoy

from pokered import ram_map
from pokered.negative_space import require

# After load_state, frame 1 shows the saved image and frame 2 is blank; frame 3 is the first real one.
SETTLE_FRAMES: Final[int] = 3


def open_emulator(rom: Path, *, fast_mode: bool = False) -> PyBoy:
    """A headless emulator at full speed. Call ``pyboy.stop(save=False)`` so no .ram file lands next to the ROM."""
    if not rom.is_file():
        raise FileNotFoundError(f"ROM not found at {rom}; it isn't distributed, so place your own copy there")
    options = {"fast_mode": True} if fast_mode else {"sound_emulated": False}
    pyboy = PyBoy(str(rom), window="null", log_level="ERROR", **options)
    pyboy.set_emulation_speed(0)
    return pyboy


def restore(pyboy: PyBoy, state: bytes, *, battle_animations: bool) -> None:
    """Load a current-format save, apply the battle-animation option, and settle the screen."""
    require(state, "start state must not be empty")
    pyboy.load_state(io.BytesIO(state))
    options = pyboy.memory[ram_map.OPTIONS]
    if battle_animations:
        pyboy.memory[ram_map.OPTIONS] = options & ~ram_map.OPTION_NO_BATTLE_ANIMATION
    else:
        pyboy.memory[ram_map.OPTIONS] = options | ram_map.OPTION_NO_BATTLE_ANIMATION
    pyboy.tick(SETTLE_FRAMES, True, False)
    animations_off = bool(pyboy.memory[ram_map.OPTIONS] & ram_map.OPTION_NO_BATTLE_ANIMATION)
    require(animations_off != battle_animations, "the battle-animation option did not stick")
