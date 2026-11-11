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


class HiddenEventWatch:
    """Records each hidden event (a PC, sign or poster with no sprite) the game matches when the player presses A.

    The hidden-event RAM is shared with other variables (pret ram/wram.asm, ram/hram.asm), so polling it can't tell a
    match from leftover bytes. A hook on the match itself can. The hook survives ``load_state``.
    """

    def __init__(self, pyboy: PyBoy) -> None:
        self._pyboy = pyboy
        self._found: list[tuple[int, int]] = []
        pyboy.hook_register(ram_map.HIDDEN_EVENT_MATCH_BANK, ram_map.HIDDEN_EVENT_MATCH_ADDRESS, self._record, None)

    def _record(self, _context: object) -> None:  # nsp: ignore NSP001 GameFrame validates each (map, index)
        memory = self._pyboy.memory
        self._found.append((memory[ram_map.MAP_ID], memory[ram_map.HIDDEN_EVENT_INDEX]))

    def drain(self) -> tuple[tuple[int, int], ...]:
        """The matches since the last drain, oldest first. The A button is held 8 frames, so one press can match more
        than once; the reward pays each event once anyway."""
        found = tuple(self._found)
        self._found.clear()
        require(not self._found, "draining must empty the watch")
        return found


class CutWatch:
    """Records each accepted Cut as (map, x, y, facing, tile): where the player stood and what was cut.

    Cutting sets no flag, and the tree grows back when the map reloads, so only a hook on the cut itself sees it.
    The hook survives ``load_state``.
    """

    def __init__(self, pyboy: PyBoy) -> None:
        self._pyboy = pyboy
        self._cuts: list[tuple[int, int, int, int, int]] = []
        pyboy.hook_register(ram_map.CUT_HOOK_BANK, ram_map.CUT_HOOK_ADDRESS, self._record, None)

    def _record(self, _context: object) -> None:  # nsp: ignore NSP001 GameFrame validates each cut
        memory = self._pyboy.memory
        self._cuts.append(
            (
                memory[ram_map.MAP_ID],
                memory[ram_map.X_POS],
                memory[ram_map.Y_POS],
                memory[ram_map.FACING],
                memory[ram_map.CUT_TILE],
            )
        )

    def drain(self) -> tuple[tuple[int, int, int, int, int], ...]:
        """The cuts since the last drain, oldest first."""
        cuts = tuple(self._cuts)
        self._cuts.clear()
        require(not self._cuts, "draining must empty the watch")
        return cuts


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
