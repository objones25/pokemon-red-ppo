"""The action space: one button held for 8 frames inside a 24-frame decision."""

from __future__ import annotations

from typing import Final, Protocol

from pokered.negative_space import require

BUTTONS: Final[tuple[str, ...]] = ("down", "left", "right", "up", "a", "b", "start")
FRAMES_PER_ACTION: Final[int] = 24
HOLD_FRAMES: Final[int] = 8


class Emulator(Protocol):
    def button_press(self, input: str) -> None: ...
    def button_release(self, input: str) -> None: ...
    def tick(self, count: int = 1, render: bool = True, sound: bool = True) -> bool: ...


def press(pyboy: Emulator, action: int) -> None:
    """Run one decision. Only the final frame is rendered, which is all an observation reads."""
    require(0 <= action < len(BUTTONS), f"action {action} is not a button")
    button = BUTTONS[action]
    pyboy.button_press(button)
    pyboy.tick(HOLD_FRAMES, False, False)
    pyboy.button_release(button)
    pyboy.tick(FRAMES_PER_ACTION - HOLD_FRAMES - 1, False, False)
    pyboy.tick(1, True, False)
