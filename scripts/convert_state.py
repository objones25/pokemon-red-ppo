"""Re-save a PyBoy state in the installed PyBoy's format, without advancing the game.

Old-format saves omit fields that newer PyBoy versions restore, so a replay differs between a fresh and a reused
emulator (UPSTREAM_FINDINGS.md D7). Usage: convert_state.py ROM SOURCE DESTINATION
"""

from __future__ import annotations

import sys
from pathlib import Path

from pyboy import PyBoy


def main(rom: str, source: str, destination: str) -> None:
    pyboy = PyBoy(rom, window="null", log_level="ERROR", sound_emulated=False)
    try:
        with Path(source).open("rb") as handle:
            pyboy.load_state(handle)
        with Path(destination).open("wb") as handle:
            pyboy.save_state(handle)
    finally:
        pyboy.stop(save=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
