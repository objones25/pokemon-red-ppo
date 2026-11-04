"""Location of the training start state: Squirtle chosen, Pokedex received, in Oak's lab."""

from __future__ import annotations

from pathlib import Path
from typing import Final

START_STATE: Final[Path] = Path(__file__).resolve().parents[2] / "states" / "has_pokedex_squirtle.state"
