"""Story milestones, from pret/pokered's map_constants.asm and event_constants.asm.

These measure progress for comparison between runs. They are separate from the reward, which can change between runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

from pokered.frame import EVENT_FLAG_BYTES, GameFrame
from pokered.negative_space import require


@dataclass(frozen=True, slots=True)
class Milestone:
    """Reached by standing on a map or by an event flag being set; ``flag`` is (byte, bit), bit 0 lowest."""

    name: str
    map_id: int | None = None
    flag: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        require((self.map_id is None) != (self.flag is None), f"{self.name} needs exactly one condition")
        if self.flag is not None:
            byte, bit = self.flag
            require(0 <= byte < EVENT_FLAG_BYTES, f"{self.name}: flag byte {byte} is outside the array")
            require(0 <= bit < 8, f"{self.name}: flag bit {bit} is outside a byte")

    def reached(self, frame: GameFrame) -> bool:  # nsp: ignore NSP001 a pure predicate; __post_init__ checks the flag
        if self.flag is None:
            return frame.map_id == self.map_id
        byte, bit = self.flag
        return bool(frame.event_flags[byte] >> bit & 1)


# In game order. Pleines et al. 2025 track Viridian Forest through the Cerulean Rocket Thief; the rest extend past them.
MILESTONES: Final[tuple[Milestone, ...]] = (
    Milestone("viridian_city", map_id=0x01),
    Milestone("viridian_forest", map_id=0x33),
    Milestone("pewter_city", map_id=0x02),
    Milestone("beat_brock", flag=(14, 7)),  # EVENT_BEAT_BROCK = 0x77
    Milestone("mt_moon", map_id=0x3B),
    Milestone("cerulean_city", map_id=0x03),
    Milestone("beat_misty", flag=(23, 7)),  # EVENT_BEAT_MISTY = 0xBF
    Milestone("got_ss_ticket", flag=(171, 4)),  # EVENT_GOT_SS_TICKET = 0x55C, the end of Bill's quest
    Milestone("beat_cerulean_rocket", flag=(20, 7)),  # EVENT_BEAT_CERULEAN_ROCKET_THIEF = 0xA7
    Milestone("vermilion_city", map_id=0x05),
    Milestone("got_hm01", flag=(188, 0)),  # EVENT_GOT_HM01 = 0x5E0
    Milestone("beat_lt_surge", flag=(44, 7)),  # EVENT_BEAT_LT_SURGE = 0x167
    Milestone("beat_champion", flag=(288, 1)),  # EVENT_BEAT_CHAMPION_RIVAL = 0x901
)


@dataclass(slots=True)
class MilestoneTracker:
    """The first step each milestone was reached this episode."""

    _first_step: dict[str, int] = field(default_factory=dict)

    def reset(self, first: GameFrame) -> None:  # nsp: ignore NSP001 update validates its step
        self._first_step = {}
        self.update(first, step=0)

    def update(self, frame: GameFrame, *, step: int) -> None:
        require(step >= 0, f"step cannot be negative, got {step}")
        for milestone in MILESTONES:
            if milestone.name not in self._first_step and milestone.reached(frame):
                self._first_step[milestone.name] = step

    def summary(self) -> dict[str, float]:
        """``milestone_<name>`` is 1 or 0; ``steps_to_<name>`` is NaN when it wasn't reached."""
        out: dict[str, float] = {}
        for milestone in MILESTONES:
            step = self._first_step.get(milestone.name)
            out[f"milestone_{milestone.name}"] = float(step is not None)
            out[f"steps_to_{milestone.name}"] = math.nan if step is None else float(step)
        require(len(out) == 2 * len(MILESTONES), "milestone names must be unique")
        return out
