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
    """Reached by standing on a map, an event flag being set, an item in the bag or a move in the party.

    ``flag`` is (byte, bit), bit 0 lowest. Items and moves cover required steps the game sets no flag for.
    """

    name: str
    map_id: int | None = None
    flag: tuple[int, int] | None = None
    item: int | None = None
    move: int | None = None

    def __post_init__(self) -> None:
        conditions = (self.map_id, self.flag, self.item, self.move)
        require(sum(condition is not None for condition in conditions) == 1, f"{self.name} needs exactly one condition")
        if self.flag is not None:
            byte, bit = self.flag
            require(0 <= byte < EVENT_FLAG_BYTES, f"{self.name}: flag byte {byte} is outside the array")
            require(0 <= bit < 8, f"{self.name}: flag bit {bit} is outside a byte")

    def reached(self, frame: GameFrame) -> bool:  # nsp: ignore NSP001 a pure predicate; __post_init__ checks the flag
        if self.item is not None:
            return self.item in frame.bag_items
        if self.move is not None:
            return self.move in frame.party_moves
        if self.flag is None:
            return frame.map_id == self.map_id
        byte, bit = self.flag
        return bool(frame.event_flags[byte] >> bit & 1)


def _event(event_id: int) -> tuple[int, int]:
    """(byte, bit) of an id from event_constants.asm."""
    require(0 <= event_id < EVENT_FLAG_BYTES * 8, f"event id {event_id:#x} is outside the array")
    return event_id // 8, event_id % 8


# In required order, as far as one exists: docs/evidence/events/EVENT_PROGRESSION.md, which checks these against the ROM.
MILESTONES: Final[tuple[Milestone, ...]] = (
    Milestone("viridian_city", map_id=0x01),
    Milestone("viridian_forest", map_id=0x33),
    Milestone("pewter_city", map_id=0x02),
    Milestone("beat_brock", flag=_event(0x077)),
    Milestone("mt_moon", map_id=0x3B),
    Milestone("cerulean_city", map_id=0x03),
    Milestone("beat_misty", flag=_event(0x0BF)),
    Milestone("beat_cerulean_rival", flag=_event(0x098)),
    Milestone("beat_route24_rocket", flag=_event(0x541)),
    Milestone("bill_restored", flag=_event(0x550)),
    Milestone("got_ss_ticket", flag=_event(0x55C)),
    Milestone("trashed_house", map_id=0x3E),
    Milestone("beat_cerulean_rocket", flag=_event(0x0A7)),
    Milestone("route_5", map_id=0x10),
    Milestone("route_6", map_id=0x11),
    Milestone("vermilion_city", map_id=0x05),
    Milestone("ss_anne", map_id=0x5F),
    Milestone("got_hm01", flag=_event(0x5E0)),
    Milestone("knows_cut", move=0x0F),
    Milestone("beat_lt_surge", flag=_event(0x167)),
    Milestone("got_bicycle", flag=_event(0x0C0)),
    Milestone("celadon_city", map_id=0x06),
    Milestone("beat_erika", flag=_event(0x1A9)),
    Milestone("beat_hideout_giovanni", flag=_event(0x6A7)),
    Milestone("has_silph_scope", item=0x48),
    Milestone("rescued_fuji", flag=_event(0x4CF)),
    Milestone("got_poke_flute", flag=_event(0x128)),
    Milestone("beat_silph_giovanni", flag=_event(0x78F)),
    Milestone("beat_sabrina", flag=_event(0x361)),
    Milestone("beat_koga", flag=_event(0x259)),
    Milestone("got_hm03", flag=_event(0x880)),
    Milestone("got_hm04", flag=_event(0x238)),
    Milestone("beat_blaine", flag=_event(0x299)),
    Milestone("beat_giovanni", flag=_event(0x051)),
    Milestone("beat_lorelei", flag=_event(0x8E1)),
    Milestone("beat_champion", flag=_event(0x901)),
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
