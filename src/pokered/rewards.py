"""Reward terms that pay for first-time progress only.

Every term is latched: it can never pay twice for the same thing, so each has a fixed lifetime maximum per episode
and no repeatable behavior (battling, healing, menu use, walking in circles) can earn more than that. Terms return
this step's reward directly rather than a change in a running total, which is what made upstream's stuck penalty
refund itself (UPSTREAM_FINDINGS.md D1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

from pokered import ram_map
from pokered.frame import GameFrame
from pokered.negative_space import check_finite, require

# (byte offset into the event flags, bit) of EVENT_BOUGHT_MUSEUM_TICKET: buying it is optional, not progress.
MUSEUM_TICKET: Final[tuple[int, int]] = (0xD754 - ram_map.EVENT_FLAGS_START, 0)

# Total team level of the next gym leader, indexed by badges held; the last entry is Lance's team. A party that
# has reached this total has trained enough for the next fight, so further levels pay nothing until it wins a
# badge. Source: pret/pokered data/trainers/parties.asm (Brock 12+14, ..., Giovanni 45+42+44+45+50).
LEVEL_CAP_BY_BADGES: Final[tuple[int, ...]] = (26, 39, 63, 82, 156, 156, 171, 226, 292)


def _require_weight(weight: float) -> None:
    require(weight >= 0.0, f"reward weight must not be negative, got {weight}: rewards only ever pay for progress")


class RewardTerm(Protocol):
    name: str

    def reset(self, first: GameFrame) -> None: ...
    def __call__(self, frame: GameFrame) -> float: ...


@dataclass(slots=True)
class NewTile:
    """Pays once per (map, x, y) per episode, outside battle."""

    weight: float
    name: str = "explore"
    _seen: set[tuple[int, int, int]] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        _require_weight(self.weight)

    @property
    def gained(self) -> int:
        return len(self._seen) - 1

    def reset(self, first: GameFrame) -> None:
        self._seen = {(first.map_id, first.x, first.y)}

    def __call__(self, frame: GameFrame) -> float:
        tile = (frame.map_id, frame.x, frame.y)
        if frame.in_battle or tile in self._seen:
            return 0.0
        self._seen.add(tile)
        return self.weight


def _bit_mask(byte: int, bit: int) -> int:
    return 1 << (8 * byte + bit)


@dataclass(slots=True)
class EventFlagGain:
    """Pays once per event flag, however often the game clears and re-sets it."""

    weight: float
    name: str = "event"
    gained: int = 0
    _seen: int = 0

    def __post_init__(self) -> None:
        _require_weight(self.weight)

    def reset(self, first: GameFrame) -> None:  # nsp: ignore NSP001 GameFrame validates the flag length
        self._seen = int.from_bytes(first.event_flags, "little") | _bit_mask(*MUSEUM_TICKET)
        self.gained = 0

    def __call__(self, frame: GameFrame) -> float:
        current = int.from_bytes(frame.event_flags, "little")
        new = current & ~self._seen
        self._seen |= current
        self.gained += new.bit_count()
        return self.weight * new.bit_count()


@dataclass(slots=True)
class BadgeGain:
    """Pays once per badge."""

    weight: float
    name: str = "badge"
    _seen: int = 0

    def __post_init__(self) -> None:
        _require_weight(self.weight)

    def reset(self, first: GameFrame) -> None:
        self._seen = first.badges

    def __call__(self, frame: GameFrame) -> float:
        new = frame.badges & ~self._seen
        self._seen |= frame.badges
        return self.weight * new.bit_count()


@dataclass(slots=True)
class LevelGain:
    """Pays per party level up to the next gym leader's team total; grinding beyond that pays nothing.

    Uses the highest level total and badge count seen, so depositing and withdrawing Pokemon earns nothing.
    """

    weight: float
    name: str = "level"
    _best_levels: int = 0
    _most_badges: int = 0
    _credited: int = 0

    def __post_init__(self) -> None:
        _require_weight(self.weight)

    def reset(self, first: GameFrame) -> None:  # nsp: ignore NSP001 GameFrame validates its fields
        self._best_levels = sum(first.party_levels)
        self._most_badges = first.badges.bit_count()
        self._credited = self._best_levels

    def __call__(self, frame: GameFrame) -> float:
        self._best_levels = max(self._best_levels, sum(frame.party_levels))
        self._most_badges = max(self._most_badges, frame.badges.bit_count())
        earned = min(self._best_levels, LEVEL_CAP_BY_BADGES[self._most_badges])
        gained = max(earned - self._credited, 0)
        self._credited += gained
        require(self._credited <= max(LEVEL_CAP_BY_BADGES[-1], self._best_levels), "credited levels passed every cap")
        return self.weight * gained


class EpisodeReward:
    """Runs the terms for one episode and reports each one's reward by name."""

    def __init__(self, terms: Sequence[RewardTerm]) -> None:
        names = [term.name for term in terms]
        require(len(names) == len(set(names)), f"duplicate reward term names: {names}")
        self._terms = terms
        self._started = False

    def reset(self, first: GameFrame) -> None:  # nsp: ignore NSP001 terms own their state; step checks the results
        for term in self._terms:
            term.reset(first)
        self._started = True

    def step(self, frame: GameFrame) -> dict[str, float]:
        if not self._started:
            raise RuntimeError("EpisodeReward.step called before reset")
        rewards = {term.name: term(frame) for term in self._terms}
        for name, value in rewards.items():
            check_finite(value, name)
            require(value >= 0.0, f"reward term {name} paid {value}: terms must never go negative")
        return rewards
