"""What one episode did: flags, fights, maps, buttons and stalls. Only observes, so it can never affect a reward."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from pokered import ram_map
from pokered.buttons import BUTTONS
from pokered.frame import GameFrame
from pokered.negative_space import require

# What ends a stall. Tiles don't count: an agent can wander for a long time, still paying for new tiles, without any
# story progress. A new map does: there are only 248, and reaching the next one is how the story moves on.
PROGRESS_TERMS: Final[tuple[str, ...]] = ("event", "badge", "level", "species", "heal", "moves", "map")
PHASE_STEPS: Final[int] = 10_240
_TRAINER_OPPONENT_BASE: Final[int] = 200

# Index is the trainer class, from pret/pokered constants/trainer_constants.asm.
_TRAINER_CLASSES: Final[tuple[str, ...]] = (
    "NOBODY", "YOUNGSTER", "BUG_CATCHER", "LASS", "SAILOR", "JR_TRAINER_M", "JR_TRAINER_F", "POKEMANIAC", "SUPER_NERD",
    "HIKER", "BIKER", "BURGLAR", "ENGINEER", "UNUSED_JUGGLER", "FISHER", "SWIMMER", "CUE_BALL", "GAMBLER", "BEAUTY",
    "PSYCHIC_TR", "ROCKER", "JUGGLER", "TAMER", "BIRD_KEEPER", "BLACKBELT", "RIVAL1", "PROF_OAK", "CHIEF", "SCIENTIST",
    "GIOVANNI", "ROCKET", "COOLTRAINER_M", "COOLTRAINER_F", "BRUNO", "BROCK", "MISTY", "LT_SURGE", "ERIKA", "KOGA",
    "BLAINE", "SABRINA", "GENTLEMAN", "RIVAL2", "RIVAL3", "LORELEI", "CHANNELER", "AGATHA", "LANCE",
)  # fmt: skip


def opponent_label(battle_type: int, opponent: int, trainer_number: int) -> str:
    """``wild``, or a trainer as CLASS#number. Every trainer with the same class and number has the same team."""
    require(battle_type in (ram_map.BATTLE_WILD, ram_map.BATTLE_TRAINER), f"not a battle in progress: {battle_type}")
    if battle_type == ram_map.BATTLE_WILD:
        return "wild"
    index = opponent - _TRAINER_OPPONENT_BASE
    if not 0 <= index < len(_TRAINER_CLASSES):
        return "unknown_trainer"
    return f"{_TRAINER_CLASSES[index]}#{trainer_number}"


@dataclass(frozen=True, slots=True)
class Fight:
    opponent: str
    map_id: int
    step: int
    party_max_level: int
    party_size: int
    lost: bool = False


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The game at one moment: ``state`` reloads in PyBoy, so a stuck position can be replayed and inspected."""

    kind: str
    step: int
    dry_steps: int
    map_id: int
    x: int
    y: int
    in_battle: bool
    party_levels: tuple[int, ...]
    screen: np.ndarray
    state: bytes


@dataclass(frozen=True, slots=True)
class EpisodeDetail:
    length: int
    flag_steps: dict[int, int]
    map_steps: dict[int, int]
    fights: tuple[Fight, ...]
    phase_rewards: dict[tuple[str, int], float]
    action_counts: tuple[int, ...]
    last_stall: Snapshot | None
    end: Snapshot


@dataclass(frozen=True, slots=True)
class EpisodeReport:
    """``summary`` holds this episode's numbers; NaN means a milestone it never reached."""

    summary: dict[str, float]
    detail: EpisodeDetail


@dataclass(slots=True)
class EpisodeTelemetry:
    stall_steps: int
    flag_steps: dict[int, int] = field(default_factory=dict)
    map_steps: dict[int, int] = field(default_factory=dict)
    fights: list[Fight] = field(default_factory=list)
    phase_rewards: dict[tuple[str, int], float] = field(default_factory=dict)
    action_counts: list[int] = field(default_factory=list)
    blackouts: int = 0
    late_progress: int = 0  # progress that came after more than half the stall limit: a sign the limit is tight
    steps: int = 0
    _last_story_step: int = 0
    _seen: int = 0
    _in_fight: bool = False
    _in_blackout: bool = False

    def __post_init__(self) -> None:
        require(self.stall_steps >= 1, f"the stall limit must be positive, got {self.stall_steps}")

    @property
    def dry_steps(self) -> int:
        """Steps since the last progress: an event, badge, level or species."""
        return self.steps - self._last_story_step

    def reset(self, first: GameFrame) -> None:  # nsp: ignore NSP001 GameFrame validates the flag length
        self.flag_steps, self.map_steps, self.fights, self.phase_rewards = {}, {}, [], {}
        self.action_counts = [0] * len(BUTTONS)
        self.blackouts = self.late_progress = self.steps = self._last_story_step = 0
        self._seen = int.from_bytes(first.event_flags, "little")
        self._in_fight = self._in_blackout = False

    def update(self, frame: GameFrame, *, action: int, terms: dict[str, float]) -> bool:
        """Record one step. True on the step a stall reaches the limit, once per stall."""
        require(0 <= action < len(BUTTONS), f"action {action} is not a button")
        self.steps += 1
        self.action_counts[action] += 1
        self.map_steps[frame.map_id] = self.map_steps.get(frame.map_id, 0) + 1
        self._record_flags(frame)
        self._record_fight(frame)
        phase = (self.steps - 1) // PHASE_STEPS
        for name, value in terms.items():
            if value != 0.0:  # a signed term's cost is recorded too, so the report shows its net
                self.phase_rewards[(name, phase)] = self.phase_rewards.get((name, phase), 0.0) + value
        if any(terms.get(name, 0.0) > 0.0 for name in PROGRESS_TERMS):
            if self.steps - self._last_story_step > self.stall_steps // 2:
                self.late_progress += 1
            self._last_story_step = self.steps
        require(self.dry_steps >= 0, f"dry steps cannot be negative, got {self.dry_steps}")
        return self.dry_steps == self.stall_steps

    def _record_flags(self, frame: GameFrame) -> None:
        current = int.from_bytes(frame.event_flags, "little")
        new = current & ~self._seen
        self._seen |= current
        while new:  # Bounded: each pass clears one of at most 2,560 bits.
            lowest = new & -new
            self.flag_steps[lowest.bit_length() - 1] = self.steps
            new ^= lowest
        require(current & ~self._seen == 0, "every set flag must have been seen")

    def _record_fight(self, frame: GameFrame) -> None:
        require(not self._in_fight or self.fights, "a fight in progress must be on record")
        if frame.battle_type == ram_map.BATTLE_LOST:
            self._in_fight = False
            if not self._in_blackout:
                self.blackouts += 1
                self._mark_last_fight_lost()
            self._in_blackout = True
            return
        self._in_blackout = False
        if frame.battle_type == ram_map.BATTLE_NONE:
            self._in_fight = False
        elif not self._in_fight:
            self._in_fight = True
            self.fights.append(
                Fight(
                    opponent=opponent_label(frame.battle_type, frame.opponent, frame.trainer_number),
                    map_id=frame.map_id,
                    step=self.steps,
                    party_max_level=max(frame.party_levels, default=0),
                    party_size=len(frame.party_levels),
                )
            )

    def _mark_last_fight_lost(self) -> None:
        # The game clears the battle byte when a battle ends, then writes LOST_BATTLE: pret home/overworld.asm.
        if self.fights:
            last = self.fights[-1]
            self.fights[-1] = Fight(last.opponent, last.map_id, last.step, last.party_max_level, last.party_size, lost=True)
