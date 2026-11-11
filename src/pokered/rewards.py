"""Reward terms that pay for first-time progress only.

Every term is latched: it can never pay twice for the same thing, so each has a fixed lifetime maximum per episode
and no repeatable behavior (battling, healing, menu use, walking in circles) can earn more than that. Terms return
this step's reward directly rather than a change in a running total, which is what made upstream's stuck penalty
refund itself (UPSTREAM_FINDINGS.md D1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Final, Protocol

from pokered import ram_map
from pokered.frame import GameFrame
from pokered.negative_space import check_finite, require

# (byte offset into the event flags, bit) of EVENT_BOUGHT_MUSEUM_TICKET: buying it is optional, not progress.
MUSEUM_TICKET: Final[tuple[int, int]] = (0xD754 - ram_map.EVENT_FLAGS_START, 0)

# Highest level on the next gym leader's team, indexed by badges held; the last entry is Lance's ace. Source:
# pret/pokered data/trainers/parties.asm (Brock's Onix 14, ..., Giovanni's Rhydon 50, Lance's Dragonite 62).
GYM_ACE_LEVEL_BY_BADGES: Final[tuple[int, ...]] = (14, 21, 24, 29, 43, 43, 47, 50, 62)
# No Pokemon earns reward past the next ace plus this margin until the next badge. run2 and run3, capped at the ace,
# beat Misty in 8% and 0% of episodes with a starter near 24; the first run's looser cap reached about 28 and 59-78%.
LEVEL_MARGIN_OVER_ACE: Final[int] = 5
LEVEL_CAP_BY_BADGES: Final[tuple[int, ...]] = tuple(ace + LEVEL_MARGIN_OVER_ACE for ace in GYM_ACE_LEVEL_BY_BADGES)
SPECIES_COUNT: Final[int] = 151
BALL_CAP: Final[int] = 5


def _require_weight(weight: float) -> None:
    require(weight >= 0.0, f"reward weight must not be negative, got {weight}: rewards only ever pay for progress")


class RewardTerm(Protocol):
    name: str
    signed: ClassVar[bool]  # only a signed term may pay a negative reward

    def reset(self, first: GameFrame) -> None: ...
    def __call__(self, frame: GameFrame) -> float: ...


@dataclass(slots=True)
class NewTile:
    """Pays once per (map, x, y) per episode, outside battle."""

    weight: float
    name: str = "explore"
    signed: ClassVar[bool] = False
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
    signed: ClassVar[bool] = False
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
    signed: ClassVar[bool] = False
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
class _PartyJoin:
    """Says whether a Pokemon is joining the party on this step.

    The game raises the party count, shows the nickname prompt, and only then writes the new Pokemon's moves and level
    (pret engine/pokemon/add_mon.asm), so a joining Pokemon reads level 0 for some steps. A step joins when the party
    grows, when a member reads level 0, or when one did on the step before.
    """

    _party_size: int = 0
    _was_unwritten: bool = False

    def reset(self, first: GameFrame) -> None:
        self._party_size = len(first.party_levels)
        self._was_unwritten = 0 in first.party_levels
        require(
            self._party_size <= ram_map.MAX_PARTY, f"a party holds at most {ram_map.MAX_PARTY}, got {self._party_size}"
        )

    def __call__(self, frame: GameFrame) -> bool:
        unwritten = 0 in frame.party_levels
        joined = len(frame.party_levels) > self._party_size or unwritten or self._was_unwritten
        self._party_size = len(frame.party_levels)
        self._was_unwritten = unwritten
        return joined


@dataclass(slots=True)
class LevelGain:
    """Pays per level gained up to the next gym's ace plus a margin, for each of the six party places.

    Places rank the party by level, highest first, and each keeps the highest level it has ever held. That is what
    stops the term being farmed: storing a Pokemon costs nothing, while withdrawing it or levelling a replacement to
    the same level pays nothing, so the lifetime total is at most six times the last cap. No identity is tracked.
    A Pokemon pays nothing for the level it joins at, so a bought or caught one pays only for levels it then gains:
    on a joining step (see ``_PartyJoin``) every place is credited without pay.
    """

    weight: float
    name: str = "level"
    signed: ClassVar[bool] = False
    _best: list[int] = field(default_factory=list, repr=False)
    _credited: list[int] = field(default_factory=list, repr=False)
    _most_badges: int = 0
    _join: _PartyJoin = field(default_factory=_PartyJoin, repr=False)

    def __post_init__(self) -> None:
        _require_weight(self.weight)

    def reset(self, first: GameFrame) -> None:
        self._best = _ranked_levels(first.party_levels)
        self._credited = list(self._best)
        self._most_badges = first.badges.bit_count()
        self._join.reset(first)
        require(len(self._best) == ram_map.MAX_PARTY, f"expected {ram_map.MAX_PARTY} places, got {len(self._best)}")

    def __call__(self, frame: GameFrame) -> float:
        self._most_badges = max(self._most_badges, frame.badges.bit_count())
        cap = LEVEL_CAP_BY_BADGES[self._most_badges]
        joined = self._join(frame)
        gained = 0
        for place, level in enumerate(_ranked_levels(frame.party_levels)):
            self._best[place] = max(self._best[place], level)
            earned = min(self._best[place], cap)
            if earned > self._credited[place]:
                gained += 0 if joined else earned - self._credited[place]
                self._credited[place] = earned
        require(gained >= 0, f"a level gain cannot be negative, got {gained}")
        return self.weight * gained


def _ranked_levels(levels: tuple[int, ...]) -> list[int]:
    """Levels highest first, with zeros for empty places."""
    require(len(levels) <= ram_map.MAX_PARTY, f"a party holds at most {ram_map.MAX_PARTY}, got {len(levels)}")
    return sorted(levels, reverse=True) + [0] * (ram_map.MAX_PARTY - len(levels))


# The Pokedex owned bits hold 152 places for 151 species; the last bit is never a species.
_ALL_SPECIES: Final[int] = (1 << SPECIES_COUNT) - 1


@dataclass(slots=True)
class NewSpecies:
    """Pays once per species owned, however it was got: a second Magikarp pays nothing."""

    weight: float
    name: str = "species"
    signed: ClassVar[bool] = False
    _seen: int = 0

    def __post_init__(self) -> None:
        _require_weight(self.weight)

    def reset(self, first: GameFrame) -> None:
        self._seen = int.from_bytes(first.pokedex_owned, "little") & _ALL_SPECIES

    def __call__(self, frame: GameFrame) -> float:
        current = int.from_bytes(frame.pokedex_owned, "little") & _ALL_SPECIES
        new = current & ~self._seen
        self._seen |= current
        require(self._seen.bit_count() <= SPECIES_COUNT, f"owned more than {SPECIES_COUNT} species")
        return self.weight * new.bit_count()


def _full_hp(frame: GameFrame) -> bool:
    return all(hp >= 1.0 for hp in frame.party_hp)


@dataclass(slots=True)
class CenterHeal:
    """Pays once per respawn point, the first time a nurse heal there restores HP.

    Accepting a heal writes the map outside (wLastMap) into the respawn point, then heals the party (pret
    engine/events/pokecenter.asm). So a heal is a step that takes a hurt party to full HP inside a Pokemon Center
    whose respawn point is this one. A blackout also heals, but on the battle map before warping
    (engine/events/black_out.asm), and the other healers (Mom, Oak, Pokemon Tower, Silph Co.) are not in centers. A heal
    at full HP sets the respawn point but pays nothing and leaves that center unpaid. The respawn point held at reset
    counts as paid, so the lifetime total is one payment per center.
    """

    weight: float
    name: str = "heal"
    signed: ClassVar[bool] = False
    _paid: set[int] = field(default_factory=set, repr=False)
    _hurt: bool = False

    def __post_init__(self) -> None:
        _require_weight(self.weight)

    def reset(self, first: GameFrame) -> None:
        self._paid = {first.respawn_map}
        self._hurt = not _full_hp(first)
        require(len(first.party_hp) > 0, "an episode needs a party to heal")

    def __call__(self, frame: GameFrame) -> float:
        healed = (
            self._hurt
            and _full_hp(frame)
            and not frame.in_battle
            and frame.map_id in ram_map.POKEMON_CENTER_MAPS
            and frame.respawn_map == frame.last_map
        )
        self._hurt = not _full_hp(frame)
        if not healed or frame.respawn_map in self._paid:
            return 0.0
        before = len(self._paid)
        self._paid.add(frame.respawn_map)
        require(len(self._paid) == before + 1, f"respawn point {frame.respawn_map} was paid twice")
        return self.weight


# Move effects whose damage does not scale with power, or that cost the user: pret constants/move_effect_constants.asm.
EXPLODE_EFFECT: Final[int] = 0x07
OHKO_EFFECT: Final[int] = 0x26
SUPER_FANG_EFFECT: Final[int] = 0x28
SPECIAL_DAMAGE_EFFECT: Final[int] = 0x29
NO_SCORE_EFFECTS: Final[frozenset[int]] = frozenset(
    {EXPLODE_EFFECT, OHKO_EFFECT, SUPER_FANG_EFFECT, SPECIAL_DAMAGE_EFFECT}
)
# Moves that take two turns to land one hit and can be hit in between: charge first (Solar Beam, Skull Bash) or recharge
# after (Hyper Beam, whose skipped recharge on a knockout the agent is not expected to use). Fly and Dig leave the user
# out of reach on the first turn, so they score in full; Dig shares the charge effect and the engine checks its id.
CHARGE_EFFECT: Final[int] = 0x27
HYPER_BEAM_EFFECT: Final[int] = 0x50
TWO_TURN_EFFECTS: Final[frozenset[int]] = frozenset({CHARGE_EFFECT, HYPER_BEAM_EFFECT})
DIG: Final[int] = 91  # pret constants/move_constants.asm
STAB: Final[float] = 1.5


@dataclass(frozen=True, slots=True)
class MoveScore:
    type: int
    score: float


def move_scores(moves: Mapping[int, ram_map.Move]) -> dict[int, MoveScore]:
    """Expected damage per turn before type matchups: power times accuracy, halved for moves that charge or recharge.

    Power 1 marks damage the game computes another way (Counter, Seismic Toss, Dragon Rage, one-hit KOs), and status
    moves have power 0: all of these score nothing, as do moves that make the user faint.
    """
    scores = {}
    for move_id, move in moves.items():
        scaled = move.power > 1 and move.effect not in NO_SCORE_EFFECTS
        score = move.power * move.accuracy / 255 if scaled else 0.0
        halved = move.effect in TWO_TURN_EFFECTS and move_id != DIG
        scores[move_id] = MoveScore(move.type, score / 2 if halved else score)
        require(0.0 <= scores[move_id].score <= 255.0, f"move {move_id} scored {scores[move_id].score}")
    return scores


@dataclass(slots=True)
class MoveGain:
    """Pays the change in the party's total move score, so learning a stronger move pays and losing one costs.

    A Pokemon's total is the sum of its four moves' scores, times 1.5 for a move of its own type, and the party's is
    the sum over members. Each step pays the members it shares with the step before: their totals now minus the whole
    party's total then. So storing a Pokemon costs its total and reordering pays nothing. A joining Pokemon (a new slot,
    or one whose level read 0 now or on the step before, while the game writes it: pret engine/pokemon/add_mon.asm) is
    credited without pay. Payments add up to the final total minus the first, minus those credits, which are never
    negative, so no sequence of frames pays more for being repeated, and the lifetime total is at most six Pokemon of
    four best moves.
    """

    weight: float
    scores: Mapping[int, MoveScore]
    name: str = "moves"
    signed: ClassVar[bool] = True
    _totals: list[float | None] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        _require_weight(self.weight)
        require(len(self.scores) > 0, "MoveGain needs the move table")

    def reset(self, first: GameFrame) -> None:
        self._totals = self._member_totals(first, established=[False] * len(first.party_levels))
        require(len(self._totals) <= ram_map.MAX_PARTY, f"a party holds at most {ram_map.MAX_PARTY}, got {len(self._totals)}")

    def __call__(self, frame: GameFrame) -> float:
        before = self._totals
        established = [slot < len(before) and before[slot] is not None for slot in range(len(frame.party_levels))]
        now = self._member_totals(frame, established=established)
        kept = sum(total for total, old in zip(now, established, strict=True) if total is not None and old)
        change = kept - sum(total for total in before if total is not None)
        credited = sum(total for total, old in zip(now, established, strict=True) if total is not None and not old)
        require(credited >= 0.0, f"a joining Pokemon's credit cannot be negative, got {credited}")
        self._totals = now
        return self.weight * change

    def _member_totals(self, frame: GameFrame, *, established: list[bool]) -> list[float | None]:
        """Each slot's total, or None for a slot whose level reads 0. Only an established member must know its moves:
        a joining slot may hold unwritten bytes, which count as nothing."""
        size = len(frame.party_levels)
        moves_per_mon = ram_map.MOVES_PER_MON
        require(len(frame.party_moves) == moves_per_mon * size, f"expected {moves_per_mon} moves per Pokemon, got {frame.party_moves}")
        require(len(frame.party_types) == ram_map.TYPES_PER_MON * size, f"expected 2 types per Pokemon, got {frame.party_types}")
        totals: list[float | None] = []
        for mon in range(size):
            if frame.party_levels[mon] == 0:
                totals.append(None)
                continue
            types = frame.party_types[ram_map.TYPES_PER_MON * mon : ram_map.TYPES_PER_MON * (mon + 1)]
            total = 0.0
            for move_id in frame.party_moves[moves_per_mon * mon : moves_per_mon * (mon + 1)]:
                move = self.scores.get(move_id)
                require(move_id == 0 or move is not None or not established[mon], f"move {move_id} is not in the move table")
                if move is not None:
                    total += move.score * (STAB if move.type in types else 1.0)
            totals.append(total)
        return totals


@dataclass(slots=True)
class HiddenEventGain:
    """Pays once per hidden event the game matches: a PC, sign, poster or trash can pressed with A (pret
    data/events/hidden_events.asm lists 212). Bill's Cell Separator and Lt. Surge's trash-can switches are hidden
    events. A match at reset, from before the episode, pays nothing."""

    weight: float
    name: str = "hidden"
    signed: ClassVar[bool] = False
    _seen: set[tuple[int, int]] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        _require_weight(self.weight)

    def reset(self, first: GameFrame) -> None:
        self._seen = set(first.hidden_events)
        require(len(self._seen) <= len(first.hidden_events), "reset cannot see more events than the frame holds")

    def __call__(self, frame: GameFrame) -> float:
        new = set(frame.hidden_events) - self._seen
        self._seen |= new
        require(new <= self._seen, "every hidden event paid must be remembered")
        return self.weight * len(new)


@dataclass(slots=True)
class NewMap:
    """Pays once per map id entered (pret map_constants.asm has 248), and nothing for the map the episode starts on.
    run5 parked in the Pokemon Fan Club once it had the Bike Voucher, a short walk from the S.S. Anne's dock: a tile
    pays 1/400 of an event, so no unentered door was worth leaving for."""

    weight: float
    name: str = "map"
    signed: ClassVar[bool] = False
    _seen: set[int] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        _require_weight(self.weight)

    def reset(self, first: GameFrame) -> None:
        self._seen = {first.map_id}

    def __call__(self, frame: GameFrame) -> float:
        if frame.map_id in self._seen:
            return 0.0
        self._seen.add(frame.map_id)
        return self.weight


def ss_anne_rival_beaten(frame: GameFrame) -> bool:
    """The S.S. Anne rival, the one story fight that sets no event flag (pret scripts/SSAnne2F.asm)."""
    return frame.ss_anne_2f_script >= ram_map.SS_ANNE_RIVAL_BEATEN


def knows_cut(frame: GameFrame) -> bool:
    return ram_map.CUT in frame.party_moves


def party_has_any(species: frozenset[int]) -> Callable[[GameFrame], bool]:
    """Whether some party Pokemon is one of these species (internal IDs). A Pokemon in the PC can't be taught."""
    require(species, "a species condition needs at least one species")
    return lambda frame: any(member in species for member in frame.party_species)


@dataclass(slots=True)
class FirstTime:
    """Pays once, the first step a condition holds. A condition already true at reset pays nothing."""

    name: str
    weight: float
    condition: Callable[[GameFrame], bool]
    signed: ClassVar[bool] = False
    _met: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        _require_weight(self.weight)
        require(self.name, "a term needs a name")

    def reset(self, first: GameFrame) -> None:
        self._met = self.condition(first)

    def __call__(self, frame: GameFrame) -> float:
        if self._met or not self.condition(frame):
            return 0.0
        self._met = True
        return self.weight


@dataclass(slots=True)
class BallStock:
    """Pays per ball for the most balls held at once, up to BALL_CAP: at a 33.6% catch rate (a Poke Ball on a full-HP
    Oddish), five throws catch one 87% of the time. Throwing balls and buying them back pays nothing."""

    weight: float
    name: str = "balls"
    signed: ClassVar[bool] = False
    _most: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        _require_weight(self.weight)

    def reset(self, first: GameFrame) -> None:
        self._most = min(first.poke_balls, BALL_CAP)

    def __call__(self, frame: GameFrame) -> float:
        held = min(frame.poke_balls, BALL_CAP)
        gained = max(held - self._most, 0)
        self._most += gained
        require(self._most <= BALL_CAP, f"the most balls held, {self._most}, passed the cap")
        return self.weight * gained


def _trees(frame: GameFrame) -> set[tuple[int, int, int, int]]:
    """Where the player stood and faced for each tree cut in this frame; grass is left out."""
    return {(map_id, x, y, facing) for map_id, x, y, facing, tile in frame.cuts if tile in ram_map.CUT_TREE_TILES}


@dataclass(slots=True)
class TreeCut:
    """Pays once per tree cut, told apart by where the player stood and faced; grass pays nothing. Trees grow back
    when their map reloads, so cutting one again pays nothing."""

    weight: float
    name: str = "trees"
    signed: ClassVar[bool] = False
    _seen: set[tuple[int, int, int, int]] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        _require_weight(self.weight)

    def reset(self, first: GameFrame) -> None:
        self._seen = _trees(first)
        require(len(self._seen) <= len(first.cuts), "reset cannot see more trees than the frame holds")

    def __call__(self, frame: GameFrame) -> float:
        new = _trees(frame) - self._seen
        self._seen |= new
        require(new <= self._seen, "every tree paid must be remembered")
        return self.weight * len(new)


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
        rewards = {}
        for term in self._terms:
            value = term(frame)
            check_finite(value, term.name)
            require(value >= 0.0 or term.signed, f"reward term {term.name} paid {value}: only a signed term may go negative")
            rewards[term.name] = value
        return rewards
