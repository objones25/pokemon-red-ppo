"""Typed readers for Pokemon Red's RAM and ROM data tables.

Only valid for the international Pokemon Red ROM (SHA-1 in ``ROM_SHA1``).
See RAM_MAP_REPORT.md for how each address was verified and for training traps.

``mem`` can be anything indexable by address, such as ``pyboy.memory``:

    state = read_state(pyboy.memory)
    if state.battle is not None and at_fight_menu(pyboy.memory):
        ...

Read after the intro has initialised RAM. Before that, money holds garbage and
``read_money`` raises ``ValueError``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pokered.negative_space import require, unreachable

ROM_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"


class Memory(Protocol):
    def __getitem__(self, address: int, /) -> int: ...


# Player and world
MONEY = 0xD347  # 3 bytes, BCD
OPTIONS = 0xD355  # low bits: text delay; bit 7: battle animations off
OPTION_NO_BATTLE_ANIMATION = 0x80
BADGES = 0xD356
PLAYER_ID = 0xD359  # 2 bytes
MAP_ID = 0xD35E
Y_POS = 0xD361
X_POS = 0xD362
FACING = 0xC109
POKEDEX_OWNED = 0xD2F7
POKEDEX_SEEN = 0xD30A
POKEDEX_BYTES = 19
BAG_COUNT = 0xD31D
PC_ITEM_COUNT = 0xD53A
BAG_CAPACITY = 20
PC_ITEM_CAPACITY = 50
TILEMAP = 0xC3A0  # 20 columns x 18 rows
MENU_CURSOR = 0xCC26

# Party
PARTY_COUNT = 0xD163
PARTY_SPECIES = 0xD164
PARTY_MONS = 0xD16B
MON_SIZE = 44
MAX_PARTY = 6
MOVES_PER_MON = 4

# Offsets inside a 44-byte party or enemy-party struct
OFF_SPECIES = 0
OFF_HP = 1
OFF_STATUS = 4
OFF_TYPES = 5
OFF_MOVES = 8
OFF_OT_ID = 12
OFF_EXP = 14
OFF_PP = 29
OFF_LEVEL = 33
OFF_MAX_HP = 34
OFF_STATS = 36

# Battle
BATTLE_TYPE = 0xD057
BATTLE_NONE, BATTLE_WILD, BATTLE_TRAINER, BATTLE_LOST = 0, 1, 2, 0xFF
OPPONENT = 0xD059  # wild species, or trainer class + 200
ENEMY_LEVEL_SETUP = 0xD127
TRAINER_NUMBER = 0xD05D
TRAINER_CLASS = 0xD031
TURN_COUNT = 0xCCD5
PLAYER_SELECTED_MOVE = 0xCCDC
ENEMY_SELECTED_MOVE = 0xCCDD
PLAYER_STAGES = 0xCD1A
ENEMY_STAGES = 0xCD2E
ENEMY_PARTY_COUNT = 0xD89C
ENEMY_PARTY_SPECIES = 0xD89D
ENEMY_PARTY_MONS = 0xD8A4

# wEventFlags is 0xD747 up to wLinkEnemyTrainerName at 0xD887 in the pokered symbol file: 320 bytes.
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_BYTES = 320

# ROM file offsets
ROM_MOVES = 0x38000
ROM_BASE_STATS = 0x383DE
ROM_MEW_STATS = 0x425B
ROM_TYPE_CHART = 0x3E474
ROM_INTERNAL_TO_DEX = 0x41024
BASE_STATS_SIZE = 28
MOVE_SIZE = 6
NUM_MOVES = 165
NUM_INTERNAL_IDS = 190

_TEXT_BOX_CORNER = 0x79
_MENU_ARROW = 0xED
_LETTER_F = 0x85


def read_u16(mem: Memory, address: int) -> int:
    """Read a big-endian 16-bit value."""
    return mem[address] << 8 | mem[address + 1]


def read_u24(mem: Memory, address: int) -> int:
    """Read a big-endian 24-bit value."""
    return mem[address] << 16 | mem[address + 1] << 8 | mem[address + 2]


def read_bcd(mem: Memory, address: int, num_bytes: int) -> int:
    """Read packed BCD. Raises ValueError if a nibble is not a decimal digit."""
    require(num_bytes > 0, f"BCD needs at least one byte, got {num_bytes}")
    value = 0
    for offset in range(num_bytes):
        byte = mem[address + offset]
        high, low = byte >> 4, byte & 0x0F
        if high > 9 or low > 9:
            raise ValueError(f"invalid BCD byte {byte:#04x} at {address + offset:#06x}")
        value = value * 100 + high * 10 + low
    return value


def read_money(mem: Memory) -> int:
    return read_bcd(mem, MONEY, 3)


def _popcount(mem: Memory, address: int, num_bytes: int) -> int:
    return sum(mem[address + i].bit_count() for i in range(num_bytes))


def badge_bits(mem: Memory) -> tuple[bool, ...]:  # nsp: ignore NSP001 one byte always yields eight flags
    """Return eight flags, Boulder first and Earth last."""
    value = mem[BADGES]
    return tuple(bool(value >> bit & 1) for bit in range(8))


def badge_count(mem: Memory) -> int:
    return mem[BADGES].bit_count()


def pokedex_owned_count(mem: Memory) -> int:
    return _popcount(mem, POKEDEX_OWNED, POKEDEX_BYTES)


def pokedex_seen_count(mem: Memory) -> int:
    return _popcount(mem, POKEDEX_SEEN, POKEDEX_BYTES)


def event_flag_count(mem: Memory) -> int:
    """Count set event flags."""
    return _popcount(mem, EVENT_FLAGS_START, EVENT_FLAGS_BYTES)


def tile(mem: Memory, row: int, col: int) -> int:
    require(0 <= row < 18, f"row {row} is off the tilemap")
    require(0 <= col < 20, f"column {col} is off the tilemap")
    return mem[TILEMAP + 20 * row + col]


def text_box_open(mem: Memory) -> bool:
    """True when the standard bottom text box is on screen."""
    return tile(mem, 12, 0) == _TEXT_BOX_CORNER


def at_fight_menu(mem: Memory) -> bool:
    """True when the main battle menu has its arrow on FIGHT.

    This is a safe decision point: battle fields have finished loading and are
    not mid-animation.
    """
    return (
        is_active_battle(mem)
        and tile(mem, 14, 9) == _MENU_ARROW
        and tile(mem, 14, 10) == _LETTER_F
    )


def is_active_battle(mem: Memory) -> bool:
    return mem[BATTLE_TYPE] in (BATTLE_WILD, BATTLE_TRAINER)


def blacked_out(mem: Memory) -> bool:
    """True for the roughly 100 frames between losing and the warp and heal."""
    return mem[BATTLE_TYPE] == BATTLE_LOST


@dataclass(frozen=True, slots=True)
class Status:
    """A decoded status byte. ``sleep_turns`` is 0 when not asleep."""

    sleep_turns: int
    poisoned: bool
    burned: bool
    frozen: bool
    paralyzed: bool

    @classmethod
    def from_byte(cls, value: int) -> Status:
        require(0 <= value <= 0xFF, f"status byte out of range: {value}")
        return cls(
            sleep_turns=value & 0b111,
            poisoned=bool(value & 0x08),
            burned=bool(value & 0x10),
            frozen=bool(value & 0x20),
            paralyzed=bool(value & 0x40),
        )

    @property
    def healthy(self) -> bool:
        return not (
            self.sleep_turns or self.poisoned or self.burned or self.frozen or self.paralyzed
        )


@dataclass(frozen=True, slots=True)
class Stats:
    attack: int
    defense: int
    speed: int
    special: int


def _read_stats(mem: Memory, address: int) -> Stats:
    return Stats(*(read_u16(mem, address + 2 * i) for i in range(4)))


def _read_pp(mem: Memory, address: int) -> tuple[int, int, int, int]:
    """Read current PP, dropping the PP Up count stored in the top two bits."""
    return tuple(mem[address + i] & 0x3F for i in range(4))  # type: ignore[return-value]


def _read_moves(mem: Memory, address: int) -> tuple[int, int, int, int]:
    return tuple(mem[address + i] for i in range(4))  # type: ignore[return-value]


def _fraction(current: int, maximum: int) -> float:  # nsp: ignore NSP001 total and clamped by construction
    if maximum == 0:
        return 0.0
    return min(current / maximum, 1.0)


@dataclass(frozen=True, slots=True)
class Pokemon:
    """A 44-byte party or trainer struct. ``species`` is an internal ID, not a dex number."""

    species: int
    hp: int
    max_hp: int
    level: int
    status: Status
    types: tuple[int, int]
    moves: tuple[int, int, int, int]
    pp: tuple[int, int, int, int]
    ot_id: int
    exp: int
    stats: Stats

    @property
    def hp_fraction(self) -> float:
        """HP as a fraction, clamped to 1.0 against injected or corrupt values."""
        return _fraction(self.hp, self.max_hp)


def read_pokemon(mem: Memory, address: int) -> Pokemon:
    return Pokemon(
        species=mem[address + OFF_SPECIES],
        hp=read_u16(mem, address + OFF_HP),
        max_hp=read_u16(mem, address + OFF_MAX_HP),
        level=mem[address + OFF_LEVEL],
        status=Status.from_byte(mem[address + OFF_STATUS]),
        types=(mem[address + OFF_TYPES], mem[address + OFF_TYPES + 1]),
        moves=_read_moves(mem, address + OFF_MOVES),
        pp=_read_pp(mem, address + OFF_PP),
        ot_id=read_u16(mem, address + OFF_OT_ID),
        exp=read_u24(mem, address + OFF_EXP),
        stats=_read_stats(mem, address + OFF_STATS),
    )


def _read_roster(mem: Memory, count_addr: int, mons_addr: int) -> tuple[Pokemon, ...]:
    count = mem[count_addr]
    if count > MAX_PARTY:
        raise ValueError(f"party count {count} at {count_addr:#06x} exceeds {MAX_PARTY}")
    return tuple(read_pokemon(mem, mons_addr + MON_SIZE * i) for i in range(count))


def read_party(mem: Memory) -> tuple[Pokemon, ...]:
    """Read the party. Battle HP is newer than this while fighting; see ``read_battle``."""
    return _read_roster(mem, PARTY_COUNT, PARTY_MONS)


def read_enemy_party(mem: Memory) -> tuple[Pokemon, ...]:
    """Read a trainer's party. Only meaningful during a trainer battle."""
    return _read_roster(mem, ENEMY_PARTY_COUNT, ENEMY_PARTY_MONS)


def _read_items(mem: Memory, count_addr: int, capacity: int) -> tuple[tuple[int, int], ...]:
    count = mem[count_addr]
    if count > capacity:
        raise ValueError(f"item count {count} at {count_addr:#06x} exceeds {capacity}")
    first = count_addr + 1
    return tuple((mem[first + 2 * i], mem[first + 2 * i + 1]) for i in range(count))


def read_bag(mem: Memory) -> tuple[tuple[int, int], ...]:
    """Return ``(item_id, quantity)`` pairs."""
    return _read_items(mem, BAG_COUNT, BAG_CAPACITY)


def read_pc_items(mem: Memory) -> tuple[tuple[int, int], ...]:
    """Return ``(item_id, quantity)`` pairs."""
    return _read_items(mem, PC_ITEM_COUNT, PC_ITEM_CAPACITY)


@dataclass(frozen=True, slots=True)
class Combatant:
    """The active Pokemon on one side of a battle.

    ``stats`` include badge boosts and stage changes, so they differ from the
    party struct.
    """

    species: int
    hp: int
    max_hp: int
    level: int
    status: Status
    types: tuple[int, int]
    moves: tuple[int, int, int, int]
    stats: Stats
    stages: tuple[int, int, int, int, int, int]  # 7 is neutral

    @property
    def hp_fraction(self) -> float:
        return _fraction(self.hp, self.max_hp)


@dataclass(frozen=True, slots=True)
class Battle:
    battle_type: int
    player: Combatant
    player_pp: tuple[int, int, int, int]
    enemy: Combatant
    turn: int
    player_selected_move: int  # selected, which is not always executed
    enemy_selected_move: int


def _read_stages(mem: Memory, address: int) -> tuple[int, int, int, int, int, int]:
    return tuple(mem[address + i] for i in range(6))  # type: ignore[return-value]


def read_battle(mem: Memory) -> Battle | None:  # nsp: ignore NSP001 None is the legal outside-battle answer
    """Return battle state, or None outside battle, when these fields go stale.

    During the intro animation and mid-turn the fields can still be loading or
    briefly wrong, so sample when ``at_fight_menu`` is true.
    """
    battle_type = mem[BATTLE_TYPE]
    if battle_type not in (BATTLE_WILD, BATTLE_TRAINER):
        return None
    return Battle(
        battle_type=battle_type,
        player=Combatant(
            species=mem[0xD014],
            hp=read_u16(mem, 0xD015),
            max_hp=read_u16(mem, 0xD023),
            level=mem[0xD022],
            status=Status.from_byte(mem[0xD018]),
            types=(mem[0xD019], mem[0xD01A]),
            moves=_read_moves(mem, 0xD01C),
            stats=_read_stats(mem, 0xD025),
            stages=_read_stages(mem, PLAYER_STAGES),
        ),
        player_pp=_read_pp(mem, 0xD02D),
        enemy=Combatant(
            species=mem[0xCFE5],
            hp=read_u16(mem, 0xCFE6),
            max_hp=read_u16(mem, 0xCFF4),
            level=mem[0xCFF3],
            status=Status.from_byte(mem[0xCFE9]),
            types=(mem[0xCFEA], mem[0xCFEB]),
            moves=_read_moves(mem, 0xCFED),
            stats=_read_stats(mem, 0xCFF6),
            stages=_read_stages(mem, ENEMY_STAGES),
        ),
        turn=mem[TURN_COUNT],
        player_selected_move=mem[PLAYER_SELECTED_MOVE],
        enemy_selected_move=mem[ENEMY_SELECTED_MOVE],
    )


@dataclass(frozen=True, slots=True)
class GameState:
    map_id: int
    x: int
    y: int
    facing: int  # 0 down, 4 up, 8 left, 12 right
    money: int
    badges: int
    party: tuple[Pokemon, ...]
    bag: tuple[tuple[int, int], ...]
    pokedex_owned: int
    pokedex_seen: int
    text_box_open: bool
    blacked_out: bool
    battle: Battle | None


def read_state(mem: Memory) -> GameState:
    return GameState(
        map_id=mem[MAP_ID],
        x=mem[X_POS],
        y=mem[Y_POS],
        facing=mem[FACING],
        money=read_money(mem),
        badges=mem[BADGES],
        party=read_party(mem),
        bag=read_bag(mem),
        pokedex_owned=pokedex_owned_count(mem),
        pokedex_seen=pokedex_seen_count(mem),
        text_box_open=text_box_open(mem),
        blacked_out=blacked_out(mem),
        battle=read_battle(mem),
    )


@dataclass(frozen=True, slots=True)
class BaseStats:
    dex: int
    hp: int
    attack: int
    defense: int
    speed: int
    special: int
    types: tuple[int, int]
    catch_rate: int
    exp_yield: int


@dataclass(frozen=True, slots=True)
class Move:
    id: int
    effect: int
    power: int
    type: int
    accuracy: int  # out of 255
    pp: int


class RomTables:
    """Static game data, parsed from the ROM after checking its hash."""

    def __init__(self, rom_path: str | Path) -> None:
        rom = Path(rom_path).read_bytes()
        digest = hashlib.sha1(rom).hexdigest()
        if digest != ROM_SHA1:
            raise ValueError(f"unsupported ROM {digest}; expected {ROM_SHA1}")
        self._rom = rom
        self.moves = self._parse_moves()
        self.base_stats = self._parse_base_stats()
        self.type_chart = self._parse_type_chart()

    def _parse_moves(self) -> dict[int, Move]:
        moves = {}
        for index in range(NUM_MOVES):
            record = self._rom[ROM_MOVES + MOVE_SIZE * index :][:MOVE_SIZE]
            require(record[0] == index + 1, f"move table misaligned at index {index}: id {record[0]}")
            moves[record[0]] = Move(*record)
        return moves

    def _parse_base_stats(self) -> dict[int, BaseStats]:
        offsets = [ROM_BASE_STATS + BASE_STATS_SIZE * i for i in range(150)] + [ROM_MEW_STATS]
        stats = {}
        for dex, offset in enumerate(offsets, start=1):
            r = self._rom[offset : offset + BASE_STATS_SIZE]
            require(r[0] == dex, f"base stats misaligned for dex {dex}: read {r[0]}")
            stats[dex] = BaseStats(dex, r[1], r[2], r[3], r[4], r[5], (r[6], r[7]), r[8], r[9])
        return stats

    def _parse_type_chart(self) -> dict[tuple[int, int], float]:
        chart: dict[tuple[int, int], float] = {}
        offset = ROM_TYPE_CHART
        for _ in range(256):
            if self._rom[offset] == 0xFF:
                return chart
            attacker, defender, multiplier = self._rom[offset : offset + 3]
            require(multiplier in (0, 5, 20), f"unexpected type multiplier {multiplier}")
            chart[(attacker, defender)] = multiplier / 10
            offset += 3
        unreachable("type chart has no terminator")

    def dex_number(self, internal_id: int) -> int:
        """Convert an internal species ID. Returns 0 for unused 'MissingNo.' slots."""
        if not 1 <= internal_id <= NUM_INTERNAL_IDS:
            raise ValueError(f"internal species ID {internal_id} out of range")
        return self._rom[ROM_INTERNAL_TO_DEX + internal_id - 1]

    def type_multiplier(self, move_type: int, defender_types: tuple[int, int]) -> float:
        """Return move effectiveness, counting a repeated defender type once."""
        require(move_type >= 0, f"move type cannot be negative, got {move_type}")
        first, second = defender_types
        result = self.type_chart.get((move_type, first), 1.0)
        if second != first:
            result *= self.type_chart.get((move_type, second), 1.0)
        return result
