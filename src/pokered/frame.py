"""One parsed snapshot of game memory: the only game data rewards are allowed to see."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pokered import ram_map
from pokered.negative_space import require

EVENT_FLAG_BYTES: Final[int] = ram_map.EVENT_FLAGS_BYTES
NO_SPECIES_OWNED: Final[bytes] = bytes(ram_map.POKEDEX_BYTES)


@dataclass(frozen=True, slots=True)
class GameFrame:
    map_id: int
    x: int
    y: int
    in_battle: bool
    event_flags: bytes
    badges: int
    party_levels: tuple[int, ...]
    party_hp: tuple[float, ...]
    battle_type: int = ram_map.BATTLE_NONE
    opponent: int = 0
    trainer_number: int = 0
    party_moves: tuple[int, ...] = ()
    bag_items: tuple[int, ...] = ()
    pokedex_owned: bytes = NO_SPECIES_OWNED  # one bit per dex number, lowest first
    party_types: tuple[int, ...] = ()  # two per Pokemon, in party order; the same type twice for a single type
    respawn_map: int = 0  # where a blackout returns the player (Pallet Town until a nurse heals)
    last_map: int = 0  # the outdoor map the player last left
    hidden_events: tuple[tuple[int, int], ...] = ()  # (map, index) of each hidden event the game matched since the last frame

    def __post_init__(self) -> None:
        require(len(self.event_flags) == EVENT_FLAG_BYTES, f"event flags must be {EVENT_FLAG_BYTES} bytes")
        require(len(self.pokedex_owned) == ram_map.POKEDEX_BYTES, f"pokedex owned must be {ram_map.POKEDEX_BYTES} bytes")
        require(len(self.party_levels) == len(self.party_hp), "party levels and hp must have one entry per Pokemon")
        require(all(0.0 <= hp <= 1.0 for hp in self.party_hp), f"hp fractions out of range: {self.party_hp}")
        require(0 <= self.badges <= 0xFF, f"badges must fit one byte, got {self.badges}")
        types = ram_map.TYPES_PER_MON * len(self.party_levels)
        require(len(self.party_types) in (0, types), f"expected {ram_map.TYPES_PER_MON} types per Pokemon, got {self.party_types}")
        require(
            all(0 <= map_id <= 0xFF and 0 <= index <= 0xFF for map_id, index in self.hidden_events),
            f"a hidden event is a (map, index) pair of bytes, got {self.hidden_events}",
        )


def _hp_fraction(mem: ram_map.Memory, address: int) -> float:  # nsp: ignore NSP001 clamped by construction; GameFrame checks the range
    maximum = ram_map.read_u16(mem, address + ram_map.OFF_MAX_HP)
    if maximum == 0:
        return 0.0
    return min(ram_map.read_u16(mem, address + ram_map.OFF_HP) / maximum, 1.0)


def read_frame(mem: ram_map.Memory, *, hidden_events: tuple[tuple[int, int], ...] = ()) -> GameFrame:
    count = mem[ram_map.PARTY_COUNT]
    if count > ram_map.MAX_PARTY:
        raise ValueError(f"party count {count} exceeds {ram_map.MAX_PARTY}")
    party = [ram_map.PARTY_MONS + ram_map.MON_SIZE * i for i in range(count)]
    return GameFrame(
        map_id=mem[ram_map.MAP_ID],
        x=mem[ram_map.X_POS],
        y=mem[ram_map.Y_POS],
        in_battle=mem[ram_map.BATTLE_TYPE] != ram_map.BATTLE_NONE,
        event_flags=bytes(mem[ram_map.EVENT_FLAGS_START + i] for i in range(EVENT_FLAG_BYTES)),
        badges=mem[ram_map.BADGES],
        party_levels=tuple(mem[address + ram_map.OFF_LEVEL] for address in party),
        party_hp=tuple(_hp_fraction(mem, address) for address in party),
        battle_type=mem[ram_map.BATTLE_TYPE],
        opponent=mem[ram_map.OPPONENT],
        trainer_number=mem[ram_map.TRAINER_NUMBER],
        party_moves=tuple(mem[address + ram_map.OFF_MOVES + i] for address in party for i in range(ram_map.MOVES_PER_MON)),
        bag_items=tuple(item for item, _ in ram_map.read_bag(mem)),
        pokedex_owned=bytes(mem[ram_map.POKEDEX_OWNED + i] for i in range(ram_map.POKEDEX_BYTES)),
        party_types=tuple(mem[address + ram_map.OFF_TYPES + i] for address in party for i in range(ram_map.TYPES_PER_MON)),
        respawn_map=mem[ram_map.RESPAWN_MAP],
        last_map=mem[ram_map.LAST_MAP],
        hidden_events=hidden_events,
    )
