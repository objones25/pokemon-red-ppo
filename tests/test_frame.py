"""Parsing a GameFrame from memory."""

from __future__ import annotations

from dataclasses import replace

import pytest
from support import emulator, requires_rom

from pokered.frame import EVENT_FLAG_BYTES, read_frame
from pokered.ram_map import POKEDEX_BYTES
from pokered.start_state import START_STATE


def _start_frame():
    with emulator() as pyboy:
        with START_STATE.open("rb") as handle:
            pyboy.load_state(handle)
        return read_frame(pyboy.memory)


class _Ram(dict):
    def __missing__(self, _address: int) -> int:
        return 0


def test_read_frame_reads_position():
    ram = _Ram({0xD35E: 12, 0xD362: 7, 0xD361: 9})
    frame = read_frame(ram)
    assert (frame.map_id, frame.x, frame.y) == (12, 7, 9)


@pytest.mark.parametrize(("battle_byte", "expected"), [(0, False), (1, True), (2, True), (0xFF, True)])
def test_read_frame_treats_any_nonzero_battle_byte_as_battle(battle_byte, expected):
    assert read_frame(_Ram({0xD057: battle_byte})).in_battle is expected


def test_read_frame_reads_levels_for_party_members_only():
    ram = _Ram({0xD163: 2, 0xD18C: 6, 0xD18C + 44: 3, 0xD18C + 88: 99})
    assert read_frame(ram).party_levels == (6, 3)


def test_read_frame_reads_all_event_flag_bytes():
    frame = read_frame(_Ram({0xD747: 0x01, 0xD747 + EVENT_FLAG_BYTES - 1: 0x80, 0xD747 + EVENT_FLAG_BYTES: 0xFF}))
    assert frame.event_flags == b"\x01" + bytes(EVENT_FLAG_BYTES - 2) + b"\x80"


def test_read_frame_accepts_full_party():
    assert len(read_frame(_Ram({0xD163: 6})).party_levels) == 6


def test_read_frame_rejects_impossible_party_count():
    with pytest.raises(ValueError, match="party count 7 exceeds 6"):
        read_frame(_Ram({0xD163: 7}))


def test_read_frame_reads_hp_fraction_for_party_members():
    ram = _Ram({0xD163: 1, 0xD16C: 0, 0xD16D: 11, 0xD18D: 0, 0xD18E: 22})
    assert read_frame(ram).party_hp == pytest.approx((0.5,))


def test_read_frame_reads_badges():
    assert read_frame(_Ram({0xD356: 0b101})).badges == 0b101


@requires_rom
def test_read_frame_on_start_state_has_level_six_starter():
    assert _start_frame().party_levels == (6,)


@requires_rom
def test_read_frame_on_start_state_is_in_oaks_lab():
    assert _start_frame().map_id == 40


def test_frame_reads_battle_type_opponent_and_trainer_number():
    frame = read_frame(_Ram({0xD057: 2, 0xD059: 234, 0xD05D: 1}))
    assert (frame.battle_type, frame.opponent, frame.trainer_number) == (2, 234, 1)


def test_frame_reads_the_moves_of_each_party_member():
    frame = read_frame(_Ram({0xD163: 1, 0xD16B + 8: 15, 0xD16B + 9: 33}))
    assert frame.party_moves == (15, 33, 0, 0)


def test_frame_reads_bag_item_ids():
    frame = read_frame(_Ram({0xD31D: 2, 0xD31E: 0x48, 0xD31F: 1, 0xD320: 0x04, 0xD321: 5}))
    assert frame.bag_items == (0x48, 0x04)


def test_frame_reads_every_pokedex_owned_byte_and_not_the_seen_ones():
    frame = read_frame(_Ram({0xD2F7: 0x01, 0xD2F7 + POKEDEX_BYTES - 1: 0x40, 0xD30A: 0xFF}))
    assert frame.pokedex_owned == b"\x01" + bytes(POKEDEX_BYTES - 2) + b"\x40"


def test_frame_rejects_a_short_pokedex():
    with pytest.raises(AssertionError, match="pokedex owned must be 19 bytes"):
        replace(read_frame(_Ram()), pokedex_owned=bytes(18))


@requires_rom
def test_start_state_owns_only_squirtle():
    owned = int.from_bytes(_start_frame().pokedex_owned, "little")
    assert owned == 1 << (7 - 1)
