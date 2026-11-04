"""Parsing a GameFrame from memory."""

from __future__ import annotations

import pytest
from support import emulator, requires_rom

from pokered.frame import EVENT_FLAG_BYTES, read_frame
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
