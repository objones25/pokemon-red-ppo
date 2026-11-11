"""Parsing a GameFrame from memory."""

from __future__ import annotations

from dataclasses import replace

import pytest
from support import REPO, emulator, requires_rom

from pokered.frame import EVENT_FLAG_BYTES, read_frame
from pokered.ram_map import POKEDEX_BYTES, SS_ANNE_RIVAL_BEATEN
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


def test_frame_reads_the_respawn_map_and_the_map_last_left():
    frame = read_frame(_Ram({0xD719: 2, 0xD365: 3}))
    assert (frame.respawn_map, frame.last_map) == (2, 3)


def test_frame_reads_both_types_of_each_party_member():
    frame = read_frame(
        _Ram({0xD163: 2, 0xD16B + 5: 0x15, 0xD16B + 6: 0x15, 0xD16B + 44 + 5: 0x16, 0xD16B + 44 + 6: 0x03})
    )
    assert frame.party_types == (0x15, 0x15, 0x16, 0x03)


def test_frame_rejects_types_that_do_not_match_the_party():
    with pytest.raises(AssertionError, match="2 types per Pokemon"):
        replace(read_frame(_Ram({0xD163: 1})), party_types=(0x15,))


def test_frame_carries_the_hidden_events_found_since_the_last_step():
    assert read_frame(_Ram(), hidden_events=((40, 3),)).hidden_events == ((40, 3),)


def test_frame_rejects_a_hidden_event_that_does_not_fit_a_byte():
    with pytest.raises(AssertionError, match="hidden event"):
        replace(read_frame(_Ram()), hidden_events=((40, 256),))


@requires_rom
def test_start_state_respawns_in_pallet_town():
    assert _start_frame().respawn_map == 0


def test_frame_reads_the_ss_anne_2f_script_byte():
    assert read_frame(_Ram({0xD665: 4})).ss_anne_2f_script == 4


def test_frame_reads_the_species_of_party_members_only():
    ram = _Ram({0xD163: 2, 0xD16B: 0xB3, 0xD16B + 44: 0x85, 0xD16B + 88: 0x09})
    assert read_frame(ram).party_species == (0xB3, 0x85)


def test_frame_counts_every_kind_of_ball_in_the_bag():
    # bag: count, then (item, quantity) pairs: 3 Poke Balls, a Potion, 2 Great Balls
    ram = _Ram({0xD31D: 3, 0xD31E: 0x04, 0xD31F: 3, 0xD320: 0x14, 0xD321: 1, 0xD322: 0x03, 0xD323: 2})
    assert read_frame(ram).poke_balls == 5


def test_frame_carries_the_cuts_made_since_the_last_step():
    cut = (5, 15, 17, 0, 0x3D)
    assert read_frame(_Ram(), cuts=(cut,)).cuts == (cut,)


def test_frame_rejects_a_cut_that_does_not_fit_bytes():
    with pytest.raises(AssertionError, match="cut"):
        replace(read_frame(_Ram()), cuts=((5, 15, 256, 0, 0x3D),))


@requires_rom
def test_start_state_has_not_met_the_ss_anne_rival():
    assert _start_frame().ss_anne_2f_script == 0


@requires_rom
def test_a_state_after_the_ss_anne_rival_win_reads_past_the_beaten_threshold():
    with emulator() as pyboy:
        with (REPO / "tests" / "states" / "ss_anne_rival_beaten.state").open("rb") as handle:
            pyboy.load_state(handle)
        assert read_frame(pyboy.memory).ss_anne_2f_script >= SS_ANNE_RIVAL_BEATEN
