"""Rewards pay for first-time progress only, so no repeatable behavior can earn more than a fixed total."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from support import ROM, requires_rom

from pokered.frame import EVENT_FLAG_BYTES, GameFrame
from pokered.ram_map import POKEDEX_BYTES, Move, RomTables
from pokered.rewards import (
    GYM_ACE_LEVEL_BY_BADGES,
    LEVEL_CAP_BY_BADGES,
    MUSEUM_TICKET,
    BadgeGain,
    CenterHeal,
    EpisodeReward,
    EventFlagGain,
    LevelGain,
    MoveGain,
    NewSpecies,
    NewTile,
    move_scores,
)

NO_FLAGS = bytes(EVENT_FLAG_BYTES)
NO_SPECIES = bytes(POKEDEX_BYTES)

# pret constants/type_constants.asm and map_constants.asm.
NORMAL, WATER = 0x00, 0x15
PALLET_TOWN, PEWTER_CITY, CERULEAN_CITY, ROUTE_3 = 0x00, 0x02, 0x03, 0x0E
PEWTER_POKECENTER, CERULEAN_POKECENTER = 0x3A, 0x40


def frame(
    *,
    tile=(0, 0, 0),
    in_battle=False,
    flags=NO_FLAGS,
    badges=0,
    levels=(6,),
    owned=NO_SPECIES,
    hp=None,
    respawn=0,
    last_map=0,
    moves=None,
    types=None,
) -> GameFrame:
    """Defaults: every Pokemon at full HP, knowing no moves, Water type; respawn in Pallet Town."""
    map_id, x, y = tile
    return GameFrame(
        map_id=map_id, x=x, y=y, in_battle=in_battle, event_flags=flags, badges=badges, party_levels=levels,
        party_hp=hp if hp is not None else tuple(1.0 for _ in levels), pokedex_owned=owned, respawn_map=respawn,
        last_map=last_map, party_moves=moves if moves is not None else (0,) * 4 * len(levels),
        party_types=types if types is not None else (WATER, WATER) * len(levels),
    )  # fmt: skip


def owned_with(*dex_numbers: int) -> bytes:
    """Pokedex owned bits: dex number n is bit (n - 1) % 8 of byte (n - 1) // 8, as in pret wPokedexOwned."""
    data = bytearray(POKEDEX_BYTES)
    for number in dex_numbers:
        data[(number - 1) // 8] |= 1 << ((number - 1) % 8)
    return bytes(data)


def flags_with(*bits: tuple[int, int]) -> bytes:
    data = bytearray(EVENT_FLAG_BYTES)
    for byte, bit in bits:
        data[byte] |= 1 << bit
    return bytes(data)


# --- NewTile -------------------------------------------------------------


def test_new_tile_pays_weight_on_first_visit():
    term = NewTile(weight=0.5)
    term.reset(frame(tile=(0, 0, 0)))
    assert term(frame(tile=(0, 1, 0))) == pytest.approx(0.5)


def test_new_tile_pays_nothing_on_revisit():
    term = NewTile(weight=0.5)
    term.reset(frame(tile=(0, 0, 0)))
    term(frame(tile=(0, 1, 0)))
    assert term(frame(tile=(0, 1, 0))) == pytest.approx(0.0)


def test_new_tile_pays_nothing_for_start_tile():
    term = NewTile(weight=0.5)
    term.reset(frame(tile=(0, 0, 0)))
    assert term(frame(tile=(0, 0, 0))) == pytest.approx(0.0)


def test_new_tile_distinguishes_maps_at_same_coordinates():
    term = NewTile(weight=0.5)
    term.reset(frame(tile=(0, 1, 1)))
    assert term(frame(tile=(1, 1, 1))) == pytest.approx(0.5)


def test_new_tile_pays_nothing_during_battle():
    term = NewTile(weight=0.5)
    term.reset(frame(tile=(0, 0, 0)))
    assert term(frame(tile=(0, 1, 0), in_battle=True)) == pytest.approx(0.0)


def test_new_tile_visited_during_battle_still_pays_after_battle():
    term = NewTile(weight=0.5)
    term.reset(frame(tile=(0, 0, 0)))
    term(frame(tile=(0, 1, 0), in_battle=True))
    assert term(frame(tile=(0, 1, 0))) == pytest.approx(0.5)


def test_new_tile_forgets_visits_on_reset():
    term = NewTile(weight=0.5)
    term.reset(frame(tile=(0, 0, 0)))
    term(frame(tile=(0, 1, 0)))
    term.reset(frame(tile=(0, 0, 0)))
    assert term(frame(tile=(0, 1, 0))) == pytest.approx(0.5)


def test_new_tile_counts_tiles_gained():
    term = NewTile(weight=0.5)
    term.reset(frame(tile=(0, 0, 0)))
    term(frame(tile=(0, 1, 0)))
    assert term.gained == 1


# --- EventFlagGain -------------------------------------------------------


def test_event_pays_weight_when_flag_first_sets():
    term = EventFlagGain(weight=2.0)
    term.reset(frame())
    assert term(frame(flags=flags_with((3, 1)))) == pytest.approx(2.0)


def test_event_pays_per_flag_set_in_one_step():
    term = EventFlagGain(weight=2.0)
    term.reset(frame())
    assert term(frame(flags=flags_with((3, 1), (9, 7)))) == pytest.approx(4.0)


def test_event_pays_nothing_for_flags_set_at_episode_start():
    start = frame(flags=flags_with((3, 1)))
    term = EventFlagGain(weight=2.0)
    term.reset(start)
    assert term(start) == pytest.approx(0.0)


def test_event_pays_nothing_when_flag_toggles_back_on():
    # Quest flags can clear and re-set (the paper notes Bill's quest resets on leaving the house).
    term = EventFlagGain(weight=2.0)
    term.reset(frame())
    term(frame(flags=flags_with((3, 1))))
    term(frame())
    assert term(frame(flags=flags_with((3, 1)))) == pytest.approx(0.0)


def test_event_pays_for_new_flag_while_another_clears():
    # A flag count would see no change here; each flag must be tracked individually.
    term = EventFlagGain(weight=2.0)
    term.reset(frame(flags=flags_with((3, 1))))
    assert term(frame(flags=flags_with((4, 0)))) == pytest.approx(2.0)


def test_event_pays_nothing_for_museum_ticket():
    term = EventFlagGain(weight=2.0)
    term.reset(frame())
    assert term(frame(flags=flags_with(MUSEUM_TICKET))) == pytest.approx(0.0)


def test_event_counts_flags_gained_for_the_step_budget():
    term = EventFlagGain(weight=2.0)
    term.reset(frame())
    term(frame(flags=flags_with((3, 1), (9, 7))))
    assert term.gained == 2


def test_event_gained_clears_on_reset():
    term = EventFlagGain(weight=2.0)
    term.reset(frame())
    term(frame(flags=flags_with((3, 1))))
    term.reset(frame())
    assert term.gained == 0


# --- BadgeGain -----------------------------------------------------------


def test_badge_pays_weight_for_new_badge():
    term = BadgeGain(weight=5.0)
    term.reset(frame())
    assert term(frame(badges=0b1)) == pytest.approx(5.0)


def test_badge_pays_nothing_for_held_badge():
    term = BadgeGain(weight=5.0)
    term.reset(frame())
    term(frame(badges=0b1))
    assert term(frame(badges=0b1)) == pytest.approx(0.0)


def test_badge_pays_nothing_when_bit_returns_after_clearing():
    term = BadgeGain(weight=5.0)
    term.reset(frame())
    term(frame(badges=0b1))
    term(frame(badges=0))
    assert term(frame(badges=0b1)) == pytest.approx(0.0)


# --- LevelGain: each Pokemon pays up to 5 over the next gym's ace, and storing one costs nothing ---


def test_gym_ace_table_matches_each_gym_leaders_highest_level():
    # pret/pokered parties.asm aces: Brock, Misty, Surge, Erika, Koga, Sabrina, Blaine, Giovanni, Lance.
    assert GYM_ACE_LEVEL_BY_BADGES == (14, 21, 24, 29, 43, 43, 47, 50, 62)


def test_level_cap_is_five_levels_over_the_next_gym_ace():
    assert LEVEL_CAP_BY_BADGES == (19, 26, 29, 34, 48, 48, 52, 55, 67)


def test_level_pays_weight_per_level_below_the_cap():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    assert term(frame(levels=(9,))) == pytest.approx(1.5)


def test_overleveled_starter_stops_paying_at_the_cap():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    assert term(frame(levels=(25,))) == pytest.approx(0.5 * (19 - 6))


def test_level_past_the_cap_pays_nothing():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    term(frame(levels=(19,)))
    assert term(frame(levels=(50,))) == pytest.approx(0.0)


def test_second_pokemon_earns_up_to_the_cap_even_with_the_starter_past_it():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(30,)))
    term(frame(levels=(30, 8)))
    assert term(frame(levels=(30, 19))) == pytest.approx(0.5 * (19 - 8))


def test_second_pokemon_stops_paying_at_the_cap():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(30, 8)))
    assert term(frame(levels=(30, 40))) == pytest.approx(0.5 * (19 - 8))


def test_slot_order_does_not_change_the_pay():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(10, 8)))
    assert term(frame(levels=(8, 10))) == pytest.approx(0.0)


def test_badge_raises_the_cap_for_levels_already_earned():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    term(frame(levels=(24,)))
    assert term(frame(levels=(24,), badges=0b1)) == pytest.approx(0.5 * (24 - 19))


def test_storing_a_pokemon_never_takes_reward_back():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(10, 10)))
    assert term(frame(levels=(10,))) == pytest.approx(0.0)


def test_withdrawing_a_stored_pokemon_pays_nothing():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(10, 10)))
    term(frame(levels=(10,)))
    assert term(frame(levels=(10, 10))) == pytest.approx(0.0)


def test_swapping_in_fresh_catches_cannot_earn_the_same_levels_again():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(14, 14)))
    term(frame(levels=(14,)))
    assert term(frame(levels=(14, 14))) == pytest.approx(0.0)


def test_a_pokemon_pays_nothing_for_the_level_it_joins_at():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    assert term(frame(levels=(6, 4))) == pytest.approx(0.0)


def test_a_bought_level_5_magikarp_pays_no_levels():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(21,), badges=0b1))
    assert term(frame(levels=(21, 5), badges=0b1)) == pytest.approx(0.0)


# The game raises the party count, shows the nickname prompt, and only then writes the level (pret
# engine/pokemon/add_mon.asm _AddPartyMon), so a joining Pokemon reads level 0 for some steps.
def test_a_pokemon_whose_level_is_written_after_it_joins_pays_nothing():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(21,), badges=0b1))
    term(frame(levels=(21, 0), badges=0b1))
    assert term(frame(levels=(21, 5), badges=0b1)) == pytest.approx(0.0)


def test_a_long_nickname_prompt_before_the_level_is_written_still_pays_nothing():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(21,), badges=0b1))
    for _ in range(3):
        term(frame(levels=(21, 0), badges=0b1))
    assert term(frame(levels=(21, 5), badges=0b1)) == pytest.approx(0.0)


def test_a_late_written_pokemon_above_the_starter_pays_nothing_on_arrival():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(10,)))
    term(frame(levels=(10, 0)))
    assert term(frame(levels=(12, 10))) == pytest.approx(0.0)


def test_an_episode_starting_mid_write_pays_nothing_when_the_level_appears():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(21, 0), badges=0b1))
    assert term(frame(levels=(21, 5), badges=0b1)) == pytest.approx(0.0)


def test_a_late_written_pokemon_pays_for_levels_it_gains_afterwards():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(21,), badges=0b1))
    term(frame(levels=(21, 0), badges=0b1))
    term(frame(levels=(21, 5), badges=0b1))
    assert term(frame(levels=(21, 7), badges=0b1)) == pytest.approx(0.5 * (7 - 5))


def test_a_new_pokemon_pays_for_each_level_it_gains_after_joining():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    term(frame(levels=(6, 4)))
    assert term(frame(levels=(6, 7))) == pytest.approx(0.5 * (7 - 4))


def test_a_new_pokemon_still_stops_paying_at_the_cap():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    term(frame(levels=(6, 4)))
    assert term(frame(levels=(6, 30))) == pytest.approx(0.5 * (19 - 4))


def test_the_starter_still_pays_for_levels_when_another_pokemon_joins_later():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    term(frame(levels=(6, 4)))
    assert term(frame(levels=(9, 4))) == pytest.approx(0.5 * (9 - 6))


@given(st.lists(st.lists(st.integers(1, 100), min_size=1, max_size=6).map(tuple), min_size=1, max_size=40))
def test_lifetime_pay_never_exceeds_six_pokemon_at_the_last_cap(trajectory):
    term = LevelGain(weight=1.0)
    term.reset(frame(levels=(1,)))
    total = sum(term(frame(levels=levels, badges=0xFF)) for levels in trajectory)
    assert total <= 6 * 67


# --- NewSpecies: each species owned pays once ----------------------------


def test_species_pays_weight_for_a_newly_owned_species():
    term = NewSpecies(weight=1.0)
    term.reset(frame(owned=owned_with(7)))
    assert term(frame(owned=owned_with(7, 129))) == pytest.approx(1.0)


def test_species_pays_nothing_for_species_owned_at_episode_start():
    term = NewSpecies(weight=1.0)
    term.reset(frame(owned=owned_with(7)))
    assert term(frame(owned=owned_with(7))) == pytest.approx(0.0)


def test_species_pays_once_however_many_of_it_are_caught():
    term = NewSpecies(weight=1.0)
    term.reset(frame(owned=owned_with(7)))
    term(frame(owned=owned_with(7, 129)))
    assert term(frame(owned=owned_with(7, 129))) == pytest.approx(0.0)


def test_species_pays_per_species_owned_in_one_step():
    term = NewSpecies(weight=1.0)
    term.reset(frame(owned=owned_with(7)))
    assert term(frame(owned=owned_with(7, 16, 151))) == pytest.approx(2.0)


def test_species_pays_nothing_when_a_bit_returns_after_clearing():
    term = NewSpecies(weight=1.0)
    term.reset(frame())
    term(frame(owned=owned_with(25)))
    term(frame())
    assert term(frame(owned=owned_with(25))) == pytest.approx(0.0)


def test_species_forgets_what_it_paid_for_on_reset():
    term = NewSpecies(weight=1.0)
    term.reset(frame())
    term(frame(owned=owned_with(25)))
    term.reset(frame())
    assert term(frame(owned=owned_with(25))) == pytest.approx(1.0)


def test_species_lifetime_pay_is_at_most_151():
    term = NewSpecies(weight=1.0)
    term.reset(frame())
    assert term(frame(owned=bytes([0xFF] * POKEDEX_BYTES))) == pytest.approx(151.0)


# --- CenterHeal: the first nurse heal that restores HP at each respawn point pays once ---
# The nurse writes wLastMap into wLastBlackoutMap, then heals the party (pret engine/events/pokecenter.asm). A blackout
# heals on the battle map before warping (engine/events/black_out.asm), and other healers are not Pokemon Centers.


def _at_pewter_center(*, hp=(1.0,), respawn=PEWTER_CITY) -> GameFrame:
    return frame(tile=(PEWTER_POKECENTER, 3, 3), levels=(12,) * len(hp), hp=hp, respawn=respawn, last_map=PEWTER_CITY)


def test_heal_pays_weight_when_a_nurse_restores_hp_at_a_new_center():
    term = CenterHeal(weight=5.0)
    term.reset(_at_pewter_center(hp=(0.4,), respawn=PALLET_TOWN))
    assert term(_at_pewter_center()) == pytest.approx(5.0)


def test_heal_pays_for_a_fainted_pokemon_restored_with_the_rest_at_full():
    term = CenterHeal(weight=5.0)
    term.reset(_at_pewter_center(hp=(1.0, 0.0), respawn=PALLET_TOWN))
    assert term(_at_pewter_center(hp=(1.0, 1.0))) == pytest.approx(5.0)


def test_heal_pays_nothing_when_the_party_was_already_at_full_hp():
    term = CenterHeal(weight=5.0)
    term.reset(_at_pewter_center(respawn=PALLET_TOWN))
    assert term(_at_pewter_center()) == pytest.approx(0.0)


def test_a_heal_at_full_hp_does_not_use_up_the_center():
    term = CenterHeal(weight=5.0)
    term.reset(_at_pewter_center(respawn=PALLET_TOWN))
    term(_at_pewter_center())
    term(_at_pewter_center(hp=(0.5,)))
    assert term(_at_pewter_center()) == pytest.approx(5.0)


def test_a_second_heal_at_the_same_center_pays_nothing():
    term = CenterHeal(weight=5.0)
    term.reset(_at_pewter_center(hp=(0.4,), respawn=PALLET_TOWN))
    term(_at_pewter_center())
    term(_at_pewter_center(hp=(0.5,)))
    assert term(_at_pewter_center()) == pytest.approx(0.0)


def test_a_heal_at_a_different_center_pays_again():
    term = CenterHeal(weight=5.0)
    term.reset(_at_pewter_center(hp=(0.4,), respawn=PALLET_TOWN))
    term(_at_pewter_center())
    term(frame(tile=(CERULEAN_POKECENTER, 3, 3), hp=(0.5,), respawn=PEWTER_CITY, last_map=CERULEAN_CITY))
    assert term(
        frame(tile=(CERULEAN_POKECENTER, 3, 3), respawn=CERULEAN_CITY, last_map=CERULEAN_CITY)
    ) == pytest.approx(5.0)


def test_a_blackout_heal_on_a_route_pays_nothing():
    term = CenterHeal(weight=5.0)
    term.reset(frame(tile=(ROUTE_3, 5, 5), in_battle=True, hp=(0.0,), respawn=PEWTER_CITY, last_map=PEWTER_CITY))
    assert term(frame(tile=(ROUTE_3, 5, 5), respawn=PEWTER_CITY, last_map=PEWTER_CITY)) == pytest.approx(0.0)


def test_hp_restored_in_a_center_without_a_nurse_heal_there_pays_nothing():
    term = CenterHeal(weight=5.0)
    term.reset(_at_pewter_center(hp=(0.4,), respawn=PALLET_TOWN))
    assert term(_at_pewter_center(respawn=PALLET_TOWN)) == pytest.approx(0.0)


def test_heal_pays_nothing_for_the_respawn_point_held_at_episode_start():
    term = CenterHeal(weight=5.0)
    term.reset(_at_pewter_center(hp=(0.4,)))
    assert term(_at_pewter_center()) == pytest.approx(0.0)


def test_heal_forgets_the_centers_it_paid_for_on_reset():
    term = CenterHeal(weight=5.0)
    term.reset(_at_pewter_center(hp=(0.4,), respawn=PALLET_TOWN))
    term(_at_pewter_center())
    term.reset(_at_pewter_center(hp=(0.4,), respawn=PALLET_TOWN))
    assert term(_at_pewter_center()) == pytest.approx(5.0)


def test_heal_rejects_an_episode_without_a_party():
    with pytest.raises(AssertionError, match="needs a party"):
        CenterHeal(weight=5.0).reset(frame(levels=()))


heal_frames = st.builds(
    lambda center, hp, respawn, last_map: frame(tile=(center, 0, 0), hp=(hp,), respawn=respawn, last_map=last_map),
    center=st.sampled_from([PEWTER_POKECENTER, CERULEAN_POKECENTER, ROUTE_3]),
    hp=st.sampled_from([0.0, 0.5, 1.0]),
    respawn=st.sampled_from([PALLET_TOWN, PEWTER_CITY, CERULEAN_CITY]),
    last_map=st.sampled_from([PEWTER_CITY, CERULEAN_CITY]),
)


@given(st.lists(heal_frames, min_size=1, max_size=40))
def test_heal_pays_at_most_once_per_respawn_point(trajectory):
    term = CenterHeal(weight=1.0)
    term.reset(trajectory[0])
    total = sum(term(item) for item in trajectory)
    assert total <= len({item.respawn_map for item in trajectory} - {trajectory[0].respawn_map})


# --- MoveGain: each party place pays for raising its best total move strength ---
# Squirtle's moves, as in pret data/moves/moves.asm: id, effect, power, type, accuracy out of 255, PP.
TACKLE, TAIL_WHIP, BUBBLE, WATER_GUN, BITE = 33, 39, 145, 55, 44
SQUIRTLE_MOVES = {
    TACKLE: Move(TACKLE, 0x00, 35, NORMAL, 242, 35),
    TAIL_WHIP: Move(TAIL_WHIP, 0x13, 0, NORMAL, 255, 30),
    BUBBLE: Move(BUBBLE, 0x45, 20, WATER, 255, 30),
    WATER_GUN: Move(WATER_GUN, 0x00, 40, WATER, 255, 25),
    BITE: Move(BITE, 0x1F, 60, NORMAL, 255, 25),
}
TACKLE_SCORE = 35 * 242 / 255
SQUIRTLE_AT_15 = (TACKLE, TAIL_WHIP, BUBBLE, WATER_GUN)


def _move(effect: int, power: int, accuracy: int = 255) -> dict[int, Move]:
    return {1: Move(1, effect, power, NORMAL, accuracy, 10)}


def test_move_score_is_power_times_accuracy():
    assert move_scores(SQUIRTLE_MOVES)[TACKLE].score == pytest.approx(TACKLE_SCORE)


def test_move_score_keeps_the_move_type_for_stab():
    assert move_scores(SQUIRTLE_MOVES)[WATER_GUN].type == WATER


@pytest.mark.parametrize(
    ("effect", "power"),
    [(0x07, 170), (0x26, 1), (0x29, 1), (0x28, 1), (0x00, 1), (0x13, 0)],
    ids=["self-faint", "one-hit-ko", "fixed-damage", "super-fang", "counter", "status"],
)
def test_moves_that_do_not_scale_with_power_score_zero(effect, power):
    assert move_scores(_move(effect, power))[1].score == pytest.approx(0.0)


@pytest.mark.parametrize("effect", [0x27, 0x2B, 0x50], ids=["charge", "fly", "hyper-beam"])
def test_two_turn_moves_score_half(effect):
    assert move_scores(_move(effect, 100))[1].score == pytest.approx(50.0)


@requires_rom
def test_the_rom_move_table_scores_bite_at_its_full_power():
    assert move_scores(RomTables(ROM).moves)[BITE].score == pytest.approx(60.0)


@requires_rom
def test_the_rom_move_table_scores_explosion_zero():
    explosion = 153  # pret constants/move_constants.asm
    assert move_scores(RomTables(ROM).moves)[explosion].score == pytest.approx(0.0)


def _squirtle(*moves: int, level: int = 22) -> GameFrame:
    return frame(levels=(level,), moves=moves, types=(WATER, WATER))


def _move_term() -> MoveGain:
    return MoveGain(weight=1.0, scores=move_scores(SQUIRTLE_MOVES))


def test_learning_bite_over_tail_whip_pays_its_score():
    term = _move_term()
    term.reset(_squirtle(*SQUIRTLE_AT_15))
    assert term(_squirtle(TACKLE, BITE, BUBBLE, WATER_GUN)) == pytest.approx(60.0)


def test_learning_bite_over_tackle_pays_the_difference():
    term = _move_term()
    term.reset(_squirtle(*SQUIRTLE_AT_15))
    assert term(_squirtle(BITE, TAIL_WHIP, BUBBLE, WATER_GUN)) == pytest.approx(60.0 - TACKLE_SCORE)


def test_a_move_of_the_pokemons_own_type_scores_one_and_a_half_times():
    term = _move_term()
    term.reset(_squirtle(TACKLE, TAIL_WHIP, 0, 0))
    assert term(_squirtle(TACKLE, TAIL_WHIP, WATER_GUN, 0)) == pytest.approx(1.5 * 40)


def test_declining_a_move_pays_nothing():
    term = _move_term()
    term.reset(_squirtle(*SQUIRTLE_AT_15))
    assert term(_squirtle(*SQUIRTLE_AT_15)) == pytest.approx(0.0)


def test_relearning_a_forgotten_move_pays_nothing():
    term = _move_term()
    term.reset(_squirtle(*SQUIRTLE_AT_15))
    term(_squirtle(TACKLE, BITE, BUBBLE, WATER_GUN))
    term(_squirtle(*SQUIRTLE_AT_15))
    assert term(_squirtle(TACKLE, BITE, BUBBLE, WATER_GUN)) == pytest.approx(0.0)


def test_a_joining_pokemon_pays_nothing_for_the_moves_it_knows():
    term = _move_term()
    term.reset(_squirtle(*SQUIRTLE_AT_15))
    joined = frame(levels=(22, 5), moves=(*SQUIRTLE_AT_15, BITE, 0, 0, 0), types=(WATER, WATER, NORMAL, NORMAL))
    assert term(joined) == pytest.approx(0.0)


def test_a_joining_pokemon_whose_moves_are_written_after_the_prompt_pays_nothing():
    term = _move_term()
    term.reset(_squirtle(*SQUIRTLE_AT_15))
    term(frame(levels=(22, 0), moves=(*SQUIRTLE_AT_15, 0, 0, 0, 0), types=(WATER, WATER, 0, 0)))
    written = frame(levels=(22, 5), moves=(*SQUIRTLE_AT_15, BITE, 0, 0, 0), types=(WATER, WATER, NORMAL, NORMAL))
    assert term(written) == pytest.approx(0.0)


def test_withdrawing_a_stored_pokemon_pays_nothing_for_its_moves():
    term = _move_term()
    both = frame(levels=(22, 9), moves=(*SQUIRTLE_AT_15, BITE, 0, 0, 0), types=(WATER, WATER, NORMAL, NORMAL))
    term.reset(both)
    term(_squirtle(*SQUIRTLE_AT_15))
    assert term(both) == pytest.approx(0.0)


def test_party_order_does_not_change_the_move_pay():
    term = _move_term()
    term.reset(frame(levels=(22, 9), moves=(*SQUIRTLE_AT_15, BITE, 0, 0, 0), types=(WATER, WATER, NORMAL, NORMAL)))
    swapped = frame(levels=(9, 22), moves=(BITE, 0, 0, 0, *SQUIRTLE_AT_15), types=(NORMAL, NORMAL, WATER, WATER))
    assert term(swapped) == pytest.approx(0.0)


def test_a_second_normal_pokemon_learning_bite_pays_its_score_with_stab():
    term = _move_term()
    term.reset(frame(levels=(22, 9), moves=(*SQUIRTLE_AT_15, TACKLE, 0, 0, 0), types=(WATER, WATER, NORMAL, NORMAL)))
    assert term(frame(levels=(22, 9), moves=(*SQUIRTLE_AT_15, TACKLE, BITE, 0, 0), types=(WATER, WATER, NORMAL, NORMAL))) == pytest.approx(1.5 * 60.0)


def test_a_move_gets_stab_only_from_its_own_users_types():
    term = _move_term()
    term.reset(frame(levels=(9, 9), moves=(0, 0, 0, 0, 0, 0, 0, 0), types=(NORMAL, NORMAL, WATER, WATER)))
    assert term(frame(levels=(9, 9), moves=(WATER_GUN, 0, 0, 0, 0, 0, 0, 0), types=(NORMAL, NORMAL, WATER, WATER))) == pytest.approx(40.0)


def test_storing_the_weaker_pokemon_pays_nothing():
    term = _move_term()
    term.reset(frame(levels=(22, 9), moves=(*SQUIRTLE_AT_15, TACKLE, 0, 0, 0), types=(WATER, WATER, NORMAL, NORMAL)))
    assert term(_squirtle(*SQUIRTLE_AT_15)) == pytest.approx(0.0)


def test_an_episode_starting_mid_write_tolerates_unwritten_move_bytes():
    term = _move_term()
    term.reset(frame(levels=(22, 0), moves=(*SQUIRTLE_AT_15, 200, 0, 0, 0), types=(WATER, WATER, 0, 0)))
    assert term(_squirtle(TACKLE, BITE, BUBBLE, WATER_GUN, level=22)) == pytest.approx(0.0)


def test_move_gain_rejects_an_empty_move_table():
    with pytest.raises(AssertionError, match="needs the move table"):
        MoveGain(weight=1.0, scores={})


def test_move_gain_accepts_a_one_move_table():
    term = MoveGain(weight=1.0, scores=move_scores({BITE: SQUIRTLE_MOVES[BITE]}))
    term.reset(_squirtle(0, 0, 0, 0))
    assert term(_squirtle(BITE, 0, 0, 0)) == pytest.approx(60.0)


def test_a_move_missing_from_the_table_is_a_bug():
    term = _move_term()
    term.reset(_squirtle(*SQUIRTLE_AT_15))
    with pytest.raises(AssertionError, match="move 200 is not in the move table"):
        term(_squirtle(TACKLE, TAIL_WHIP, BUBBLE, 200))


def test_move_gain_rejects_a_frame_without_four_moves_per_pokemon():
    term = _move_term()
    with pytest.raises(AssertionError, match="4 moves per Pokemon"):
        term.reset(frame(levels=(22,), moves=(TACKLE,)))


move_lists = st.lists(st.sampled_from([0, *SQUIRTLE_MOVES]), min_size=4, max_size=4)


@given(st.lists(st.lists(move_lists, min_size=1, max_size=6), min_size=1, max_size=30))
def test_lifetime_move_pay_never_exceeds_six_places_of_four_best_moves(trajectory):
    term = _move_term()
    term.reset(_squirtle(0, 0, 0, 0))
    party = [frame(levels=(9,) * len(mons), moves=tuple(m for mon in mons for m in mon)) for mons in trajectory]
    total = sum(term(item) for item in party)
    assert total <= 6 * 4 * 1.5 * 60


# --- EpisodeReward and the anti-farming properties -----------------------


def _episode() -> EpisodeReward:
    return EpisodeReward(
        [
            NewTile(weight=0.005), EventFlagGain(weight=2.0), BadgeGain(weight=5.0), LevelGain(weight=0.5),
            NewSpecies(weight=1.0), CenterHeal(weight=5.0), MoveGain(weight=0.01, scores=move_scores(SQUIRTLE_MOVES)),
        ]
    )  # fmt: skip


def test_episode_reward_reports_each_term_by_name():
    episode = _episode()
    episode.reset(frame())
    assert episode.step(frame(tile=(0, 1, 0))) == pytest.approx(
        {"explore": 0.005, "event": 0.0, "badge": 0.0, "level": 0.0, "species": 0.0, "heal": 0.0, "moves": 0.0}
    )


def test_episode_reward_requires_reset_before_step():
    with pytest.raises(RuntimeError, match="before reset"):
        _episode().step(frame())


def _play(episode: EpisodeReward, trajectory: list[GameFrame]) -> float:
    return sum(sum(episode.step(item).values()) for item in trajectory)


def _generated_frame(**fields) -> GameFrame:
    """Full HP always: a heal pays the first time HP is restored, which a replay can create (see CenterHeal tests)."""
    levels = fields.pop("party_levels")
    moves = fields.pop("moves")
    return GameFrame(
        party_levels=levels, party_hp=tuple(1.0 for _ in levels), party_moves=tuple(moves[: 4 * len(levels)]),
        party_types=(WATER, WATER) * len(levels), **fields,
    )  # fmt: skip


frames = st.builds(
    _generated_frame,
    map_id=st.integers(0, 3),
    x=st.integers(0, 3),
    y=st.integers(0, 3),
    in_battle=st.booleans(),
    event_flags=st.lists(st.tuples(st.integers(0, EVENT_FLAG_BYTES - 1), st.integers(0, 7)), max_size=4).map(lambda bits: flags_with(*bits)),
    badges=st.integers(0, 255),
    party_levels=st.lists(st.integers(1, 100), min_size=1, max_size=6).map(tuple),
    pokedex_owned=st.lists(st.integers(1, 151), max_size=4).map(lambda numbers: owned_with(*numbers)),
    moves=st.lists(st.sampled_from([0, *SQUIRTLE_MOVES]), min_size=24, max_size=24),
    respawn_map=st.integers(0, 3),
    last_map=st.integers(0, 3),
)


@given(st.lists(frames, min_size=1, max_size=30))
def test_replaying_any_trajectory_pays_nothing_the_second_time(trajectory):
    episode = _episode()
    episode.reset(trajectory[0])
    _play(episode, trajectory)
    assert _play(episode, trajectory) == pytest.approx(0.0)


@given(st.lists(frames, min_size=1, max_size=30))
def test_no_step_ever_pays_a_negative_reward(trajectory):
    episode = _episode()
    episode.reset(trajectory[0])
    rewards = [value for item in trajectory for value in episode.step(item).values()]
    assert min(rewards) >= 0.0


def test_grinding_forever_earns_at_most_the_level_budget():
    episode = _episode()
    episode.reset(frame(levels=(6,)))
    total = sum(sum(episode.step(frame(levels=(level,))).values()) for level in range(6, 101))
    assert total == pytest.approx(0.5 * (19 - 6))
