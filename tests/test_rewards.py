"""Rewards pay for first-time progress only, so no repeatable behavior can earn more than a fixed total."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pokered.frame import EVENT_FLAG_BYTES, GameFrame
from pokered.ram_map import POKEDEX_BYTES
from pokered.rewards import (
    GYM_ACE_LEVEL_BY_BADGES,
    MUSEUM_TICKET,
    BadgeGain,
    EpisodeReward,
    EventFlagGain,
    LevelGain,
    NewSpecies,
    NewTile,
)

NO_FLAGS = bytes(EVENT_FLAG_BYTES)
NO_SPECIES = bytes(POKEDEX_BYTES)


def frame(*, tile=(0, 0, 0), in_battle=False, flags=NO_FLAGS, badges=0, levels=(6,), owned=NO_SPECIES) -> GameFrame:
    map_id, x, y = tile
    return GameFrame(
        map_id=map_id, x=x, y=y, in_battle=in_battle, event_flags=flags, badges=badges, party_levels=levels,
        party_hp=tuple(1.0 for _ in levels), pokedex_owned=owned,
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


# --- LevelGain: each Pokemon pays up to the next gym's ace, and storing one costs nothing ---


def test_level_cap_table_matches_each_gym_leaders_highest_level():
    # pret/pokered parties.asm aces: Brock, Misty, Surge, Erika, Koga, Sabrina, Blaine, Giovanni, Lance.
    assert GYM_ACE_LEVEL_BY_BADGES == (14, 21, 24, 29, 43, 43, 47, 50, 62)


def test_level_pays_weight_per_level_below_the_ace():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    assert term(frame(levels=(9,))) == pytest.approx(1.5)


def test_overleveled_starter_stops_paying_at_the_ace():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    assert term(frame(levels=(20,))) == pytest.approx(0.5 * (14 - 6))


def test_level_past_the_ace_pays_nothing():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    term(frame(levels=(14,)))
    assert term(frame(levels=(50,))) == pytest.approx(0.0)


def test_second_pokemon_earns_up_to_the_ace_even_with_the_starter_past_it():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(30,)))
    term(frame(levels=(30, 8)))
    assert term(frame(levels=(30, 14))) == pytest.approx(0.5 * (14 - 8))


def test_second_pokemon_stops_paying_at_the_ace():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(30, 8)))
    assert term(frame(levels=(30, 40))) == pytest.approx(0.5 * (14 - 8))


def test_slot_order_does_not_change_the_pay():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(10, 8)))
    assert term(frame(levels=(8, 10))) == pytest.approx(0.0)


def test_badge_raises_the_ace_for_levels_already_earned():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    term(frame(levels=(18,)))
    assert term(frame(levels=(18,), badges=0b1)) == pytest.approx(0.5 * (18 - 14))


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


def test_a_new_pokemon_still_stops_paying_at_the_ace():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    term(frame(levels=(6, 4)))
    assert term(frame(levels=(6, 30))) == pytest.approx(0.5 * (14 - 4))


def test_the_starter_still_pays_for_levels_when_another_pokemon_joins_later():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    term(frame(levels=(6, 4)))
    assert term(frame(levels=(9, 4))) == pytest.approx(0.5 * (9 - 6))


@given(st.lists(st.lists(st.integers(1, 100), min_size=1, max_size=6).map(tuple), min_size=1, max_size=40))
def test_lifetime_pay_never_exceeds_six_pokemon_at_the_last_ace(trajectory):
    term = LevelGain(weight=1.0)
    term.reset(frame(levels=(1,)))
    total = sum(term(frame(levels=levels, badges=0xFF)) for levels in trajectory)
    assert total <= 6 * GYM_ACE_LEVEL_BY_BADGES[-1]


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


# --- EpisodeReward and the anti-farming properties -----------------------


def _episode() -> EpisodeReward:
    return EpisodeReward([NewTile(weight=0.005), EventFlagGain(weight=2.0), BadgeGain(weight=5.0), LevelGain(weight=0.5), NewSpecies(weight=1.0)])


def test_episode_reward_reports_each_term_by_name():
    episode = _episode()
    episode.reset(frame())
    assert episode.step(frame(tile=(0, 1, 0))) == pytest.approx({"explore": 0.005, "event": 0.0, "badge": 0.0, "level": 0.0, "species": 0.0})


def test_episode_reward_requires_reset_before_step():
    with pytest.raises(RuntimeError, match="before reset"):
        _episode().step(frame())


def _play(episode: EpisodeReward, trajectory: list[GameFrame]) -> float:
    return sum(sum(episode.step(item).values()) for item in trajectory)


def _generated_frame(**fields) -> GameFrame:
    levels = fields.pop("party_levels")
    return GameFrame(party_levels=levels, party_hp=tuple(1.0 for _ in levels), **fields)


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
    assert total == pytest.approx(0.5 * (14 - 6))
