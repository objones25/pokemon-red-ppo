"""Rewards pay for first-time progress only, so no repeatable behavior can earn more than a fixed total."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pokered.frame import EVENT_FLAG_BYTES, GameFrame
from pokered.rewards import (
    LEVEL_CAP_BY_BADGES,
    MUSEUM_TICKET,
    BadgeGain,
    EpisodeReward,
    EventFlagGain,
    LevelGain,
    NewTile,
)

NO_FLAGS = bytes(EVENT_FLAG_BYTES)


def frame(*, tile=(0, 0, 0), in_battle=False, flags=NO_FLAGS, badges=0, levels=(6,)) -> GameFrame:
    map_id, x, y = tile
    return GameFrame(map_id=map_id, x=x, y=y, in_battle=in_battle, event_flags=flags, badges=badges, party_levels=levels, party_hp=tuple(1.0 for _ in levels))


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


# --- LevelGain: training pays, grinding past the next gym does not --------


def test_level_cap_table_matches_gym_leader_team_totals():
    # Team totals from pret/pokered parties.asm: Brock, Misty, Surge, Erika, Koga, Sabrina, Blaine, Giovanni, Lance.
    assert LEVEL_CAP_BY_BADGES == (26, 39, 63, 82, 156, 156, 171, 226, 292)


def test_level_pays_weight_per_level_below_cap():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    assert term(frame(levels=(9,))) == pytest.approx(1.5)


def test_level_pays_only_up_to_cap_for_current_badges():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    assert term(frame(levels=(40,))) == pytest.approx(0.5 * (26 - 6))


def test_level_pays_nothing_for_grinding_past_cap():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    term(frame(levels=(26,)))
    assert term(frame(levels=(50,))) == pytest.approx(0.0)


def test_level_badge_raises_cap_for_levels_already_earned():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    term(frame(levels=(30,)))
    assert term(frame(levels=(30,), badges=0b1)) == pytest.approx(0.5 * (30 - 26))


def test_level_pays_nothing_for_regaining_levels_after_depositing_a_pokemon():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(10, 10)))
    term(frame(levels=(10,)))
    assert term(frame(levels=(10, 10))) == pytest.approx(0.0)


def test_level_counts_whole_party():
    term = LevelGain(weight=0.5)
    term.reset(frame(levels=(6,)))
    assert term(frame(levels=(6, 4))) == pytest.approx(2.0)


# --- EpisodeReward and the anti-farming properties -----------------------


def _episode() -> EpisodeReward:
    return EpisodeReward([NewTile(weight=0.005), EventFlagGain(weight=2.0), BadgeGain(weight=5.0), LevelGain(weight=0.5)])


def test_episode_reward_reports_each_term_by_name():
    episode = _episode()
    episode.reset(frame())
    assert episode.step(frame(tile=(0, 1, 0))) == pytest.approx({"explore": 0.005, "event": 0.0, "badge": 0.0, "level": 0.0})


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
    assert total == pytest.approx(0.5 * (26 - 6))
