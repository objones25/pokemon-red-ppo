"""Telemetry records where an episode went: flags, fights, maps and stalls. It only observes, so it can never pay."""

from __future__ import annotations

import pytest

from pokered import ram_map
from pokered.frame import EVENT_FLAG_BYTES, GameFrame
from pokered.negative_space import CheckFailed
from pokered.telemetry import EpisodeTelemetry, opponent_label

NO_TERMS = {"event": 0.0, "badge": 0.0}


def frame(*, map_id=1, flags=bytes(EVENT_FLAG_BYTES), battle=0, opponent=0, number=0, levels=(6,)) -> GameFrame:
    return GameFrame(
        map_id=map_id, x=0, y=0, in_battle=battle != 0, event_flags=flags, badges=0, party_levels=levels,
        party_hp=tuple(1.0 for _ in levels), battle_type=battle, opponent=opponent, trainer_number=number,
    )  # fmt: skip


def flags_with(byte: int, bit: int) -> bytes:
    data = bytearray(EVENT_FLAG_BYTES)
    data[byte] = 1 << bit
    return bytes(data)


def started(stall_steps: int = 100, first: GameFrame | None = None) -> EpisodeTelemetry:
    telemetry = EpisodeTelemetry(stall_steps=stall_steps)
    telemetry.reset(first or frame())
    return telemetry


def test_new_flag_records_its_id_and_first_step():
    telemetry = started()
    telemetry.update(frame(), action=0, terms=NO_TERMS)
    telemetry.update(frame(flags=flags_with(1, 2)), action=0, terms=NO_TERMS)
    assert telemetry.flag_steps == {10: 2}


def test_flag_already_set_at_reset_is_not_recorded():
    telemetry = started(first=frame(flags=flags_with(1, 2)))
    telemetry.update(frame(flags=flags_with(1, 2)), action=0, terms=NO_TERMS)
    assert telemetry.flag_steps == {}


def test_flag_cleared_and_set_again_keeps_its_first_step():
    telemetry = started()
    telemetry.update(frame(flags=flags_with(1, 2)), action=0, terms=NO_TERMS)
    telemetry.update(frame(), action=0, terms=NO_TERMS)
    telemetry.update(frame(flags=flags_with(1, 2)), action=0, terms=NO_TERMS)
    assert telemetry.flag_steps == {10: 1}


def test_steps_are_counted_per_map():
    telemetry = started()
    telemetry.update(frame(map_id=1), action=0, terms=NO_TERMS)
    telemetry.update(frame(map_id=1), action=0, terms=NO_TERMS)
    telemetry.update(frame(map_id=2), action=0, terms=NO_TERMS)
    assert telemetry.map_steps == {1: 2, 2: 1}


def test_buttons_are_counted():
    telemetry = started()
    telemetry.update(frame(), action=3, terms=NO_TERMS)
    telemetry.update(frame(), action=3, terms=NO_TERMS)
    assert telemetry.action_counts[3] == 2


def test_trainer_battle_is_one_fight_however_long_it_lasts():
    telemetry = started()
    fight = frame(battle=ram_map.BATTLE_TRAINER, opponent=200 + 0x22, number=1)
    telemetry.update(fight, action=0, terms=NO_TERMS)
    telemetry.update(fight, action=0, terms=NO_TERMS)
    assert [item.opponent for item in telemetry.fights] == ["BROCK#1"]


def test_two_battles_are_two_fights():
    telemetry = started()
    wild = frame(battle=ram_map.BATTLE_WILD, opponent=36)
    telemetry.update(wild, action=0, terms=NO_TERMS)
    telemetry.update(frame(), action=0, terms=NO_TERMS)
    telemetry.update(wild, action=0, terms=NO_TERMS)
    assert len(telemetry.fights) == 2


def test_fight_records_party_strength_at_the_start():
    telemetry = started()
    telemetry.update(frame(battle=ram_map.BATTLE_WILD, levels=(12, 5)), action=0, terms=NO_TERMS)
    fight = telemetry.fights[0]
    assert (fight.party_max_level, fight.party_size) == (12, 2)


def test_fight_won_is_not_a_loss():
    telemetry = started()
    telemetry.update(frame(battle=ram_map.BATTLE_TRAINER, opponent=200 + 0x22, number=1), action=0, terms=NO_TERMS)
    telemetry.update(frame(), action=0, terms=NO_TERMS)
    assert telemetry.fights[0].lost is False


def test_lost_marker_after_the_battle_ended_marks_that_fight_lost():
    # pret home/overworld.asm: EndOfBattle clears D057, then .allPokemonFainted writes LOST_BATTLE to it.
    telemetry = started()
    telemetry.update(frame(battle=ram_map.BATTLE_TRAINER, opponent=200 + 0x22, number=1), action=0, terms=NO_TERMS)
    telemetry.update(frame(), action=0, terms=NO_TERMS)
    telemetry.update(frame(battle=ram_map.BATTLE_LOST), action=0, terms=NO_TERMS)
    assert telemetry.fights[0].lost is True


def test_lost_marker_never_counts_as_a_new_fight():
    telemetry = started()
    telemetry.update(frame(battle=ram_map.BATTLE_WILD), action=0, terms=NO_TERMS)
    telemetry.update(frame(battle=ram_map.BATTLE_LOST), action=0, terms=NO_TERMS)
    telemetry.update(frame(battle=ram_map.BATTLE_LOST), action=0, terms=NO_TERMS)
    assert len(telemetry.fights) == 1


def test_lost_marker_counts_one_blackout_however_many_steps_it_shows():
    telemetry = started()
    telemetry.update(frame(battle=ram_map.BATTLE_WILD), action=0, terms=NO_TERMS)
    telemetry.update(frame(battle=ram_map.BATTLE_LOST), action=0, terms=NO_TERMS)
    telemetry.update(frame(battle=ram_map.BATTLE_LOST), action=0, terms=NO_TERMS)
    assert telemetry.blackouts == 1


def test_lost_marker_with_no_fight_is_still_a_blackout():
    telemetry = started()
    telemetry.update(frame(battle=ram_map.BATTLE_LOST), action=0, terms=NO_TERMS)
    assert (telemetry.blackouts, telemetry.fights) == (1, [])


def test_stall_fires_once_when_no_story_reward_arrives_for_the_limit():
    telemetry = started(stall_steps=3)
    fired = [telemetry.update(frame(), action=0, terms=NO_TERMS) for _ in range(5)]
    assert fired == [False, False, True, False, False]


def test_new_tiles_do_not_end_a_stall():
    telemetry = started(stall_steps=3)
    terms = {"event": 0.0, "badge": 0.0, "explore": 0.005}
    fired = [telemetry.update(frame(), action=0, terms=terms) for _ in range(3)]
    assert fired == [False, False, True]


def test_story_reward_restarts_the_stall_count():
    telemetry = started(stall_steps=3)
    telemetry.update(frame(), action=0, terms=NO_TERMS)
    telemetry.update(frame(), action=0, terms=NO_TERMS)
    telemetry.update(frame(), action=0, terms={"event": 2.0, "badge": 0.0})
    assert telemetry.dry_steps == 0


def test_dry_steps_count_since_the_last_story_reward():
    telemetry = started()
    telemetry.update(frame(), action=0, terms={"event": 2.0, "badge": 0.0})
    telemetry.update(frame(), action=0, terms=NO_TERMS)
    telemetry.update(frame(), action=0, terms=NO_TERMS)
    assert telemetry.dry_steps == 2


def test_reward_is_binned_by_episode_phase():
    telemetry = started()
    telemetry.update(frame(), action=0, terms={"event": 2.0})
    telemetry.update(frame(), action=0, terms={"event": 2.0})
    assert telemetry.phase_rewards == {("event", 0): pytest.approx(4.0)}


def test_reset_forgets_the_last_episode():
    telemetry = started()
    telemetry.update(frame(flags=flags_with(1, 2), battle=ram_map.BATTLE_WILD), action=0, terms={"event": 2.0})
    telemetry.reset(frame())
    assert (telemetry.flag_steps, telemetry.fights, telemetry.map_steps) == ({}, [], {})


def test_stall_limit_must_be_positive():
    with pytest.raises(CheckFailed, match="stall"):
        EpisodeTelemetry(stall_steps=0)


@pytest.mark.parametrize(
    ("battle", "opponent", "number", "expected"),
    [
        (ram_map.BATTLE_WILD, 36, 0, "wild"),
        (ram_map.BATTLE_TRAINER, 200 + 0x22, 1, "BROCK#1"),
        (ram_map.BATTLE_TRAINER, 200 + 0x19, 4, "RIVAL1#4"),
        (ram_map.BATTLE_TRAINER, 0, 0, "unknown_trainer"),
    ],
)
def test_opponent_labels(battle, opponent, number, expected):
    assert opponent_label(battle, opponent, number) == expected


def test_stall_limit_of_one_is_allowed():
    assert started(stall_steps=1).update(frame(), action=0, terms=NO_TERMS) is True


def test_action_past_the_last_button_is_refused():
    with pytest.raises(CheckFailed, match="not a button"):
        started().update(frame(), action=7, terms=NO_TERMS)


def test_last_step_of_a_phase_stays_in_that_phase():
    telemetry = started()
    telemetry.steps = 10_239
    telemetry.update(frame(), action=0, terms={"event": 2.0})
    assert telemetry.phase_rewards == {("event", 0): pytest.approx(2.0)}


def test_first_step_of_the_next_phase_moves_on():
    telemetry = started()
    telemetry.steps = 10_240
    telemetry.update(frame(), action=0, terms={"event": 2.0})
    assert telemetry.phase_rewards == {("event", 1): pytest.approx(2.0)}


def test_a_term_that_paid_nothing_records_nothing():
    telemetry = started()
    telemetry.update(frame(), action=0, terms=NO_TERMS)
    assert telemetry.phase_rewards == {}


def test_second_blackout_after_a_new_fight_counts_again():
    telemetry = started()
    telemetry.update(frame(battle=ram_map.BATTLE_LOST), action=0, terms=NO_TERMS)
    telemetry.update(frame(), action=0, terms=NO_TERMS)
    telemetry.update(frame(battle=ram_map.BATTLE_LOST), action=0, terms=NO_TERMS)
    assert telemetry.blackouts == 2


def test_battle_right_after_a_blackout_is_a_new_fight():
    telemetry = started()
    telemetry.update(frame(battle=ram_map.BATTLE_WILD), action=0, terms=NO_TERMS)
    telemetry.update(frame(battle=ram_map.BATTLE_LOST), action=0, terms=NO_TERMS)
    telemetry.update(frame(battle=ram_map.BATTLE_WILD), action=0, terms=NO_TERMS)
    assert len(telemetry.fights) == 2


def test_fight_with_an_empty_party_records_level_zero():
    telemetry = started()
    telemetry.update(frame(battle=ram_map.BATTLE_WILD, levels=()), action=0, terms=NO_TERMS)
    assert telemetry.fights[0].party_max_level == 0
