"""Milestones: the story progress every comparison is measured by."""

from __future__ import annotations

import math

import pytest

from pokered.frame import EVENT_FLAG_BYTES, GameFrame
from pokered.milestones import MILESTONES, Milestone, MilestoneTracker
from pokered.negative_space import CheckFailed


def _frame(*, map_id: int = 40, flag: tuple[int, int] | None = None) -> GameFrame:
    flags = bytearray(EVENT_FLAG_BYTES)
    if flag is not None:
        flags[flag[0]] |= 1 << flag[1]
    return GameFrame(map_id=map_id, x=0, y=0, in_battle=False, event_flags=bytes(flags), badges=0, party_levels=(6,), party_hp=(1.0,))


def _tracker() -> MilestoneTracker:
    tracker = MilestoneTracker()
    tracker.reset(_frame())
    return tracker


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        # Map ids from pret/pokered constants/map_constants.asm.
        ("viridian_city", {"map_id": 1}),
        ("viridian_forest", {"map_id": 51}),
        ("pewter_city", {"map_id": 2}),
        ("mt_moon", {"map_id": 59}),
        ("cerulean_city", {"map_id": 3}),
        ("vermilion_city", {"map_id": 5}),
        # Event numbers from constants/event_constants.asm, as (byte, bit) = (n // 8, n % 8).
        ("beat_brock", {"flag": (14, 7)}),
        ("beat_misty", {"flag": (23, 7)}),
        ("got_ss_ticket", {"flag": (171, 4)}),
        ("beat_cerulean_rocket", {"flag": (20, 7)}),
        ("beat_lt_surge", {"flag": (44, 7)}),
        ("got_hm01", {"flag": (188, 0)}),
        ("beat_champion", {"flag": (288, 1)}),
    ],
)
def test_milestone_is_reached_by_its_game_condition(name, kwargs):
    tracker = _tracker()
    tracker.update(_frame(**kwargs), step=5)
    assert tracker.summary()[f"milestone_{name}"] == pytest.approx(1.0)


def test_milestone_starts_unreached():
    assert _tracker().summary()["milestone_beat_brock"] == pytest.approx(0.0)


def test_unreached_milestone_has_no_step_count():
    assert math.isnan(_tracker().summary()["steps_to_beat_brock"])


def test_milestone_records_first_step_reached():
    tracker = _tracker()
    tracker.update(_frame(map_id=1), step=5)
    tracker.update(_frame(map_id=1), step=9)
    assert tracker.summary()["steps_to_viridian_city"] == pytest.approx(5.0)


def test_milestone_stays_reached_after_leaving():
    tracker = _tracker()
    tracker.update(_frame(map_id=1), step=5)
    tracker.update(_frame(map_id=40), step=6)
    assert tracker.summary()["milestone_viridian_city"] == pytest.approx(1.0)


def test_reset_clears_reached_milestones():
    tracker = _tracker()
    tracker.update(_frame(map_id=1), step=5)
    tracker.reset(_frame())
    assert tracker.summary()["milestone_viridian_city"] == pytest.approx(0.0)


def test_milestone_met_in_the_first_frame_is_not_credited_as_progress():
    tracker = MilestoneTracker()
    tracker.reset(_frame(map_id=1))
    assert tracker.summary()["steps_to_viridian_city"] == pytest.approx(0.0)


def test_milestone_names_are_unique():
    names = [milestone.name for milestone in MILESTONES]
    assert len(names) == len(set(names))


def test_milestone_refuses_two_conditions():
    with pytest.raises(CheckFailed, match="exactly one"):
        Milestone("both", map_id=1, flag=(0, 0))


def test_milestone_refuses_no_condition():
    with pytest.raises(CheckFailed, match="exactly one"):
        Milestone("neither")


@pytest.mark.parametrize("flag", [(EVENT_FLAG_BYTES, 0), (0, 8), (-1, 0)])
def test_milestone_refuses_flag_outside_the_array(flag):
    with pytest.raises(CheckFailed, match="flag"):
        Milestone("outside", flag=flag)


def test_item_milestone_needs_the_item_in_the_bag():
    milestone = Milestone("scope", item=0x48)
    bare = GameFrame(map_id=0, x=0, y=0, in_battle=False, event_flags=bytes(EVENT_FLAG_BYTES), badges=0, party_levels=(), party_hp=(), bag_items=(0x48,))
    assert milestone.reached(bare) is True


def test_move_milestone_needs_the_move_in_the_party():
    milestone = Milestone("cut", move=0x0F)
    bare = GameFrame(map_id=0, x=0, y=0, in_battle=False, event_flags=bytes(EVENT_FLAG_BYTES), badges=0, party_levels=(), party_hp=(), party_moves=(1, 2, 3, 4))
    assert milestone.reached(bare) is False


@pytest.mark.parametrize("name", ["beat_cerulean_rival", "got_ss_ticket", "trashed_house", "knows_cut", "has_silph_scope"])
def test_story_chain_tracks_the_steps_past_misty(name):
    assert name in {milestone.name for milestone in MILESTONES}
