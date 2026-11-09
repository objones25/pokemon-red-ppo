"""Reports turn a window of episodes into the numbers and tables worth reading."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pokered.negative_space import CheckFailed
from pokered.report import PlaceTotals, build_report, save_report
from pokered.telemetry import EpisodeDetail, EpisodeReport, Fight, Snapshot


def snapshot(*, kind="stall", map_id=3, step=50) -> Snapshot:
    return Snapshot(
        kind=kind, step=step, dry_steps=10, map_id=map_id, x=1, y=2, in_battle=False, party_levels=(9,),
        screen=np.zeros((144, 160), np.uint8), state=b"saved",
    )  # fmt: skip


def episode(*, flags=None, fights=(), stall=None, end_map=3, maps=None, actions=(1, 0, 0, 0, 0, 0, 0), phases=None, **summary) -> EpisodeReport:
    detail = EpisodeDetail(
        length=100, flag_steps=flags or {}, map_steps=maps or {3: 100}, fights=tuple(fights), phase_rewards=phases or {},
        action_counts=actions, last_stall=stall, end=snapshot(kind="end", map_id=end_map),
    )  # fmt: skip
    return EpisodeReport(summary={"events": 1.0, **summary}, detail=detail)


def rows_by_first(report, table):
    return {row[0]: row for row in report.tables[table][1]}


def test_mean_of_a_progress_statistic():
    report = build_report([episode(events=2.0), episode(events=4.0)])
    assert report.scalars["progress/events_mean"] == pytest.approx(3.0)


def test_max_is_kept_for_progress_counts():
    report = build_report([episode(events=2.0), episode(events=4.0)])
    assert report.scalars["progress/events_max"] == pytest.approx(4.0)


def test_max_is_not_kept_for_other_statistics():
    report = build_report([episode(fights=()), episode()])
    assert "progress/blackouts_max" not in report.scalars


def test_story_rate_counts_every_episode():
    report = build_report([episode(milestone_beat_brock=1.0, steps_to_beat_brock=10.0), episode(milestone_beat_brock=0.0, steps_to_beat_brock=float("nan"))])
    assert report.scalars["story/beat_brock_rate"] == pytest.approx(0.5)


def test_story_median_step_uses_only_episodes_that_reached_it():
    report = build_report([episode(milestone_a=1.0, steps_to_a=10.0), episode(milestone_a=0.0, steps_to_a=float("nan"))])
    assert report.scalars["story/a_median_step"] == pytest.approx(10.0)


def test_story_never_reached_has_no_median():
    report = build_report([episode(milestone_a=0.0, steps_to_a=float("nan"))])
    assert "story/a_median_step" not in report.scalars


def test_episode_count_is_reported():
    assert build_report([episode(), episode()]).scalars["progress/episodes"] == pytest.approx(2.0)


def test_flag_row_gives_address_bit_rate_and_median_step():
    report = build_report([episode(flags={0x55C: 30}), episode()])
    assert rows_by_first(report, "flags")["0x55C"] == ("0x55C", "0xD7F2", 4, pytest.approx(0.5), pytest.approx(30.0))


def test_flags_are_sorted_by_how_rarely_they_are_reached_last():
    report = build_report([episode(flags={1: 5, 2: 5}), episode(flags={2: 5})])
    assert [row[0] for row in report.tables["flags"][1]] == ["0x002", "0x001"]


def test_fight_row_counts_fights_and_losses_per_episode():
    fights = [Fight("BROCK#1", 54, 100, 12, 1, lost=True), Fight("BROCK#1", 54, 200, 13, 1, lost=False)]
    report = build_report([episode(fights=fights), episode()])
    assert rows_by_first(report, "fights")["BROCK#1"][2:5] == (pytest.approx(1.0), pytest.approx(0.5), pytest.approx(0.5))


def test_fight_row_gives_mean_party_max_level():
    fights = [Fight("BROCK#1", 54, 100, 12, 1, lost=True), Fight("BROCK#1", 54, 200, 14, 1, lost=False)]
    assert rows_by_first(build_report([episode(fights=fights)]), "fights")["BROCK#1"][5] == pytest.approx(13.0)


def test_wild_fights_are_grouped_by_map():
    fights = [Fight("wild", 3, 1, 5, 1, lost=False), Fight("wild", 4, 2, 5, 1, lost=False)]
    assert sorted(rows_by_first(build_report([episode(fights=fights)]), "fights")) == ["wild@3", "wild@4"]


def test_lost_fights_sort_first():
    fights = [Fight("YOUNGSTER#1", 3, 1, 5, 1, lost=False), Fight("MISTY#1", 65, 2, 5, 1, lost=True)]
    assert build_report([episode(fights=fights)]).tables["fights"][1][0][0] == "MISTY#1"


def test_blackout_total_comes_from_the_fights():
    fights = [Fight("MISTY#1", 65, 2, 5, 1, lost=True)]
    assert build_report([episode(fights=fights)]).scalars["progress/lost_fights_mean"] == pytest.approx(1.0)


def test_map_row_gives_share_of_all_steps():
    report = build_report([episode(maps={1: 75, 2: 25})])
    assert rows_by_first(report, "maps")[1][1] == pytest.approx(0.75)


def test_map_row_gives_share_of_episodes_ending_there():
    report = build_report([episode(end_map=3, maps={3: 1}), episode(end_map=4, maps={4: 1})])
    assert rows_by_first(report, "maps")[3][2] == pytest.approx(0.5)


def test_map_row_gives_share_of_stalls_there():
    report = build_report([episode(stall=snapshot(map_id=3), maps={3: 1}), episode(stall=snapshot(map_id=4), maps={4: 1})])
    assert rows_by_first(report, "maps")[4][3] == pytest.approx(0.5)


def test_map_row_gives_rollout_entropy_and_value_error():
    report = build_report([episode(maps={3: 1})], places={3: PlaceTotals(steps=4, entropy=2.0, value_error=1.0)})
    assert rows_by_first(report, "maps")[3][4:6] == (pytest.approx(0.5), pytest.approx(0.25))


def test_button_shares_sum_to_one():
    report = build_report([episode(actions=(1, 1, 2, 0, 0, 0, 0))])
    assert report.scalars["actions/right"] == pytest.approx(0.5)


def test_reward_by_term_is_per_episode():
    report = build_report([episode(phases={("event", 0): 4.0, ("event", 1): 2.0}), episode()])
    assert report.scalars["reward/event"] == pytest.approx(3.0)


def test_reward_by_term_reports_a_net_move_cost():
    report = build_report([episode(phases={("moves", 0): -0.4}), episode()])
    assert report.scalars["reward/moves"] == pytest.approx(-0.2)


def test_reward_phase_table_is_per_episode():
    report = build_report([episode(phases={("event", 1): 4.0}), episode()])
    assert report.tables["reward_phases"][1] == [("event", 10_240, pytest.approx(2.0))]


def test_snapshots_cover_different_stall_maps_first():
    episodes = [episode(stall=snapshot(map_id=3)), episode(stall=snapshot(map_id=3)), episode(stall=snapshot(map_id=4))]
    assert [item.map_id for item in build_report(episodes, max_snapshots=2).snapshots] == [3, 4]


def test_snapshots_are_limited():
    episodes = [episode(stall=snapshot(map_id=i)) for i in range(5)]
    assert len(build_report(episodes, max_snapshots=2).snapshots) == 2


def test_report_needs_episodes():
    with pytest.raises(CheckFailed, match="no episodes"):
        build_report([])


def test_episodes_must_report_the_same_statistics():
    with pytest.raises(CheckFailed, match="same statistics"):
        build_report([episode(a=1.0), episode(b=1.0)])


def test_saved_report_is_strict_json(tmp_path):
    save_report(build_report([episode(milestone_a=0.0, steps_to_a=float("nan"))]), tmp_path, step=5)
    saved = json.loads((tmp_path / "reports" / "report_000000000005.json").read_text())
    assert saved["scalars"]["story/a_rate"] == pytest.approx(0.0)


def test_saved_report_writes_each_snapshot_state_and_picture(tmp_path):
    save_report(build_report([episode(stall=snapshot(map_id=3))]), tmp_path, step=5)
    names = sorted(path.name for path in (tmp_path / "stuck").iterdir())
    assert names == ["000000000005_0_stall_map3.png", "000000000005_0_stall_map3.state"]
