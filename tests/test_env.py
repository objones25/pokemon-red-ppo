"""The Gymnasium environment on the real game."""

from __future__ import annotations

from dataclasses import replace

import pytest
from support import ROM, requires_rom

from pokered.buttons import BUTTONS
from pokered.config import EnvConfig
from pokered.env import PokemonRedEnv
from pokered.frame import EVENT_FLAG_BYTES, read_frame
from pokered.observations import OBSERVATION_SHAPES

pytestmark = requires_rom


@pytest.fixture
def make_env():
    envs = []

    def make(**overrides) -> PokemonRedEnv:
        env = PokemonRedEnv(EnvConfig(rom=ROM, **overrides))
        envs.append(env)
        return env

    yield make
    for env in envs:
        env.close()


def test_reset_returns_declared_shapes(make_env):
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert {key: value.shape for key, value in obs.items()} == OBSERVATION_SHAPES


def test_observation_is_inside_declared_space(make_env):
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_action_space_has_one_action_per_button(make_env):
    assert make_env().action_space.n == len(BUTTONS)


def test_step_reports_reward_terms_that_sum_to_reward(make_env):
    env = make_env()
    env.reset(seed=0)
    _, reward, _, _, info = env.step(BUTTONS.index("down"))
    assert reward == pytest.approx(sum(info["reward_terms"].values()))


def test_walking_to_a_new_tile_pays_exploration(make_env):
    env = make_env(tile_weight=0.25)
    env.reset(seed=0)
    _, _, _, _, info = env.step(BUTTONS.index("down"))
    assert info["reward_terms"]["explore"] == pytest.approx(0.25)


def test_time_limit_truncates_and_never_terminates(make_env):
    env = make_env(base_budget=3, stagger_first_episode=False)
    env.reset(seed=0)
    results = [env.step(BUTTONS.index("b"))[2:4] for _ in range(3)]
    assert results == [(False, False), (False, False), (False, True)]


def test_first_episode_is_staggered_by_seed(make_env):
    budgets = {make_env(base_budget=10_000).reset(seed=seed)[1]["budget"] for seed in range(4)}
    assert len(budgets) == 4


def test_second_episode_has_the_full_budget(make_env):
    env = make_env(base_budget=10_000)
    env.reset(seed=1)
    _, info = env.reset()
    assert info["budget"] == 10_000


def test_truncation_reports_episode_summary(make_env):
    env = make_env(base_budget=2, stagger_first_episode=False)
    env.reset(seed=0)
    env.step(BUTTONS.index("down"))
    _, _, _, _, info = env.step(BUTTONS.index("down"))
    assert info["episode"]["length"] == 2
    assert info["episode"]["explore"] > 0


def test_episode_summary_reports_milestones(make_env):
    env = make_env(base_budget=1, budget_per_event=0, stagger_first_episode=False)
    env.reset(seed=0)
    env.pyboy.memory[0xD755] |= 0x80  # EVENT_BEAT_BROCK
    *_, info = env.step(BUTTONS.index("b"))
    assert info["episode"]["milestone_beat_brock"] == pytest.approx(1.0)


def test_event_gain_extends_the_budget(make_env):
    env = make_env(base_budget=10, budget_per_event=5, stagger_first_episode=False)
    assert env.budget_for(events_gained=2) == 20


def test_setting_an_event_flag_extends_the_current_episode(make_env):
    env = make_env(base_budget=2, budget_per_event=1, stagger_first_episode=False)
    env.reset(seed=0)
    env.step(BUTTONS.index("b"))
    env.pyboy.memory[0xD860] |= 1
    assert env.step(BUTTONS.index("b"))[3] is False


def test_visited_map_ignores_position_during_battle(make_env, monkeypatch):
    env = make_env()
    obs, _ = env.reset(seed=0)
    start = read_frame(env.pyboy.memory)
    monkeypatch.setattr("pokered.env.read_frame", lambda _mem: replace(start, in_battle=True, x=start.x + 1))
    obs, *_ = env.step(BUTTONS.index("b"))
    assert obs["visited"][0, 24, 24] == 0


def test_visited_map_marks_position_outside_battle(make_env, monkeypatch):
    env = make_env()
    env.reset(seed=0)
    start = read_frame(env.pyboy.memory)
    monkeypatch.setattr("pokered.env.read_frame", lambda _mem: replace(start, x=start.x + 1))
    obs, *_ = env.step(BUTTONS.index("b"))
    assert obs["visited"][0, 24, 24] == 255


def test_battle_animations_are_off_by_default(make_env):
    env = make_env()
    env.reset(seed=0)
    assert env.pyboy.memory[0xD355] & 0x80 == 0x80


def test_event_flags_observation_matches_flag_bytes(make_env):
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs["events"].shape == (EVENT_FLAG_BYTES,)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_staggered_budget_stays_within_the_full_budget(make_env, seed):
    assert make_env(base_budget=1).reset(seed=seed)[1]["budget"] == 1
