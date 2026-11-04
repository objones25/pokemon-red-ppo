"""The fixed evaluator: same start, sampled actions, and every env's first episode counted exactly once."""

from __future__ import annotations

import json
import math
from functools import partial

import numpy as np
import pytest
import torch
from support import ROM, requires_rom

from pokered.config import EnvConfig
from pokered.env import PokemonRedEnv
from pokered.evaluate import evaluate, summarize, write_report
from pokered.model import ActorCritic
from pokered.negative_space import CheckFailed
from pokered.observations import OBSERVATION_SHAPES
from pokered.vec import StepResult, VecEnv


class ScriptedEnvs:
    """Env i finishes its episodes after i + 1 steps, reporting which episode it was."""

    def __init__(self, num_envs: int = 3) -> None:
        self.num_envs = num_envs
        self.obs = {key: np.full((num_envs, *shape), 128, np.uint8) for key, shape in OBSERVATION_SHAPES.items()}
        self.steps = np.zeros(num_envs, np.int64)
        self.finished = np.zeros(num_envs, np.int64)
        self.actions: list[list[int]] = []

    def step(self, actions):
        self.actions.append(actions.tolist())
        self.steps += 1
        done = self.steps == np.arange(1, self.num_envs + 1)
        episodes = []
        for slot in np.nonzero(done)[0]:
            self.finished[slot] += 1
            episodes.append({"env": float(slot), "episode": float(self.finished[slot])})
        self.steps[done] = 0
        return StepResult(
            rewards=np.zeros(self.num_envs, np.float32), terminated=np.zeros(self.num_envs, np.bool_), truncated=done,
            final_obs=self.obs, episodes=episodes,
        )  # fmt: skip


def _evaluate(envs=None, seed=0, max_steps=50):
    return evaluate(ActorCritic(), envs or ScriptedEnvs(), device=torch.device("cpu"), seed=seed, max_steps=max_steps)


def test_counts_one_episode_per_env():
    assert sorted(episode["env"] for episode in _evaluate()) == [0.0, 1.0, 2.0]


def test_counts_each_envs_first_episode_not_a_later_short_one():
    assert [episode["episode"] for episode in _evaluate()] == [1.0, 1.0, 1.0]


def test_stops_once_every_env_has_finished():
    envs = ScriptedEnvs()
    _evaluate(envs)
    assert len(envs.actions) == 3


def test_same_seed_gives_same_actions():
    torch.manual_seed(0)
    model = ActorCritic()
    first, second = ScriptedEnvs(), ScriptedEnvs()
    evaluate(model, first, device=torch.device("cpu"), seed=7, max_steps=50)
    evaluate(model, second, device=torch.device("cpu"), seed=7, max_steps=50)
    assert first.actions == second.actions


def test_different_seeds_give_different_actions():
    torch.manual_seed(0)
    model = ActorCritic()
    first, second = ScriptedEnvs(16), ScriptedEnvs(16)
    evaluate(model, first, device=torch.device("cpu"), seed=1, max_steps=50)
    evaluate(model, second, device=torch.device("cpu"), seed=2, max_steps=50)
    assert first.actions != second.actions


def test_does_not_disturb_the_training_random_stream():
    torch.manual_seed(0)
    model = ActorCritic()
    before = torch.get_rng_state()
    evaluate(model, ScriptedEnvs(), device=torch.device("cpu"), seed=0, max_steps=50)
    assert torch.equal(before, torch.get_rng_state())


def test_restores_training_mode():
    model = ActorCritic()
    evaluate(model, ScriptedEnvs(), device=torch.device("cpu"), seed=0, max_steps=50)
    assert model.training


def test_refuses_to_run_past_the_step_limit():
    with pytest.raises(CheckFailed, match="did not finish"):
        _evaluate(max_steps=2)


def test_summary_reports_milestone_rate():
    report = summarize([{"milestone_beat_brock": 1.0, "steps_to_beat_brock": 10.0}, {"milestone_beat_brock": 0.0, "steps_to_beat_brock": math.nan}])
    assert report["milestone_beat_brock"] == pytest.approx(0.5)


def test_summary_averages_steps_over_episodes_that_reached_it():
    report = summarize([{"steps_to_beat_brock": 10.0}, {"steps_to_beat_brock": 20.0}, {"steps_to_beat_brock": math.nan}])
    assert report["steps_to_beat_brock"] == pytest.approx(15.0)


def test_summary_is_nan_free_when_no_episode_reached_a_milestone():
    report = summarize([{"steps_to_beat_brock": math.nan}])
    assert report["steps_to_beat_brock"] is None


def test_summary_counts_episodes():
    assert summarize([{"tiles": 1.0}, {"tiles": 3.0}])["episodes"] == 2


def test_summary_refuses_no_episodes():
    with pytest.raises(CheckFailed, match="no episodes"):
        summarize([])


def test_report_file_is_strict_json(tmp_path):
    path = tmp_path / "eval.json"
    write_report(path, {"steps_to_beat_brock": None, "episodes": 2})
    assert json.loads(path.read_text()) == {"steps_to_beat_brock": None, "episodes": 2}


@requires_rom
@pytest.mark.slow
def test_real_game_evaluation_reports_every_milestone():
    config = EnvConfig(rom=ROM, base_budget=8, stagger_first_episode=False)
    envs = VecEnv(partial(PokemonRedEnv, config), OBSERVATION_SHAPES, workers=1, envs_per_worker=2, seed=0)
    try:
        envs.reset()
        report = summarize(evaluate(ActorCritic(), envs, device=torch.device("cpu"), seed=0, max_steps=100))
    finally:
        envs.close()
    assert (report["episodes"], report["milestone_beat_brock"]) == (2, pytest.approx(0.0))


def test_refuses_a_zero_step_limit():
    with pytest.raises(CheckFailed, match="max_steps"):
        _evaluate(max_steps=0)


def test_accepts_a_step_limit_of_one_when_envs_finish_in_one_step():
    assert len(_evaluate(ScriptedEnvs(1), max_steps=1)) == 1


def test_sends_one_integer_action_per_env():
    envs = ScriptedEnvs()
    _evaluate(envs)
    assert all(isinstance(action, int) for row in envs.actions for action in row)


def test_report_creates_missing_directories(tmp_path):
    path = tmp_path / "a" / "b" / "eval.json"
    write_report(path, {"episodes": 1})
    assert json.loads(path.read_text()) == {"episodes": 1}


def test_report_refuses_nan(tmp_path):
    with pytest.raises(ValueError, match="JSON"):
        write_report(tmp_path / "eval.json", {"bad": math.nan})


def test_failed_report_leaves_no_temporary_file(tmp_path):
    with pytest.raises(ValueError, match="JSON"):
        write_report(tmp_path / "eval.json", {"bad": math.nan})
    assert list(tmp_path.iterdir()) == []
