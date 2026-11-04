"""The training loop, on a fake env that pays only for action 3, so learning is checkable in seconds."""

from __future__ import annotations

import math
import signal
from functools import partial

import numpy as np
import pytest
import torch

from pokered.checkpoints import CheckpointStore, ConfigChanged
from pokered.config import EnvConfig, PPOConfig, TrainConfig
from pokered.model import ActorCritic
from pokered.negative_space import CheckFailed, expect
from pokered.observations import OBSERVATION_SHAPES
from pokered.runpod import TerminateFailed
from pokered.train import PodStopping, episode_metrics, holding_sigterm, pick_device, run_guarded, train
from pokered.vec import StepResult

EPISODE_LENGTH = 6


class FakeVec:
    def __init__(self, num_envs: int = 4) -> None:
        self.num_envs = num_envs
        # Nonzero: on all-zero inputs the ReLU body outputs zero and only the actor's bias could learn.
        self.obs = {key: np.full((num_envs, *shape), 128, np.uint8) for key, shape in OBSERVATION_SHAPES.items()}
        self._t = 0
        self.action_counts = np.zeros(7)

    def reset(self):
        return self.obs

    def step(self, actions):
        self._t += 1
        self.action_counts += np.bincount(actions, minlength=7)
        done = self._t % EPISODE_LENGTH == 0
        return StepResult(
            rewards=(actions == 3).astype(np.float32),
            terminated=np.zeros(self.num_envs, np.bool_),
            truncated=np.full(self.num_envs, done),
            final_obs=self.obs,
            episodes=[{"tiles": 2.0, "events": 1.0}] * self.num_envs if done else [],
        )


class ListLogger:
    def __init__(self) -> None:
        self.records: list[tuple[int, dict]] = []

    def log(self, metrics, *, step: int) -> None:
        self.records.append((step, metrics))

    def finish(self) -> None:
        pass


def _latest(store: CheckpointStore) -> dict:
    return expect(store.load_latest(), "a checkpoint should exist here")


def _config(tmp_path, total_steps=64, **ppo) -> TrainConfig:
    return TrainConfig(
        env=EnvConfig(rom=tmp_path / "unused.gb"),
        ppo=PPOConfig(rollout_steps=8, minibatch_size=16, epochs=2, **ppo),
        workers=2,
        envs_per_worker=2,
        total_steps=total_steps,
        run_dir=tmp_path,
        checkpoint_every_steps=32,
    )


def _train(config, *, seed=0, envs=None, logger=None) -> tuple[ActorCritic, ListLogger, CheckpointStore]:
    torch.manual_seed(seed)
    logger = logger or ListLogger()
    store = CheckpointStore(config.run_dir / "checkpoints")
    model = ActorCritic()
    train(config, model, envs or FakeVec(), logger, store, torch.device("cpu"))
    return model, logger, store


def test_batch_size_is_rollout_times_envs(tmp_path):
    assert _config(tmp_path).batch_size == 32


def test_rejects_minibatch_larger_than_batch(tmp_path):
    with pytest.raises(ValueError, match="minibatch_size 64 exceeds batch 8"):
        TrainConfig(
            env=EnvConfig(rom=tmp_path / "x"), ppo=PPOConfig(rollout_steps=2, minibatch_size=64),
            workers=2, envs_per_worker=2, total_steps=8, run_dir=tmp_path,
        )  # fmt: skip


def test_trains_for_requested_steps(tmp_path):
    _, logger, _ = _train(_config(tmp_path))
    assert logger.records[-1][0] == 64


def test_logs_ppo_health_metrics(tmp_path):
    metrics = _train(_config(tmp_path))[1].records[-1][1]
    assert {"ppo/approx_kl", "ppo/clip_fraction", "ppo/explained_variance", "ppo/entropy", "ppo/grad_norm"} <= metrics.keys()


def test_logs_throughput(tmp_path):
    assert _train(_config(tmp_path))[1].records[-1][1]["perf/steps_per_second"] > 0


def test_logs_mean_and_max_of_finished_episodes(tmp_path):
    metrics = _train(_config(tmp_path))[1].records[-1][1]
    assert (metrics["episode/tiles_mean"], metrics["episode/events_max"]) == (pytest.approx(2.0), pytest.approx(1.0))


def test_writes_a_final_checkpoint(tmp_path):
    assert _latest(_train(_config(tmp_path))[2])["env_steps"] == 64


def test_resume_continues_from_checkpoint_steps(tmp_path):
    _train(_config(tmp_path, total_steps=32))
    _, logger, _ = _train(_config(tmp_path, total_steps=64))
    assert [step for step, _ in logger.records] == [64]


def test_resume_restores_model_weights(tmp_path):
    first, *_ = _train(_config(tmp_path, total_steps=32))
    restarted = ActorCritic()
    torch.manual_seed(99)
    train(_config(tmp_path, total_steps=32), restarted, FakeVec(), ListLogger(), CheckpointStore(tmp_path / "checkpoints"), torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(first.state_dict().values(), restarted.state_dict().values(), strict=True))


def test_resume_matches_uninterrupted_training_with_same_data(tmp_path):
    straight, *_ = _train(_config(tmp_path / "a", total_steps=64))
    _train(_config(tmp_path / "b", total_steps=32))
    resumed, *_ = _train(_config(tmp_path / "b", total_steps=64), seed=123, envs=_advanced_fake())
    assert all(torch.equal(a, b) for a, b in zip(straight.state_dict().values(), resumed.state_dict().values(), strict=True))


def _advanced_fake() -> FakeVec:
    """A fake env already 8 lockstep steps in: 32 samples across 4 envs."""
    fake = FakeVec()
    fake._t = 8
    return fake


def test_resume_refuses_changed_hyperparameters(tmp_path):
    _train(_config(tmp_path, total_steps=32))
    with pytest.raises(ConfigChanged, match="learning_rate"):
        _train(_config(tmp_path, total_steps=64, learning_rate=1e-5))


def test_checkpoint_holds_optimizer_state(tmp_path):
    payload = _latest(_train(_config(tmp_path))[2])
    assert payload["optimizer"]["state"]


def test_policy_learns_the_paying_action(tmp_path):
    config = TrainConfig(
        env=EnvConfig(rom=tmp_path / "x"),
        ppo=PPOConfig(rollout_steps=32, minibatch_size=64, epochs=3, learning_rate=1e-3),
        workers=2, envs_per_worker=2, total_steps=1_280, run_dir=tmp_path, checkpoint_every_steps=10_000,
    )  # fmt: skip
    model, *_ = _train(config)
    obs = {key: torch.full((1, *shape), 128, dtype=torch.uint8) for key, shape in OBSERVATION_SHAPES.items()}
    with torch.no_grad():
        assert model(obs)[0].softmax(1)[0, 3].item() > 0.5


def test_episode_metrics_ignore_episodes_that_never_reached_a_milestone():
    metrics = episode_metrics([{"steps_to_x": 10.0}, {"steps_to_x": math.nan}])
    assert metrics["episode/steps_to_x_mean"] == pytest.approx(10.0)


def test_episode_metrics_omit_a_statistic_with_no_values():
    assert "episode/steps_to_x_mean" not in episode_metrics([{"steps_to_x": math.nan}])


def test_pick_device_returns_a_usable_device():
    assert torch.zeros(1, device=pick_device()).numel() == 1


class FakeTerminator:
    def __init__(self, *, fails: bool = False) -> None:
        self.calls = 0
        self._fails = fails

    def terminate(self) -> None:
        self.calls += 1
        if self._fails:
            raise TerminateFailed("pod still running")


def _work_raising(error: BaseException):
    def work() -> None:
        raise error

    return work


def test_guard_terminates_after_success():
    terminator = FakeTerminator()
    run_guarded(lambda: None, terminator)
    assert terminator.calls == 1


def test_guard_terminates_after_failure():
    terminator = FakeTerminator()
    with pytest.raises(ValueError, match="training broke"):
        run_guarded(_work_raising(ValueError("training broke")), terminator)
    assert terminator.calls == 1


def test_guard_keeps_the_pod_on_ctrl_c():
    terminator = FakeTerminator()
    with pytest.raises(KeyboardInterrupt):
        run_guarded(_work_raising(KeyboardInterrupt()), terminator)
    assert terminator.calls == 0


def test_guard_terminates_when_sigterm_arrives():
    terminator = FakeTerminator()
    with pytest.raises(PodStopping):
        run_guarded(lambda: signal.raise_signal(signal.SIGTERM), terminator)
    assert terminator.calls == 1


def test_guard_restores_the_previous_sigterm_handler():
    before = signal.getsignal(signal.SIGTERM)
    run_guarded(lambda: None, FakeTerminator())
    assert signal.getsignal(signal.SIGTERM) is before


def test_guard_runs_without_a_terminator():
    calls = []
    run_guarded(lambda: calls.append(1), None)
    assert calls == [1]


def test_guard_reports_a_failed_termination_loudly():
    with pytest.raises(TerminateFailed, match="still running"):
        run_guarded(lambda: None, FakeTerminator(fails=True))


def test_guard_keeps_training_error_when_termination_also_fails():
    with pytest.raises(ValueError, match="training broke"):
        run_guarded(_work_raising(ValueError("training broke")), FakeTerminator(fails=True))


class StoppingVec(FakeVec):
    def __init__(self, stop_at: int) -> None:
        super().__init__()
        self._stop_at = stop_at

    def step(self, actions):
        if self._t == self._stop_at:
            raise PodStopping("received SIGTERM")
        return super().step(actions)


def test_sigterm_during_rollout_saves_the_last_finished_update(tmp_path):
    config = _config(tmp_path, total_steps=64)
    store = CheckpointStore(tmp_path / "checkpoints")
    with pytest.raises(PodStopping):
        train(config, ActorCritic(), StoppingVec(stop_at=10), ListLogger(), store, torch.device("cpu"))
    assert _latest(store)["env_steps"] == 32


class StoppingLogger(ListLogger):
    """SIGTERM lands while the first update's results are being logged, which is outside the rollout."""

    def log(self, metrics, *, step: int) -> None:
        super().log(metrics, step=step)
        signal.raise_signal(signal.SIGTERM)


def test_sigterm_outside_the_rollout_still_saves_the_finished_update(tmp_path):
    config = _config(tmp_path, total_steps=64)
    store = CheckpointStore(tmp_path / "checkpoints")
    work = partial(train, config, ActorCritic(), FakeVec(), StoppingLogger(), store, torch.device("cpu"))
    with pytest.raises(PodStopping):
        run_guarded(work, None)
    assert _latest(store)["env_steps"] == 32


def test_sigterm_during_the_update_lets_the_update_finish(tmp_path, monkeypatch):
    completed = []

    def update_interrupted_midway(*args):
        signal.raise_signal(signal.SIGTERM)
        completed.append(True)  # A handler that raised would never reach this line.
        return {}

    monkeypatch.setattr("pokered.train.ppo_update", update_interrupted_midway)
    config = _config(tmp_path, total_steps=64)
    work = partial(train, config, ActorCritic(), FakeVec(), ListLogger(), CheckpointStore(tmp_path / "checkpoints"), torch.device("cpu"))
    with pytest.raises(PodStopping):
        run_guarded(work, None)
    assert completed == [True]


def test_holding_sigterm_restores_the_handler():
    before = signal.getsignal(signal.SIGTERM)
    with holding_sigterm():
        assert signal.getsignal(signal.SIGTERM) is not before
    assert signal.getsignal(signal.SIGTERM) is before


def test_sigterm_before_any_update_saves_nothing(tmp_path):
    config = _config(tmp_path, total_steps=64)
    store = CheckpointStore(tmp_path / "checkpoints")
    with pytest.raises(PodStopping):
        train(config, ActorCritic(), StoppingVec(stop_at=2), ListLogger(), store, torch.device("cpu"))
    assert store.load_latest() is None


def test_episode_metrics_of_nothing_is_empty():
    assert episode_metrics([]) == {}


def test_episode_metrics_refuse_mismatched_statistics():
    with pytest.raises(CheckFailed, match="different statistics"):
        episode_metrics([{"tiles": 1.0}, {"events": 1.0}])


def test_episode_metrics_refuse_one_mismatch_among_many():
    with pytest.raises(CheckFailed, match="different statistics"):
        episode_metrics([{"tiles": 1.0}, {"tiles": 1.0}, {"events": 1.0}])


def test_episode_metrics_report_the_largest_value():
    assert episode_metrics([{"tiles": 1.0}, {"tiles": 5.0}])["episode/tiles_max"] == pytest.approx(5.0)


def test_throughput_metrics_use_the_measured_times(tmp_path, monkeypatch):
    clock = iter([100.0, 102.0, 108.0, 200.0, 201.0, 202.0])
    monkeypatch.setattr("pokered.train.time.perf_counter", lambda: next(clock))
    metrics = _train(_config(tmp_path, total_steps=32))[1].records[0][1]
    assert metrics["perf/collect_seconds"] == pytest.approx(2.0)
    assert metrics["perf/update_seconds"] == pytest.approx(6.0)
    assert metrics["perf/steps_per_second"] == pytest.approx(32 / 8.0)


def _checkpoint_steps(tmp_path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in (tmp_path / "checkpoints").glob("checkpoint_*.pt"))


def _cadence_config(tmp_path, every: int, total: int) -> TrainConfig:
    return TrainConfig(
        env=EnvConfig(rom=tmp_path / "x"), ppo=PPOConfig(rollout_steps=8, minibatch_size=16, epochs=1),
        workers=2, envs_per_worker=2, total_steps=total, run_dir=tmp_path, checkpoint_every_steps=every,
    )  # fmt: skip


def test_checkpoints_save_when_crossing_each_interval_and_at_the_end(tmp_path):
    torch.manual_seed(0)
    config = _cadence_config(tmp_path, every=48, total=128)
    store = CheckpointStore(tmp_path / "checkpoints", keep=10)
    train(config, ActorCritic(), FakeVec(), ListLogger(), store, torch.device("cpu"))
    assert _checkpoint_steps(tmp_path) == [64, 96, 128]


def test_checkpoint_lands_when_an_interval_is_hit_exactly(tmp_path):
    torch.manual_seed(0)
    config = _cadence_config(tmp_path, every=32, total=32)
    store = CheckpointStore(tmp_path / "checkpoints", keep=10)
    train(config, ActorCritic(), FakeVec(), ListLogger(), store, torch.device("cpu"))
    assert _checkpoint_steps(tmp_path) == [32]


def test_no_checkpoint_before_the_first_interval(tmp_path):
    torch.manual_seed(0)
    config = _cadence_config(tmp_path, every=1000, total=64)
    store = CheckpointStore(tmp_path / "checkpoints", keep=10)
    train(config, ActorCritic(), FakeVec(), ListLogger(), store, torch.device("cpu"))
    assert _checkpoint_steps(tmp_path) == [64]
