"""Worker processes stepping several envs each, with observations in shared memory."""

from __future__ import annotations

import os
import sys
from functools import partial
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest
from support import ROM, AffinityEnv, CountingEnv, requires_rom

from pokered.buttons import BUTTONS
from pokered.config import EnvConfig
from pokered.env import PokemonRedEnv
from pokered.negative_space import CheckFailed
from pokered.observations import OBSERVATION_SHAPES
from pokered.vec import VecEnv, WorkerFailure


@pytest.fixture
def make_vec():
    created = []

    def make(factory=CountingEnv, workers=2, envs_per_worker=2, shapes=CountingEnv.SHAPES, seed=10) -> VecEnv:
        vec = VecEnv(factory, shapes, workers=workers, envs_per_worker=envs_per_worker, seed=seed)
        created.append(vec)
        return vec

    yield make
    for vec in created:
        vec.close()


def test_num_envs_is_workers_times_envs_per_worker(make_vec):
    assert make_vec(workers=2, envs_per_worker=3).num_envs == 6


def test_reset_seeds_each_env_differently(make_vec):
    assert make_vec().reset()["x"][:, 0].tolist() == [10, 11, 12, 13]


def test_step_routes_each_action_to_its_env(make_vec):
    vec = make_vec()
    vec.reset()
    assert vec.step(np.array([3, 1, 4, 5])).rewards.tolist() == [3.0, 1.0, 4.0, 5.0]


def test_step_updates_observations(make_vec):
    vec = make_vec()
    vec.reset()
    vec.step(np.zeros(4, np.int64))
    assert vec.obs["x"][:, 1].tolist() == [1, 1, 1, 1]


def test_truncation_flag_is_set_on_the_last_step_only(make_vec):
    vec = make_vec()
    vec.reset()
    flags = [vec.step(np.zeros(4, np.int64)).truncated.tolist() for _ in range(3)]
    assert flags == [[False] * 4, [False] * 4, [True] * 4]


def test_truncated_env_keeps_its_final_observation(make_vec):
    vec = make_vec()
    vec.reset()
    results = [vec.step(np.zeros(4, np.int64)) for _ in range(3)]
    assert results[-1].final_obs["x"][:, 1].tolist() == [3, 3, 3, 3]


def test_truncated_env_auto_resets(make_vec):
    vec = make_vec()
    vec.reset()
    for _ in range(3):
        vec.step(np.zeros(4, np.int64))
    assert vec.obs["x"][:, 1].tolist() == [0, 0, 0, 0]


def test_finished_episodes_are_reported_once(make_vec):
    vec = make_vec()
    vec.reset()
    counts = [len(vec.step(np.zeros(4, np.int64)).episodes) for _ in range(3)]
    assert counts == [0, 0, 4]


def test_worker_crash_raises_with_the_cause(make_vec):
    vec = make_vec(factory=partial(CountingEnv, crash_at=0), workers=1, envs_per_worker=1)
    vec.reset()
    with pytest.raises(WorkerFailure, match="planted crash"):
        vec.step(np.zeros(1, np.int64))


def test_close_releases_shared_memory():
    vec = VecEnv(CountingEnv, CountingEnv.SHAPES, workers=1, envs_per_worker=1, seed=0)
    name = vec.shared_memory_name
    vec.close()
    with pytest.raises(FileNotFoundError, match=name.lstrip("/")):
        SharedMemory(name=name)


def _first_env_screens(vec: VecEnv, actions) -> list[np.ndarray]:
    screens = []
    for action in actions:
        vec.step(np.array([action, 0]))
        screens.append(vec.obs["screens"][0].copy())
    return screens


@requires_rom
@pytest.mark.slow
def test_real_envs_match_direct_stepping():
    config = EnvConfig(rom=ROM, stagger_first_episode=False)
    actions = np.random.default_rng(0).integers(0, len(BUTTONS), 60)
    direct = PokemonRedEnv(config)
    direct.reset(seed=0)
    expected = [direct.step(int(a))[0]["screens"].copy() for a in actions]
    direct.close()
    vec = VecEnv(partial(PokemonRedEnv, config), OBSERVATION_SHAPES, workers=1, envs_per_worker=2, seed=0)
    vec.reset()
    got = _first_env_screens(vec, actions)
    vec.close()
    assert all(np.array_equal(x, y) for x, y in zip(expected, got, strict=True))


def _first_allowed_cpu() -> int:
    if sys.platform != "linux":
        raise RuntimeError("needs Linux")
    return min(os.sched_getaffinity(0))


@pytest.mark.skipif(sys.platform != "linux", reason="CPU pinning needs Linux")
def test_pinned_worker_runs_on_its_core():
    core = _first_allowed_cpu()
    vec = VecEnv(AffinityEnv, CountingEnv.SHAPES, workers=1, envs_per_worker=1, seed=0, worker_cores=[core])
    try:
        assert vec.step(np.zeros(1, np.int64)).rewards.tolist() == [float(core)]
    finally:
        vec.close()


def test_vec_refuses_a_core_list_of_the_wrong_length():
    with pytest.raises(CheckFailed, match="one core per worker"):
        VecEnv(CountingEnv, CountingEnv.SHAPES, workers=2, envs_per_worker=1, seed=0, worker_cores=[0])
