"""Negative space: states the code must refuse rather than silently carry on with."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
import torch
from support import ROM, CountingEnv, emulator, requires_rom

from pokered.config import EnvConfig, PPOConfig
from pokered.emulator import restore
from pokered.env import PokemonRedEnv
from pokered.frame import EVENT_FLAG_BYTES, GameFrame
from pokered.model import ActorCritic
from pokered.negative_space import CheckFailed
from pokered.observations import OBSERVATION_SHAPES, FrameStack, downscale_screen
from pokered.ppo import Rollout, collect_rollout, compute_gae, make_optimizer, ppo_update
from pokered.rewards import (
    BadgeGain,
    BallStock,
    EpisodeReward,
    EventFlagGain,
    FirstTime,
    HiddenEventGain,
    LevelGain,
    NewMap,
    NewTile,
    TreeCut,
    knows_cut,
)
from pokered.vec import StepResult, VecEnv


def _frame(**overrides) -> GameFrame:
    values = {
        "map_id": 0, "x": 0, "y": 0, "in_battle": False, "event_flags": bytes(EVENT_FLAG_BYTES),
        "badges": 0, "party_levels": (6,), "party_hp": (1.0,),
    }  # fmt: skip
    return GameFrame(**(values | overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"event_flags": b"\x00"}, "event flags"),
        ({"party_hp": (1.0, 1.0)}, "party"),
        ({"party_hp": (1.5,)}, "hp"),
        ({"badges": 256}, "badges"),
    ],
)
def test_frame_refuses_impossible_values(overrides, message):
    with pytest.raises(CheckFailed, match=message):
        _frame(**overrides)


@pytest.mark.parametrize(
    "term", [NewTile, EventFlagGain, BadgeGain, LevelGain, HiddenEventGain, NewMap, BallStock, TreeCut]
)
def test_reward_term_refuses_negative_weight(term):
    with pytest.raises(CheckFailed, match="weight"):
        term(weight=-1.0)


def test_first_time_refuses_negative_weight():
    with pytest.raises(CheckFailed, match="weight"):
        FirstTime("knows_cut", -1.0, knows_cut)


class _Penalty:
    name = "penalty"
    signed: ClassVar[bool] = False

    def reset(self, first) -> None:
        pass

    def __call__(self, frame) -> float:
        return -1.0


def test_episode_reward_refuses_a_negative_term():
    episode = EpisodeReward([_Penalty()])
    episode.reset(_frame())
    with pytest.raises(CheckFailed, match="penalty"):
        episode.step(_frame())


def test_episode_reward_refuses_duplicate_term_names():
    with pytest.raises(CheckFailed, match="duplicate"):
        EpisodeReward([NewTile(weight=1.0), NewTile(weight=1.0)])


def test_downscale_refuses_wrong_screen_shape():
    with pytest.raises(CheckFailed, match="screen"):
        downscale_screen(np.zeros((72, 80), np.uint8))


def test_frame_stack_refuses_wrong_frame_shape():
    with pytest.raises(CheckFailed, match="frame"):
        FrameStack().push(np.zeros((144, 160), np.uint8))


def test_unpack_bits_refuses_non_byte_tensor():
    with pytest.raises(CheckFailed, match="uint8"):
        ActorCritic.unpack_bits(torch.zeros((1, 1)))


def test_model_refuses_float_screens():
    batch = {key: torch.zeros((1, *shape), dtype=torch.uint8) for key, shape in OBSERVATION_SHAPES.items()}
    batch["screens"] = batch["screens"].float()
    with pytest.raises(CheckFailed, match="screens"):
        ActorCritic()(batch)


def _gae(**overrides):
    values = {
        "rewards": torch.zeros(2, 1), "values": torch.zeros(2, 1), "next_values": torch.zeros(2, 1),
        "terminated": torch.zeros(2, 1), "truncated": torch.zeros(2, 1), "gamma": 0.9, "lam": 0.9,
    }  # fmt: skip
    return compute_gae(**(values | overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rewards": torch.zeros(3, 1)}, "rewards"),
        ({"gamma": 1.5}, "gamma"),
        ({"lam": -0.1}, "lam"),
        ({"rewards": torch.tensor([[float("nan")], [0.0]])}, "advantages"),
        ({"terminated": torch.ones(2, 1), "truncated": torch.ones(2, 1)}, "both"),
    ],
)
def test_gae_refuses_bad_inputs(overrides, message):
    with pytest.raises(CheckFailed, match=message):
        _gae(**overrides)


def test_collect_refuses_mismatched_env_count():
    class _Vec:
        def __init__(self) -> None:
            self.num_envs = 3
            self.obs: dict[str, np.ndarray] = {}
            self.map_ids = np.zeros(3, np.uint8)
            self.in_battle = np.zeros(3, np.bool_)

        def step(self, actions: np.ndarray) -> StepResult:
            raise AssertionError("must refuse before stepping")

    rollout = Rollout(1, 2, OBSERVATION_SHAPES, torch.device("cpu"))
    with pytest.raises(CheckFailed, match="envs"):
        collect_rollout(ActorCritic(), _Vec(), rollout, torch.device("cpu"))


def test_rollout_refuses_empty_size():
    with pytest.raises(CheckFailed, match="steps"):
        Rollout(0, 2, OBSERVATION_SHAPES, torch.device("cpu"))


def test_update_crashes_on_non_finite_returns():
    model = ActorCritic()
    config = PPOConfig(rollout_steps=2, minibatch_size=4, epochs=1)
    rollout = Rollout(2, 2, OBSERVATION_SHAPES, torch.device("cpu"))
    returns = torch.full((2, 2), float("nan"))
    with pytest.raises(CheckFailed, match="loss"):
        ppo_update(model, make_optimizer(model, config), rollout, torch.ones(2, 2), returns, config)


def test_update_refuses_mismatched_advantages():
    model = ActorCritic()
    config = PPOConfig(rollout_steps=2, minibatch_size=4, epochs=1)
    rollout = Rollout(2, 2, OBSERVATION_SHAPES, torch.device("cpu"))
    with pytest.raises(CheckFailed, match="advantages"):
        ppo_update(model, make_optimizer(model, config), rollout, torch.ones(3, 2), torch.ones(2, 2), config)


def test_vec_refuses_wrong_action_count():
    vec = VecEnv(CountingEnv, CountingEnv.SHAPES, workers=1, envs_per_worker=1, seed=0)
    try:
        with pytest.raises(CheckFailed, match="actions"):
            vec.step(np.zeros(2, np.int64))
    finally:
        vec.close()


@requires_rom
def test_restore_refuses_empty_state():
    with emulator() as pyboy, pytest.raises(CheckFailed, match="state"):
        restore(pyboy, b"", battle_animations=False)


@requires_rom
@pytest.mark.parametrize("action", [-1, 7])
def test_env_refuses_action_outside_space(action):
    env = PokemonRedEnv(EnvConfig(rom=ROM))
    try:
        env.reset(seed=0)
        with pytest.raises(CheckFailed, match="action"):
            env.step(action)
    finally:
        env.close()
