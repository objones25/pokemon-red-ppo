"""GAE and the PPO update."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from support import fake_episode

from pokered.config import PPOConfig
from pokered.model import ActorCritic
from pokered.negative_space import CheckFailed
from pokered.observations import OBSERVATION_SHAPES
from pokered.ppo import (
    Rollout,
    collect_rollout,
    compute_gae,
    explained_variance,
    learning_by_place,
    make_optimizer,
    minibatch_losses,
    normalize_advantage,
    place_totals,
    ppo_update,
    total_loss,
)
from pokered.vec import StepResult


def _t(*rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def test_gae_matches_hand_computation_without_episode_end():
    # gamma .5, lambda .5: d1 = 1 + .5*3 - 2 = .5; d0 = 1 + .5*2 - 1 = 1; a1 = .5; a0 = 1 + .25*.5 = 1.125.
    adv, returns = compute_gae(
        rewards=_t([1.0], [1.0]), values=_t([1.0], [2.0]), next_values=_t([2.0], [3.0]),
        terminated=_t([0.0], [0.0]), truncated=_t([0.0], [0.0]), gamma=0.5, lam=0.5,
    )  # fmt: skip
    assert adv.squeeze(1).tolist() == pytest.approx([1.125, 0.5])
    assert returns.squeeze(1).tolist() == pytest.approx([2.125, 2.5])


def test_gae_bootstraps_from_final_state_on_truncation():
    # Truncated at step 0: delta keeps gamma * V(final state) = .5 * 10; the trace stops, so a0 = 1 + 5 - 1 = 5.
    adv, _ = compute_gae(
        rewards=_t([1.0], [100.0]), values=_t([1.0], [0.0]), next_values=_t([10.0], [0.0]),
        terminated=_t([0.0], [0.0]), truncated=_t([1.0], [0.0]), gamma=0.5, lam=0.5,
    )  # fmt: skip
    assert adv[0, 0].item() == pytest.approx(5.0)


def test_gae_drops_future_value_on_termination():
    adv, _ = compute_gae(
        rewards=_t([1.0], [100.0]), values=_t([1.0], [0.0]), next_values=_t([10.0], [0.0]),
        terminated=_t([1.0], [0.0]), truncated=_t([0.0], [0.0]), gamma=0.5, lam=0.5,
    )  # fmt: skip
    assert adv[0, 0].item() == pytest.approx(0.0)


def test_gae_keeps_envs_independent():
    adv, _ = compute_gae(
        rewards=_t([1.0, 0.0]), values=_t([0.0, 0.0]), next_values=_t([0.0, 0.0]),
        terminated=_t([0.0, 0.0]), truncated=_t([0.0, 0.0]), gamma=1.0, lam=1.0,
    )  # fmt: skip
    assert adv[0].tolist() == pytest.approx([1.0, 0.0])


def _rollout(steps=8, envs=4) -> Rollout:
    rollout = Rollout(steps, envs, OBSERVATION_SHAPES, torch.device("cpu"))
    generator = torch.Generator().manual_seed(0)
    for key in rollout.obs:
        rollout.obs[key] = torch.randint(0, 256, rollout.obs[key].shape, dtype=torch.uint8, generator=generator)
    return rollout


def _config(**overrides) -> PPOConfig:
    return PPOConfig(rollout_steps=8, minibatch_size=8, epochs=4, **overrides)


def test_update_makes_rewarded_action_more_likely():
    torch.manual_seed(0)
    model = ActorCritic()
    rollout = _rollout()
    flat = {k: v.flatten(0, 1) for k, v in rollout.obs.items()}
    with torch.no_grad():
        logits, values = model(flat)
    dist = torch.distributions.Categorical(logits=logits)
    rollout.actions[:] = torch.randint(0, 7, rollout.actions.shape, generator=torch.Generator().manual_seed(1))
    rollout.log_probs[:] = dist.log_prob(rollout.actions.flatten()).view(8, 4)
    rollout.values[:] = values.view(8, 4)
    advantages = (rollout.actions == 3).float() * 2 - 1
    before = logits.softmax(1)[:, 3].mean().item()
    ppo_update(model, make_optimizer(model, _config()), rollout, advantages, rollout.values + advantages, _config())
    with torch.no_grad():
        after = model(flat)[0].softmax(1)[:, 3].mean().item()
    assert after > before


def _value_loss_after_updates(model, optimizer, rollout, flat, returns, config, count) -> float:
    losses = []
    for _ in range(count):
        with torch.no_grad():
            rollout.values[:] = model(flat)[1].view(8, 4)
        losses.append(ppo_update(model, optimizer, rollout, torch.ones(8, 4), returns, config)["value_loss"])
    return losses[-1]


def test_update_reduces_value_error():
    torch.manual_seed(0)
    model = ActorCritic()
    rollout = _rollout()
    flat = {k: v.flatten(0, 1) for k, v in rollout.obs.items()}
    returns = torch.full((8, 4), 3.0)
    config = _config()
    optimizer = make_optimizer(model, config)
    first = ppo_update(model, optimizer, rollout, torch.ones(8, 4), returns, config)
    assert _value_loss_after_updates(model, optimizer, rollout, flat, returns, config, 5) < first["value_loss"]


@pytest.mark.parametrize("key", ["approx_kl", "clip_fraction", "explained_variance", "entropy", "grad_norm", "value_loss", "approx_kl_max", "update_ratio", "weight_norm"])
def test_update_reports_health_metric(key):
    torch.manual_seed(0)
    model = ActorCritic()
    rollout = _rollout()
    returns = torch.arange(32.0).view(8, 4)
    metrics = ppo_update(model, make_optimizer(model, _config()), rollout, torch.ones(8, 4), returns, _config())
    assert math.isfinite(metrics[key])


def test_optimizer_is_adamw_with_configured_learning_rate():
    optimizer = make_optimizer(ActorCritic(), PPOConfig(learning_rate=1e-3))
    assert (type(optimizer).__name__, optimizer.param_groups[0]["lr"]) == ("AdamW", 1e-3)


def test_explained_variance_is_one_for_perfect_values():
    torch.manual_seed(0)
    model = ActorCritic()
    rollout = _rollout()
    returns = torch.arange(32.0).view(8, 4)
    rollout.values[:] = returns
    metrics = ppo_update(model, make_optimizer(model, _config()), rollout, torch.ones(8, 4), returns, _config())
    assert metrics["explained_variance"] == pytest.approx(1.0)


class _ValueIsFirstPartyByte(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        return torch.zeros(obs["party"].shape[0], 7), obs["party"][:, 0].float()


class _ScriptedVec:
    """Env 0 truncates on the first step with final-state value 50; env 1 runs on."""

    num_envs = 2

    def __init__(self) -> None:
        self.obs = {key: np.zeros((2, *shape), np.uint8) for key, shape in OBSERVATION_SHAPES.items()}
        self.obs["party"][:, 0] = [1, 2]
        self.map_ids = np.array([3, 4], np.uint8)
        self.in_battle = np.array([False, True])
        self._t = 0

    def step(self, _actions):
        final = {key: np.zeros((2, *shape), np.uint8) for key, shape in OBSERVATION_SHAPES.items()}
        final["party"][0, 0] = 50
        truncated = np.array([self._t == 0, False])
        self._t += 1
        self.obs["party"][:, 0] += 1
        return StepResult(
            rewards=np.ones(2, np.float32), terminated=np.zeros(2, np.bool_), truncated=truncated,
            final_obs=final, episodes=[fake_episode(length=1.0)] if truncated.any() else [],
        )  # fmt: skip


def _collected() -> tuple[Rollout, list]:
    rollout = Rollout(2, 2, OBSERVATION_SHAPES, torch.device("cpu"))
    episodes = collect_rollout(_ValueIsFirstPartyByte(), _ScriptedVec(), rollout, torch.device("cpu"))
    return rollout, episodes


def test_collect_uses_final_state_value_for_truncated_env():
    assert _collected()[0].next_values[0, 0].item() == pytest.approx(50.0)


def test_collect_uses_next_step_value_for_running_env():
    assert _collected()[0].next_values[0, 1].item() == pytest.approx(3.0)


def test_collect_bootstraps_last_step_from_current_observation():
    assert _collected()[0].next_values[1].tolist() == pytest.approx([3.0, 4.0])


def test_collect_records_the_observation_seen_before_stepping():
    assert _collected()[0].obs["party"][1, :, 0].tolist() == [2, 3]


def test_collect_returns_finished_episodes():
    assert [episode.summary for episode in _collected()[1]] == [{"length": 1.0}]


# Hand values: ratio 1.5 clips to 1.2 at clip 0.2. Value 2 from old 1 clips to 1.2, against a return of 3.
_RATIO_LOG = float(np.log(1.5))


def _losses(advantage=1.0, log_ratio=_RATIO_LOG):
    return minibatch_losses(
        log_ratio=torch.tensor([log_ratio]), advantage=torch.tensor([advantage]), values=torch.tensor([2.0]),
        old_values=torch.tensor([1.0]), returns=torch.tensor([3.0]), clip=0.2,
    )  # fmt: skip


def test_policy_loss_clips_the_ratio_for_positive_advantage():
    assert _losses(advantage=1.0).policy.item() == pytest.approx(-1.2)


def test_policy_loss_keeps_the_worse_unclipped_ratio_for_negative_advantage():
    assert _losses(advantage=-1.0).policy.item() == pytest.approx(1.5)


def test_value_loss_takes_the_larger_of_clipped_and_unclipped():
    assert _losses().value.item() == pytest.approx(1.8**2)


def test_approx_kl_matches_k3_estimator():
    assert _losses().approx_kl.item() == pytest.approx(0.5 - np.log(1.5))


@pytest.mark.parametrize(("log_ratio", "expected"), [(float(np.log(1.5)), 1.0), (float(np.log(1.1)), 0.0)])
def test_clip_fraction_counts_ratios_outside_the_clip(log_ratio, expected):
    assert _losses(log_ratio=log_ratio).clip_fraction.item() == pytest.approx(expected)


def test_total_loss_applies_value_and_entropy_coefficients():
    config = PPOConfig(vf_coef=0.5, ent_coef=0.1)
    total = total_loss(torch.tensor(1.0), torch.tensor(4.0), torch.tensor(2.0), config)
    assert total.item() == pytest.approx(1.0 + 2.0 - 0.2)


def test_normalized_advantage_has_zero_mean_and_unit_std():
    out = normalize_advantage(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert (out.mean().item(), out.std().item()) == (pytest.approx(0.0, abs=1e-6), pytest.approx(1.0, abs=1e-5))


# --- Pinning what the mutation pass found unpinned ---------------------------


def test_policy_loss_clips_a_low_ratio_for_negative_advantage():
    # Ratio 0.5 clips up to 0.8: min(-0.5, -0.8) = -0.8, so the loss is 0.8.
    assert _losses(advantage=-1.0, log_ratio=float(np.log(0.5))).policy.item() == pytest.approx(0.8)


def test_policy_loss_keeps_a_low_unclipped_ratio_for_positive_advantage():
    assert _losses(advantage=1.0, log_ratio=float(np.log(0.5))).policy.item() == pytest.approx(-0.5)


def _value_loss(values: float, old: float, returns: float) -> float:
    return minibatch_losses(
        log_ratio=torch.zeros(1), advantage=torch.ones(1), values=torch.tensor([values]),
        old_values=torch.tensor([old]), returns=torch.tensor([returns]), clip=0.2,
    ).value.item()  # fmt: skip


def test_value_loss_uses_unclipped_error_when_it_is_larger():
    # Value 0.5 from old 1 clips to 0.8. Against a return of 3 the unclipped error, 2.5, is larger.
    assert _value_loss(0.5, 1.0, 3.0) == pytest.approx(2.5**2)


def test_value_loss_clips_downward_moves():
    # Value 0.5 from old 1 clips to 0.8. Against a return of 0 the clipped error, 0.8, is larger.
    assert _value_loss(0.5, 1.0, 0.0) == pytest.approx(0.8**2)


def test_losses_refuse_zero_clip():
    with pytest.raises(CheckFailed, match="clip must be positive"):
        minibatch_losses(
            log_ratio=torch.zeros(1), advantage=torch.ones(1), values=torch.zeros(1),
            old_values=torch.zeros(1), returns=torch.zeros(1), clip=0.0,
        )  # fmt: skip


def test_normalize_refuses_a_single_advantage():
    with pytest.raises(CheckFailed, match="at least two"):
        normalize_advantage(torch.ones(1))


def test_normalize_accepts_two_advantages():
    assert normalize_advantage(torch.tensor([1.0, 3.0])).tolist() == pytest.approx([-0.7071068, 0.7071068], abs=1e-5)


@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_gae_accepts_gamma_at_its_limits(gamma):
    _, returns = compute_gae(
        rewards=_t([1.0]), values=_t([0.0]), next_values=_t([0.0]),
        terminated=_t([0.0]), truncated=_t([0.0]), gamma=gamma, lam=0.5,
    )  # fmt: skip
    assert returns.item() == pytest.approx(1.0)


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_gae_accepts_lambda_at_its_limits(lam):
    _, returns = compute_gae(
        rewards=_t([1.0]), values=_t([0.0]), next_values=_t([0.0]),
        terminated=_t([0.0]), truncated=_t([0.0]), gamma=0.5, lam=lam,
    )  # fmt: skip
    assert returns.item() == pytest.approx(1.0)


def _synced_rollout(model: ActorCritic) -> tuple[Rollout, torch.Tensor]:
    """A rollout whose stored log-probs and values come from ``model``, so the first update starts at ratio 1."""
    rollout = _rollout()
    flat = {k: v.flatten(0, 1) for k, v in rollout.obs.items()}
    rollout.actions[:] = torch.randint(0, 7, rollout.actions.shape, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits, values = model(flat)
    rollout.log_probs[:] = torch.distributions.Categorical(logits=logits).log_prob(rollout.actions.flatten()).view(8, 4)
    rollout.values[:] = values.view(8, 4)
    return rollout, values


def _one_full_batch_update() -> tuple[dict[str, float], torch.Tensor, Rollout]:
    torch.manual_seed(0)
    model = ActorCritic()
    rollout, values = _synced_rollout(model)
    returns = rollout.values + 1.0
    config = PPOConfig(rollout_steps=8, minibatch_size=32, epochs=1, learning_rate=1e-6)
    metrics = ppo_update(model, make_optimizer(model, config), rollout, torch.arange(32.0).view(8, 4), returns, config)
    return metrics, values, rollout


def test_first_update_step_starts_with_zero_kl():
    assert _one_full_batch_update()[0]["approx_kl"] == pytest.approx(0.0, abs=1e-6)


def test_first_update_step_starts_with_nothing_clipped():
    assert _one_full_batch_update()[0]["clip_fraction"] == pytest.approx(0.0)


def test_value_loss_metric_is_the_minibatch_mean_not_a_sum():
    # Returns are values + 1 and the old values are the current ones, so the unclipped error is 1 everywhere.
    assert _one_full_batch_update()[0]["value_loss"] == pytest.approx(1.0, rel=1e-3)


def test_entropy_metric_matches_a_near_uniform_policy():
    assert _one_full_batch_update()[0]["entropy"] == pytest.approx(math.log(7), abs=0.01)


def test_metrics_average_over_every_minibatch():
    torch.manual_seed(0)
    model = ActorCritic()
    rollout, _ = _synced_rollout(model)
    config = PPOConfig(rollout_steps=8, minibatch_size=8, epochs=2, learning_rate=1e-9)
    returns = rollout.values + 1.0
    metrics = ppo_update(model, make_optimizer(model, config), rollout, torch.arange(32.0).view(8, 4), returns, config)
    assert metrics["value_loss"] == pytest.approx(1.0, rel=1e-3)


def test_update_skips_a_one_sample_remainder():
    torch.manual_seed(0)
    model = ActorCritic()
    rollout = Rollout(5, 1, OBSERVATION_SHAPES, torch.device("cpu"))
    config = PPOConfig(rollout_steps=5, minibatch_size=4, epochs=1)
    returns = torch.arange(5.0).view(5, 1)
    metrics = ppo_update(model, make_optimizer(model, config), rollout, returns.clone(), returns, config)
    assert math.isfinite(metrics["value_loss"])


def test_update_runs_a_two_sample_remainder():
    torch.manual_seed(0)
    model = ActorCritic()
    before = model.actor.weight.detach().clone()
    rollout = Rollout(2, 1, OBSERVATION_SHAPES, torch.device("cpu"))
    config = PPOConfig(rollout_steps=2, minibatch_size=2, epochs=1, learning_rate=1e-2)
    returns = torch.arange(2.0).view(2, 1)
    ppo_update(model, make_optimizer(model, config), rollout, returns.clone(), returns, config)
    assert not torch.equal(before, model.actor.weight)


def _explained(values, returns) -> float:
    return explained_variance(torch.tensor(values), torch.tensor(returns))


def test_explained_variance_of_half_values_is_three_quarters():
    assert _explained([0.0, 0.5, 1.0, 1.5], [0.0, 1.0, 2.0, 3.0]) == pytest.approx(0.75)


def test_explained_variance_is_nan_for_constant_returns():
    assert math.isnan(_explained([0.0, 1.0], [2.0, 2.0]))


def test_explained_variance_is_near_zero_for_uninformative_values():
    assert _explained([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0, abs=1e-6)


def test_collect_records_where_each_step_happened():
    rollout, _ = _collected()
    assert rollout.map_ids[0].tolist() == [3, 4]


def test_collect_records_battle_state_before_acting():
    rollout, _ = _collected()
    assert rollout.in_battle[0].tolist() == [False, True]


def test_collect_records_policy_entropy():
    rollout, _ = _collected()
    assert rollout.entropies[0, 0].item() == pytest.approx(math.log(7))


def _placed_rollout() -> Rollout:
    rollout = Rollout(2, 2, OBSERVATION_SHAPES, torch.device("cpu"))
    rollout.map_ids[:] = torch.tensor([[3, 4], [3, 4]], dtype=torch.uint8)
    rollout.in_battle[:] = torch.tensor([[False, True], [False, True]])
    rollout.entropies[:] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    return rollout


def test_place_totals_sum_steps_entropy_and_absolute_value_error_per_map():
    advantages = torch.tensor([[1.0, 5.0], [-1.0, 5.0]])
    totals = place_totals(_placed_rollout(), advantages)
    assert (totals[3].steps, totals[3].entropy, totals[3].value_error) == (2, pytest.approx(4.0), pytest.approx(2.0))


def test_learning_by_place_splits_battle_from_overworld():
    advantages = torch.tensor([[1.0, 5.0], [-1.0, 5.0]])
    metrics = learning_by_place(_placed_rollout(), advantages)
    assert (metrics["entropy_battle"], metrics["entropy_overworld"]) == (pytest.approx(3.0), pytest.approx(2.0))


def test_learning_by_place_omits_a_state_with_no_steps():
    rollout = _placed_rollout()
    rollout.in_battle[:] = False
    assert "entropy_battle" not in learning_by_place(rollout, torch.zeros(2, 2))
