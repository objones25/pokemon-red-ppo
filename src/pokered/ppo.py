"""PPO: rollout storage on the device, GAE with correct time-limit bootstrapping, and the clipped update."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, NamedTuple, Protocol

import numpy as np
import torch
from torch import nn

from pokered.config import PPOConfig
from pokered.negative_space import check_finite, require
from pokered.report import PlaceTotals
from pokered.telemetry import EpisodeReport
from pokered.vec import StepResult

_ADVANTAGE_EPS: Final[float] = 1e-8


class Envs(Protocol):
    """What a rollout needs from a batch of environments."""

    @property
    def num_envs(self) -> int: ...
    @property
    def obs(self) -> Mapping[str, np.ndarray]: ...
    @property
    def map_ids(self) -> np.ndarray: ...
    @property
    def in_battle(self) -> np.ndarray: ...

    def step(self, actions: np.ndarray, /) -> StepResult: ...


class MinibatchLosses(NamedTuple):
    policy: torch.Tensor
    value: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advantages and value targets for (T, N) tensors.

    ``next_values[t]`` is V of the true next state, including the final state of a truncated episode. A truncation
    keeps that bootstrap but stops the trace, because the next row belongs to a new episode. A termination drops both.
    """
    require(rewards.shape == values.shape, f"rewards {rewards.shape} and values {values.shape} must match")
    require(rewards.shape == next_values.shape, f"rewards {rewards.shape} and next_values {next_values.shape} must match")
    require(0.0 <= gamma <= 1.0, f"gamma out of range: {gamma}")
    require(0.0 <= lam <= 1.0, f"lam out of range: {lam}")
    require(not (terminated * truncated).any(), "a step cannot be both terminated and truncated")
    advantages = torch.zeros_like(rewards)
    trace = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * (1.0 - terminated[t]) * next_values[t] - values[t]
        continuing = (1.0 - terminated[t]) * (1.0 - truncated[t])
        trace = delta + gamma * lam * continuing * trace
        advantages[t] = trace
    check_finite(advantages, "advantages")
    return advantages, advantages + values


class Rollout:
    """One rollout's data. Observations stay uint8, so storage costs about 20 KB per sample."""

    def __init__(self, steps: int, envs: int, shapes: Mapping[str, tuple[int, ...]], device: torch.device) -> None:
        require(steps >= 1, f"rollout steps must be positive, got {steps}")
        require(envs >= 1, f"rollout envs must be positive, got {envs}")
        self.steps, self.envs = steps, envs
        self.obs = {key: torch.zeros((steps, envs, *shape), dtype=torch.uint8, device=device) for key, shape in shapes.items()}
        self.actions = torch.zeros((steps, envs), dtype=torch.int64, device=device)
        self.log_probs = torch.zeros((steps, envs), device=device)
        self.values = torch.zeros((steps, envs), device=device)
        self.rewards = torch.zeros((steps, envs), device=device)
        self.terminated = torch.zeros((steps, envs), device=device)
        self.truncated = torch.zeros((steps, envs), device=device)
        self.next_values = torch.zeros((steps, envs), device=device)
        self.entropies = torch.zeros((steps, envs), device=device)
        self.map_ids = torch.zeros((steps, envs), dtype=torch.uint8, device=device)
        self.in_battle = torch.zeros((steps, envs), dtype=torch.bool, device=device)


def autocast(device: torch.device) -> torch.autocast:
    """bfloat16 on CUDA, which needs no gradient scaling. Autocast is left off elsewhere; MPS was measured without it."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def _to_device(arrays: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}


@torch.inference_mode()
def collect_rollout(model: nn.Module, envs: Envs, rollout: Rollout, device: torch.device) -> list[EpisodeReport]:
    """Fill ``rollout`` from ``envs``, continuing from their current observations. Returns finished episodes."""
    require(envs.num_envs == rollout.envs, f"{envs.num_envs} envs cannot fill a rollout sized for {rollout.envs}")
    episodes: list[EpisodeReport] = []
    for t in range(rollout.steps):
        obs = _to_device(envs.obs, device)
        with autocast(device):
            logits, values = model(obs)
        distribution = torch.distributions.Categorical(logits=logits.float())
        actions = distribution.sample()
        for key, value in obs.items():
            rollout.obs[key][t] = value
        rollout.actions[t] = actions
        rollout.log_probs[t] = distribution.log_prob(actions)
        rollout.values[t] = values.float()
        rollout.entropies[t] = distribution.entropy()
        rollout.map_ids[t] = torch.from_numpy(envs.map_ids).to(device)
        rollout.in_battle[t] = torch.from_numpy(envs.in_battle).to(device)
        result = envs.step(actions.cpu().numpy())
        rollout.rewards[t] = torch.from_numpy(result.rewards).to(device)
        rollout.terminated[t] = torch.from_numpy(result.terminated).to(device).float()
        truncated = torch.from_numpy(result.truncated).to(device)
        rollout.truncated[t] = truncated.float()
        rollout.next_values[t] = 0.0
        if truncated.any():
            final = {key: value[truncated] for key, value in _to_device(result.final_obs, device).items()}
            with autocast(device):
                rollout.next_values[t][truncated] = model(final)[1].float()
        episodes.extend(result.episodes)
    with autocast(device):
        last = model(_to_device(envs.obs, device))[1].float()
    open_rows = rollout.truncated[:-1] == 0
    rollout.next_values[:-1][open_rows] = rollout.values[1:][open_rows]
    final_open = rollout.truncated[-1] == 0
    rollout.next_values[-1][final_open] = last[final_open]
    check_finite(rollout.rewards, "rollout rewards")
    check_finite(rollout.next_values, "rollout next values")
    return episodes


def place_totals(rollout: Rollout, advantages: torch.Tensor) -> dict[int, PlaceTotals]:
    """Entropy and value error summed per map. A GAE advantage is the return minus the value, so it is the error."""
    require(advantages.shape == rollout.map_ids.shape, f"advantages {tuple(advantages.shape)} must match the rollout")
    errors = advantages.abs()
    totals: dict[int, PlaceTotals] = {}
    for map_id in rollout.map_ids.unique().tolist():
        here = rollout.map_ids == map_id
        totals[map_id] = PlaceTotals(int(here.sum().item()), rollout.entropies[here].sum().item(), errors[here].sum().item())
    return totals


def learning_by_place(rollout: Rollout, advantages: torch.Tensor) -> dict[str, float]:
    """Battle and overworld separately: one average can hide a policy that is sure in one and lost in the other."""
    require(advantages.shape == rollout.in_battle.shape, f"advantages {tuple(advantages.shape)} must match the rollout")
    metrics: dict[str, float] = {}
    for label, here in (("battle", rollout.in_battle), ("overworld", ~rollout.in_battle)):
        if here.any():
            metrics[f"entropy_{label}"] = rollout.entropies[here].mean().item()
            metrics[f"value_error_{label}"] = advantages[here].abs().mean().item()
    return metrics


def make_optimizer(model: nn.Module, config: PPOConfig) -> torch.optim.Optimizer:
    require(config.learning_rate > 0.0, f"learning rate must be positive, got {config.learning_rate}")
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    require(values.shape == returns.shape, f"values {values.shape} and returns {returns.shape} must match")
    variance = returns.var()
    if variance == 0:
        return float("nan")
    return float(1.0 - (returns - values).var() / variance)


def normalize_advantage(advantage: torch.Tensor) -> torch.Tensor:
    require(advantage.numel() >= 2, "normalizing needs at least two advantages")
    return (advantage - advantage.mean()) / (advantage.std() + _ADVANTAGE_EPS)


def minibatch_losses(
    *,
    log_ratio: torch.Tensor,
    advantage: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip: float,
) -> MinibatchLosses:
    """Clipped policy and value losses (Pleines et al. 2025 eq. 3 to 5), and the k3 KL estimate."""
    require(clip > 0.0, f"clip must be positive, got {clip}")
    require(log_ratio.shape == advantage.shape, "log ratios and advantages must match")
    ratio = log_ratio.exp()
    clipped_values = old_values + (values - old_values).clamp(-clip, clip)
    return MinibatchLosses(
        policy=-torch.min(advantage * ratio, advantage * ratio.clamp(1 - clip, 1 + clip)).mean(),
        value=torch.max((values - returns) ** 2, (clipped_values - returns) ** 2).mean(),
        approx_kl=((ratio - 1) - log_ratio).mean().detach(),
        clip_fraction=((ratio - 1).abs() > clip).float().mean(),
    )


def total_loss(policy: torch.Tensor, value: torch.Tensor, entropy: torch.Tensor, config: PPOConfig) -> torch.Tensor:
    loss = policy + config.vf_coef * value - config.ent_coef * entropy
    check_finite(loss, "loss")
    return loss


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
) -> dict[str, float]:
    """Run the clipped update and return health metrics, averaged over minibatches."""
    expected = (rollout.steps, rollout.envs)
    require(advantages.shape == expected, f"advantages {tuple(advantages.shape)} must be {expected}")
    require(returns.shape == expected, f"returns {tuple(returns.shape)} must be {expected}")
    total = advantages.numel()
    require(config.minibatch_size <= total, f"minibatch {config.minibatch_size} exceeds batch {total}")
    device = advantages.device
    flat_obs = {key: value.flatten(0, 1) for key, value in rollout.obs.items()}
    actions, old_log_probs = rollout.actions.flatten(), rollout.log_probs.flatten()
    old_values, flat_advantages, flat_returns = rollout.values.flatten(), advantages.flatten(), returns.flatten()
    names = ("value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")
    sums = torch.zeros(len(names), device=device)
    count = 0
    kl_max = 0.0
    before = nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    for _ in range(config.epochs):
        order = torch.randperm(total, device=device)
        for start in range(0, total, config.minibatch_size):
            index = order[start : start + config.minibatch_size]
            if index.numel() < 2:
                continue  # A one-sample remainder has no advantage spread to normalize.
            with autocast(device):
                logits, values = model({key: value[index] for key, value in flat_obs.items()})
            distribution = torch.distributions.Categorical(logits=logits.float())
            entropy = distribution.entropy().mean()
            losses = minibatch_losses(
                log_ratio=distribution.log_prob(actions[index]) - old_log_probs[index],
                advantage=normalize_advantage(flat_advantages[index]),
                values=values.float(),
                old_values=old_values[index],
                returns=flat_returns[index],
                clip=config.clip,
            )
            loss = total_loss(losses.policy, losses.value, entropy, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            check_finite(grad_norm, "gradient norm")
            optimizer.step()
            with torch.no_grad():
                sums += torch.stack([losses.value, entropy, losses.approx_kl, losses.clip_fraction, grad_norm])
                kl_max = max(kl_max, losses.approx_kl.item())
            count += 1
    require(count > 0, "the update ran no minibatches")
    metrics: dict[str, float] = dict(zip(names, (sums / count).tolist(), strict=True))
    metrics["explained_variance"] = explained_variance(old_values, flat_returns)
    metrics["approx_kl_max"] = kl_max
    after = nn.utils.parameters_to_vector(model.parameters()).detach()
    metrics["weight_norm"] = after.norm().item()
    metrics["update_ratio"] = (after - before).norm().item() / max(metrics["weight_norm"], _ADVANTAGE_EPS)
    return metrics
