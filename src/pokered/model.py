"""Actor-critic: two Nature CNNs and a state MLP, about 2M parameters as in Pleines et al. 2025."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

from pokered.buttons import BUTTONS
from pokered.negative_space import check_finite, expect, require
from pokered.observations import OBSERVATION_SHAPES

_FEATURES: Final[int] = 512
_STATE_FEATURES: Final[int] = 256
_PIXEL_MAX: Final[float] = 255.0
_BITS_PER_BYTE: Final[int] = 8


def _nature_cnn(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, 32, 8, stride=4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        nn.Flatten(),
    )  # fmt: skip


def _flat_size(cnn: nn.Module, shape: tuple[int, ...]) -> int:
    with torch.no_grad():
        return int(cnn(torch.zeros(1, *shape)).shape[1])


class ActorCritic(nn.Module):
    """Takes uint8 observations, converts them on the model's device, and returns raw logits and a value."""

    def __init__(self) -> None:
        super().__init__()
        self.screens = _nature_cnn(OBSERVATION_SHAPES["screens"][0])
        self.visited = _nature_cnn(OBSERVATION_SHAPES["visited"][0])
        party_slots = OBSERVATION_SHAPES["party"][0] - 1
        state_size = party_slots + _BITS_PER_BYTE * (1 + OBSERVATION_SHAPES["events"][0])
        self.state = nn.Sequential(nn.Linear(state_size, _STATE_FEATURES), nn.ReLU())
        body_size = (
            _flat_size(self.screens, OBSERVATION_SHAPES["screens"])
            + _flat_size(self.visited, OBSERVATION_SHAPES["visited"])
            + _STATE_FEATURES
        )
        self.body = nn.Sequential(nn.Linear(body_size, _FEATURES), nn.ReLU())
        self.actor = nn.Linear(_FEATURES, len(BUTTONS))
        self.critic = nn.Linear(_FEATURES, 1)
        self.apply(_orthogonal)
        nn.init.orthogonal_(self.actor.weight, 0.01)  # A near-uniform first policy, per the PPO implementation details.
        nn.init.orthogonal_(self.critic.weight, 1.0)

    @staticmethod
    def unpack_bits(data: torch.Tensor) -> torch.Tensor:
        """(B, n) uint8 to (B, 8n) float, lowest bit first, matching the game's event-flag numbering."""
        require(data.dtype == torch.uint8, f"bits unpack from uint8, got {data.dtype}")
        shifts = torch.arange(_BITS_PER_BYTE, dtype=torch.uint8, device=data.device)
        return ((data.unsqueeze(-1) >> shifts) & 1).flatten(1).float()

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        for key, value in obs.items():
            require(value.dtype == torch.uint8, f"{key} must stay uint8 until here, got {value.dtype}")
        party = obs["party"]
        state = torch.cat(
            [party[:, :-1].float() / _PIXEL_MAX, self.unpack_bits(party[:, -1:]), self.unpack_bits(obs["events"])], dim=1
        )
        features = self.body(
            torch.cat(
                [
                    self.screens(obs["screens"].float() / _PIXEL_MAX),
                    self.visited(obs["visited"].float() / _PIXEL_MAX),
                    self.state(state),
                ],
                dim=1,
            )
        )
        logits, values = self.actor(features), self.critic(features).squeeze(1)
        require(logits.shape[0] == values.shape[0], "logits and values must have one row per observation")
        check_finite(values, "values")
        return logits, values


def _orthogonal(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d | nn.Linear):
        nn.init.orthogonal_(module.weight, nn.init.calculate_gain("relu"))
        nn.init.zeros_(expect(module.bias, f"{type(module).__name__} was built without a bias"))
