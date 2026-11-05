"""The actor-critic takes uint8 observations and emits raw logits."""

from __future__ import annotations

import pytest
import torch

from pokered.buttons import BUTTONS
from pokered.model import ActorCritic, feature_health
from pokered.observations import OBSERVATION_SHAPES


def _batch(size: int = 4) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        key: torch.randint(0, 256, (size, *shape), dtype=torch.uint8, generator=generator)
        for key, shape in OBSERVATION_SHAPES.items()
    }


def test_outputs_one_logit_per_button():
    assert ActorCritic()(_batch())[0].shape == (4, len(BUTTONS))


def test_outputs_one_value_per_observation():
    assert ActorCritic()(_batch())[1].shape == (4,)


def test_logits_are_not_a_probability_distribution():
    assert not torch.allclose(ActorCritic()(_batch())[0].sum(1), torch.ones(4))


def test_parameter_count_is_near_the_papers_two_million():
    count = sum(p.numel() for p in ActorCritic().parameters())
    assert 1_500_000 < count < 2_500_000


def test_event_bits_change_the_output():
    model = ActorCritic()
    batch = _batch(1)
    batch["events"] = torch.zeros_like(batch["events"])
    before = model(batch)[1]
    batch["events"][0, 0] = 1
    assert model(batch)[1].item() != pytest.approx(before.item(), abs=1e-7)


def test_event_bits_unpack_lowest_first():
    bits = ActorCritic.unpack_bits(torch.tensor([[0b00000101]], dtype=torch.uint8))
    assert bits[0].tolist() == [1, 0, 1, 0, 0, 0, 0, 0]


def test_every_parameter_receives_a_gradient():
    model = ActorCritic()
    logits, value = model(_batch())
    (logits.sum() + value.sum()).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_initial_policy_is_near_uniform():
    logits, _ = ActorCritic()(_batch())
    probs = logits.softmax(1)
    assert probs.max().item() < 0.2


def test_features_are_what_the_heads_read():
    model = ActorCritic()
    batch = _batch()
    assert torch.equal(model.actor(model.features(batch)), model(batch)[0])


def test_feature_health_finds_a_unit_that_never_fires():
    features = torch.eye(4)
    features[:, 3] = 0.0
    assert feature_health(features)["dead_fraction"] == pytest.approx(0.25)


def test_feature_health_rank_of_one_repeated_row_is_one():
    assert feature_health(torch.ones(8, 4))["feature_rank"] == pytest.approx(1.0)


def test_feature_health_rank_of_independent_rows_is_full():
    assert feature_health(torch.eye(4))["feature_rank"] == pytest.approx(4.0)


def test_feature_health_reports_the_mean_row_norm():
    assert feature_health(torch.eye(4) * 3.0)["feature_norm"] == pytest.approx(3.0)
