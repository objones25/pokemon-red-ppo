"""The fixed evaluator: every checkpoint is judged the same way.

Each env plays one episode from the start state with sampled actions. Sampling matters: the emulator is deterministic,
so a deterministic policy would play one episode many times. Only each env's first episode counts, because counting the
first N to finish would favor short episodes, and events extend an episode's budget.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from torch import nn

from pokered.checkpoints import CheckpointStore
from pokered.config import EnvConfig
from pokered.env import PokemonRedEnv
from pokered.frame import EVENT_FLAG_BYTES
from pokered.model import ActorCritic
from pokered.negative_space import expect, require, unreachable
from pokered.observations import OBSERVATION_SHAPES
from pokered.ppo import Envs
from pokered.train import pick_device
from pokered.vec import VecEnv

log = logging.getLogger(__name__)

_BITS_PER_BYTE = 8


def evaluate(model: nn.Module, envs: Envs, *, device: torch.device, seed: int, max_steps: int) -> list[dict[str, float]]:
    """Each env's first episode. Uses its own random stream, so evaluating never changes a training run."""
    require(max_steps >= 1, f"max_steps must be positive, got {max_steps}")
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    first_episode: dict[int, dict[str, float]] = {}
    try:
        for _ in range(max_steps):
            with torch.inference_mode():
                logits, _ = model({key: torch.from_numpy(value).to(device) for key, value in envs.obs.items()})
            actions = torch.multinomial(logits.float().softmax(1).cpu(), 1, generator=generator).squeeze(1)
            result = envs.step(actions.numpy())
            finished = np.nonzero(result.terminated | result.truncated)[0]
            for slot, episode in zip(finished.tolist(), result.episodes, strict=True):
                first_episode.setdefault(slot, episode)
            if len(first_episode) == envs.num_envs:
                return [first_episode[slot] for slot in sorted(first_episode)]
    finally:
        model.train(was_training)
    unreachable(f"{envs.num_envs - len(first_episode)} of {envs.num_envs} envs did not finish within {max_steps} steps")


def summarize(episodes: Sequence[dict[str, float]]) -> dict[str, Any]:
    """Mean of each statistic over the episodes that have a value; None when none do, since JSON has no NaN."""
    require(episodes, "no episodes to summarize")
    report: dict[str, Any] = {"episodes": len(episodes)}
    for name in episodes[0]:
        values = [episode[name] for episode in episodes if math.isfinite(episode[name])]
        report[name] = sum(values) / len(values) if values else None
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write strict JSON atomically, so a pod stopped mid-write can't leave half a report."""
    require(report, "refusing to write an empty report")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(report, handle, allow_nan=False, indent=2, sort_keys=True)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class EvalConfig:
    rom: Path
    checkpoints: Path
    output: Path
    workers: int = 4
    envs_per_worker: int = 8
    seed: int = 0
    device: str = "auto"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = tyro.cli(EvalConfig)
    payload = expect(CheckpointStore(config.checkpoints).load_latest(), f"no intact checkpoint in {config.checkpoints}")
    stored = payload["config"]["env"]
    env_config = EnvConfig(**{**stored, "rom": config.rom, "state": Path(stored["state"]), "stagger_first_episode": False})
    device = pick_device(config.device)
    model = ActorCritic()
    model.load_state_dict(payload["model"])
    model.to(device)
    envs = VecEnv(
        partial(PokemonRedEnv, env_config),
        OBSERVATION_SHAPES,
        workers=config.workers,
        envs_per_worker=config.envs_per_worker,
        seed=config.seed,
    )
    try:
        envs.reset()
        longest = env_config.base_budget + env_config.budget_per_event * EVENT_FLAG_BYTES * _BITS_PER_BYTE
        report = summarize(evaluate(model, envs, device=device, seed=config.seed, max_steps=longest))
    finally:
        envs.close()
    report["env_steps"] = payload["env_steps"]
    write_report(config.output, report)
    log.info("wrote %s: %s", config.output, {k: v for k, v in report.items() if k.startswith("milestone_") and v})


if __name__ == "__main__":
    main()
