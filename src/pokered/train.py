"""The training loop and its composition root."""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from types import FrameType
from typing import Any, Final, Protocol

import httpx
import torch
import tyro

from pokered.affinity import pin_process, plan_for_this_machine
from pokered.checkpoints import CheckpointStore
from pokered.config import TrainConfig
from pokered.env import PokemonRedEnv
from pokered.model import ActorCritic, feature_health
from pokered.negative_space import check_finite, require
from pokered.observations import OBSERVATION_SHAPES
from pokered.ppo import (
    Envs,
    Rollout,
    collect_rollout,
    compute_gae,
    learning_by_place,
    make_optimizer,
    place_totals,
    ppo_update,
)
from pokered.report import PlaceTotals, build_report, save_report
from pokered.runpod import TerminateFailed, own_terminator
from pokered.telemetry import EpisodeReport
from pokered.vec import VecEnv
from pokered.wandb_logger import Logger, WandbLogger

log = logging.getLogger(__name__)

_HEALTH_ROWS: Final[int] = 512


class PodStopping(Exception):
    """SIGTERM arrived, usually Runpod stopping the pod. Raised so checkpoints save and cleanup runs."""


class Terminator(Protocol):
    def terminate(self) -> None: ...


def run_guarded(work: Callable[[], None], terminator: Terminator | None) -> None:  # nsp: ignore NSP001 control flow only; each path is tested
    """Run ``work``, then delete the pod however it ended, so a crash can't leave one billing.

    Ctrl-C at a terminal keeps the pod: someone is there and may want it. If training and termination both fail, the
    training error propagates and the termination failure is logged.
    """

    def on_sigterm(_signal: int, _frame: FrameType | None) -> None:  # nsp: ignore NSP001 a signal handler that only raises
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # A repeated SIGTERM must not interrupt the cleanup.
        raise PodStopping("received SIGTERM")

    previous = signal.signal(signal.SIGTERM, on_sigterm)
    try:
        try:
            work()
        except KeyboardInterrupt:
            raise
        except BaseException:
            if terminator is not None:
                try:
                    terminator.terminate()
                except TerminateFailed:
                    log.exception("THE POD MAY STILL BE BILLING: delete it by hand")
            raise
        if terminator is not None:
            terminator.terminate()
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextmanager
def holding_sigterm() -> Iterator[None]:  # nsp: ignore NSP001 control flow only; each path is tested
    """Hold SIGTERM until the block ends, then deliver it: an update or a save must never stop halfway."""
    received: list[int] = []
    previous = signal.signal(signal.SIGTERM, lambda number, _frame: received.append(number))
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
    if received:
        signal.raise_signal(signal.SIGTERM)


def pick_device(requested: str = "auto") -> torch.device:  # nsp: ignore NSP001 torch.device rejects bad names
    if requested != "auto":
        return torch.device(requested)
    return torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")


def _device_rng(device: torch.device) -> torch.Tensor | None:  # nsp: ignore NSP001 None is the legal CPU answer
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def _set_device_rng(device: torch.device, state: torch.Tensor | None) -> None:  # nsp: ignore NSP001 torch validates the state
    if state is None:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state)


def config_dict(config: TrainConfig) -> dict[str, Any]:
    """The config as plain values, which a weights-only checkpoint can hold and W&B can display."""
    return json.loads(json.dumps(asdict(config), default=str))


def network_health(model: ActorCritic, rollout: Rollout) -> dict[str, float]:
    """Measured on evenly spaced rollout rows, so that taking the measurement uses no random numbers."""
    rows = rollout.steps * rollout.envs
    require(rows >= 1, f"a rollout needs at least one row, got {rows}")
    index = torch.linspace(0, rows - 1, min(_HEALTH_ROWS, rows), device=rollout.actions.device).long()
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            features = model.features({key: value.flatten(0, 1)[index] for key, value in rollout.obs.items()})
    finally:
        model.train(was_training)
    return {f"net/{name}": value for name, value in feature_health(features).items()}


def _merge_places(window: dict[int, PlaceTotals], update: dict[int, PlaceTotals]) -> None:
    for map_id, place in update.items():
        old = window.get(map_id, PlaceTotals(0, 0.0, 0.0))
        window[map_id] = PlaceTotals(old.steps + place.steps, old.entropy + place.entropy, old.value_error + place.value_error)
    require(all(place.steps > 0 for place in window.values()), "a map with no steps has no place in the totals")


def train(config: TrainConfig, model: ActorCritic, envs: Envs, logger: Logger, store: CheckpointStore, device: torch.device) -> None:
    """Train to ``config.total_steps``, resuming from the newest intact checkpoint in ``store``."""
    require(envs.num_envs == config.num_envs, f"{envs.num_envs} envs given, the config says {config.num_envs}")
    model.to(device)
    optimizer = make_optimizer(model, config.ppo)
    current = config_dict(config)
    env_steps = 0
    saved = store.load_latest()
    if saved is not None:
        CheckpointStore.require_same_config(saved["config"], current)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        env_steps = saved["env_steps"]
        torch.set_rng_state(saved["rng_cpu"])
        _set_device_rng(device, saved["rng_device"])
        log.info("resumed from %d env steps", env_steps)

    def save() -> None:
        store.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "env_steps": env_steps,
                "config": current,
                "rng_cpu": torch.get_rng_state(),
                "rng_device": _device_rng(device),
            },
            env_steps=env_steps,
        )

    rollout = Rollout(config.ppo.rollout_steps, config.num_envs, OBSERVATION_SHAPES, device)
    updates = math.ceil(max(config.total_steps - env_steps, 0) / config.batch_size)
    window: list[EpisodeReport] = []
    places: dict[int, PlaceTotals] = {}
    try:
        for _ in range(updates):
            started = time.perf_counter()
            window.extend(collect_rollout(model, envs, rollout, device))
            collected = time.perf_counter()
            with holding_sigterm():
                advantages, returns = compute_gae(
                    rewards=rollout.rewards,
                    values=rollout.values,
                    next_values=rollout.next_values,
                    terminated=rollout.terminated,
                    truncated=rollout.truncated,
                    gamma=config.ppo.gamma,
                    lam=config.ppo.lam,
                )
                _merge_places(places, place_totals(rollout, advantages))
                by_place = learning_by_place(rollout, advantages)
                health = ppo_update(model, optimizer, rollout, advantages, returns, config.ppo)
                finished = time.perf_counter()
                previous_steps = env_steps
                env_steps += config.batch_size
                check_finite(rollout.rewards, "rollout rewards")
                metrics = {
                    **{f"ppo/{name}": value for name, value in {**health, **by_place}.items()},
                    "perf/steps_per_second": config.batch_size / (finished - started),
                    "perf/update_seconds": finished - collected,
                }
                report_due = env_steps // config.report_every_steps > previous_steps // config.report_every_steps
                if report_due or env_steps >= config.total_steps:
                    metrics.update(network_health(model, rollout))
                    if window:
                        report = build_report(window, places=places)
                        logger.log_report(report, step=env_steps)
                        save_report(report, config.run_dir, step=env_steps)
                    window, places = [], {}
                logger.log(metrics, step=env_steps)
                crossed = env_steps // config.checkpoint_every_steps > previous_steps // config.checkpoint_every_steps
                if crossed or env_steps >= config.total_steps:
                    save()
    except PodStopping:
        if env_steps > 0:
            save()  # Weights only change inside the held block, so this is the state at the end of the last update.
        raise
    require(env_steps >= config.total_steps, f"stopped at {env_steps} of {config.total_steps} steps")


def main() -> None:  # nsp: ignore NSP001 the composition root; train and the constructors check their inputs
    """Composition root: the one place concrete classes are chosen."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = tyro.cli(TrainConfig)
    # Built first, so a missing key or pod id fails before any money is spent training.
    terminator = own_terminator(os.environ, httpx.Client(timeout=30.0)) if config.terminate_pod_on_exit else None
    run_guarded(partial(_run, config), terminator)


def _run(config: TrainConfig) -> None:  # nsp: ignore NSP001 wiring; train and the constructors check their inputs
    torch.set_num_threads(config.torch_threads)
    torch.manual_seed(config.seed)
    device = pick_device(config.device)
    plan = plan_for_this_machine(config.workers, config.numa_node) if config.pin_workers else None
    envs = VecEnv(
        partial(PokemonRedEnv, config.env),
        OBSERVATION_SHAPES,
        workers=config.workers,
        envs_per_worker=config.envs_per_worker,
        seed=config.seed,
        worker_cores=None if plan is None else plan.workers,
    )
    if plan is not None:
        pin_process(plan.main)  # After starting the workers, which pin themselves.
    logger = None
    try:
        logger = WandbLogger(
            config.run_name,
            project=config.wandb_project,
            config=config_dict(config),
            mode=config.wandb_mode,
            directory=config.run_dir,
        )
        envs.reset()
        store = CheckpointStore(config.run_dir / "checkpoints", keep=config.keep_checkpoints)
        train(config, ActorCritic(), envs, logger, store, device)
    finally:
        if logger is not None:
            logger.finish()
        envs.close()


if __name__ == "__main__":
    main()
