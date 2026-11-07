"""One source of truth for run settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pokered.start_state import START_STATE

WandbMode = Literal["online", "offline", "disabled"]


@dataclass(frozen=True, slots=True)
class EnvConfig:
    rom: Path
    state: Path = START_STATE
    battle_animations: bool = False
    fast_mode: bool = False
    tile_weight: float = 0.005
    event_weight: float = 2.0
    badge_weight: float = 5.0
    level_weight: float = 0.5
    species_weight: float = 1.0
    heal_weight: float = 5.0
    move_weight: float = 0.01
    base_budget: int = 10_240
    budget_per_event: int = 2_048
    stagger_first_episode: bool = True
    stall_steps: int = 10_240  # Steps with no reward but exploration before the episode ends (terminated).


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """Pleines et al. 2025 Table I, except T, which the batch arithmetic in TrainConfig states."""

    gamma: float = 0.997
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    rollout_steps: int = 2_048
    minibatch_size: int = 8_192
    epochs: int = 3


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """A whole run. Values arrive from the command line, so bad ones raise ValueError rather than crash a check."""

    env: EnvConfig
    ppo: PPOConfig = PPOConfig()
    workers: int = 12
    envs_per_worker: int = 8
    total_steps: int = 400_000_000
    seed: int = 0
    run_dir: Path = Path("runs/default")
    run_name: str = "default"
    wandb_project: str = "pokemon-red-ppo"
    wandb_mode: WandbMode = "online"
    checkpoint_every_steps: int = 5_000_000
    report_every_steps: int = 5_000_000
    keep_checkpoints: int = 3
    device: str = "auto"
    torch_threads: int = 4
    terminate_pod_on_exit: bool = False
    pin_workers: bool = False
    numa_node: int = 0
    # A checkpoint directory a new run starts from: its newest weights and optimizer state, with steps counted from 0.
    # A checkpoint in this run's own directory still wins, so a restarted pod resumes.
    init_from: Path | None = None
    # False collects and reports exactly as training does but never changes the weights: a frozen-policy control.
    update_policy: bool = True

    def __post_init__(self) -> None:
        for name in ("workers", "envs_per_worker", "total_steps", "checkpoint_every_steps", "report_every_steps", "keep_checkpoints"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.ppo.minibatch_size > self.batch_size:
            raise ValueError(f"minibatch_size {self.ppo.minibatch_size} exceeds batch {self.batch_size}")

    @property
    def num_envs(self) -> int:
        return self.workers * self.envs_per_worker

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.ppo.rollout_steps
