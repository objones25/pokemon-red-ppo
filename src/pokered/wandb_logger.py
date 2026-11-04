"""W&B logging under one fixed run id, so a restart on a new pod continues the same run."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final, Protocol

import wandb
from wandb.errors import Error as WandbError
from wandb.errors import UsageError

from pokered.config import WandbMode
from pokered.negative_space import require

log = logging.getLogger(__name__)

_OUTSIDE_WORLD_ERRORS: Final = (WandbError, TimeoutError)


class Logger(Protocol):
    def log(self, metrics: dict[str, float], *, step: int) -> None: ...
    def finish(self) -> None: ...


class WandbLogger:
    """One W&B outage shouldn't stop an unattended run: online failures fall back to offline with the same id."""

    def __init__(self, run_id: str, *, project: str, config: dict[str, Any], mode: WandbMode, directory: Path) -> None:
        require(run_id, "a W&B run needs a fixed id so that it can resume")
        directory.mkdir(parents=True, exist_ok=True)
        options: dict[str, Any] = {"id": run_id, "resume": "allow", "project": project, "config": config, "dir": str(directory)}
        try:
            self._run = wandb.init(mode=mode, **options)
        except UsageError:
            raise  # A bad argument is a bug, not an outage.
        except _OUTSIDE_WORLD_ERRORS as error:
            if mode != "online":
                raise
            log.warning("wandb.init failed for %s (%s); continuing offline. Upload later with `wandb sync`.", run_id, error)
            wandb.teardown()  # The failed init keeps the id claimed until the service shuts down.
            self._run = wandb.init(mode="offline", **options)

    @property
    def run_id(self) -> str:
        return self._run.id

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        require(step >= 0, f"logging step cannot be negative, got {step}")
        self._run.log(metrics, step=step)

    def finish(self) -> None:
        self._run.finish()
