"""W&B logging under one fixed run id, so a restart continues the same run."""

from __future__ import annotations

import pytest
import wandb
from wandb import errors as wandb_errors

from pokered.wandb_logger import WandbLogger


def test_two_sessions_share_a_run_id(tmp_path):
    first = WandbLogger("run-a", project="test", config={}, mode="offline", directory=tmp_path)
    first_id = first.run_id
    first.finish()
    second = WandbLogger("run-a", project="test", config={}, mode="offline", directory=tmp_path)
    second_id = second.run_id
    second.finish()
    assert (first_id, second_id) == ("run-a", "run-a")


def _init_failing_online(real_init, calls: list[str]):
    def init(**kwargs):
        calls.append(kwargs["mode"])
        if kwargs["mode"] == "online":
            raise wandb_errors.CommError("no network")
        return real_init(**kwargs)

    return init


def test_online_failure_falls_back_to_offline(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(wandb, "init", _init_failing_online(wandb.init, calls))
    logger = WandbLogger("run-b", project="test", config={}, mode="online", directory=tmp_path)
    logger.finish()
    assert calls == ["online", "offline"]


def test_usage_error_is_not_swallowed(tmp_path, monkeypatch):
    def bad_init(**_kwargs):
        raise wandb_errors.UsageError("bad argument")

    monkeypatch.setattr(wandb, "init", bad_init)
    with pytest.raises(wandb_errors.UsageError, match="bad argument"):
        WandbLogger("run-c", project="test", config={}, mode="online", directory=tmp_path)
