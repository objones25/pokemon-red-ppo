"""Atomic checkpoints with a fallback past damaged files."""

from __future__ import annotations

import pickle

import pytest
import torch

from pokered.checkpoints import CheckpointStore, ConfigChanged
from pokered.negative_space import CheckFailed, expect


def _latest(store: CheckpointStore) -> dict:
    return expect(store.load_latest(), "a checkpoint should exist here")


def _payload(step: int) -> dict:
    return {"env_steps": step, "weights": torch.tensor([float(step)]), "config": {"gamma": 0.9}}


def test_latest_is_none_when_empty(tmp_path):
    assert CheckpointStore(tmp_path).load_latest() is None


def test_save_then_load_round_trips(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save(_payload(10), env_steps=10)
    assert _latest(store)["env_steps"] == 10


def test_latest_means_most_steps_not_newest_file(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save(_payload(20), env_steps=20)
    store.save(_payload(10), env_steps=10)
    assert _latest(store)["env_steps"] == 20


def test_save_leaves_no_temporary_file(tmp_path):
    CheckpointStore(tmp_path).save(_payload(10), env_steps=10)
    assert [p.name for p in tmp_path.iterdir()] == ["checkpoint_000000000010.pt"]


def test_failed_write_leaves_previous_checkpoint_as_latest(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path)
    store.save(_payload(10), env_steps=10)

    def fail(_payload, _path):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        store.save(_payload(20), env_steps=20)
    monkeypatch.undo()
    assert _latest(store)["env_steps"] == 10


def test_truncated_latest_falls_back_to_older_checkpoint(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save(_payload(10), env_steps=10)
    newest = store.save(_payload(20), env_steps=20)
    newest.write_bytes(newest.read_bytes()[:100])
    assert _latest(store)["env_steps"] == 10


def test_damaged_checkpoint_is_set_aside(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save(_payload(10), env_steps=10)
    newest = store.save(_payload(20), env_steps=20)
    newest.write_bytes(newest.read_bytes()[:100])
    store.load_latest()
    assert newest.with_name(newest.name + ".corrupt").exists()


def test_flipped_byte_in_tensor_data_is_detected(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save(_payload(10), env_steps=10)
    newest = store.save({"env_steps": 20, "weights": torch.zeros(1000)}, env_steps=20)
    data = bytearray(newest.read_bytes())
    data[len(data) // 2] ^= 0xFF
    newest.write_bytes(bytes(data))
    assert _latest(store)["env_steps"] == 10


def _save_steps(store: CheckpointStore, *steps: int) -> None:
    for step in steps:
        store.save(_payload(step), env_steps=step)


def test_only_recent_checkpoints_are_kept(tmp_path):
    store = CheckpointStore(tmp_path, keep=2)
    _save_steps(store, 10, 20, 30)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint_000000000020.pt", "checkpoint_000000000030.pt"]


def test_checkpoint_loads_with_weights_only(tmp_path):
    path = CheckpointStore(tmp_path).save(_payload(10), env_steps=10)
    assert torch.load(path, weights_only=True)["weights"].item() == pytest.approx(10.0)


def test_config_changed_error_names_the_field():
    with pytest.raises(ConfigChanged, match="gamma"):
        CheckpointStore.require_same_config({"gamma": 0.9, "total_steps": 1}, {"gamma": 0.8, "total_steps": 1})


def test_total_steps_may_change_on_resume():
    assert CheckpointStore.require_same_config({"gamma": 0.9, "total_steps": 1}, {"gamma": 0.9, "total_steps": 2}) is None


@pytest.mark.parametrize("steps", [-1, 10**12])
def test_save_refuses_steps_outside_the_file_name(tmp_path, steps):
    with pytest.raises(CheckFailed, match="env_steps"):
        CheckpointStore(tmp_path).save({"env_steps": steps}, env_steps=steps)


@pytest.mark.parametrize("steps", [0, 10**12 - 1])
def test_save_accepts_steps_at_the_limits(tmp_path, steps):
    assert CheckpointStore(tmp_path).save({"env_steps": steps}, env_steps=steps).is_file()


def test_save_refuses_a_payload_for_other_steps(tmp_path):
    with pytest.raises(CheckFailed, match="agree"):
        CheckpointStore(tmp_path).save({"env_steps": 1}, env_steps=2)


def test_save_works_when_the_directory_already_exists(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save(_payload(1), env_steps=1)
    assert store.save(_payload(2), env_steps=2).is_file()


class NotAWeight:
    """Anything arbitrary a pickle could carry; weights-only loading must refuse it."""


def test_load_refuses_arbitrary_pickled_objects(tmp_path):
    store = CheckpointStore(tmp_path)
    torch.save({"env_steps": 1, "evil": NotAWeight()}, tmp_path / "checkpoint_000000000001.pt")
    with pytest.raises(pickle.UnpicklingError, match=r"[Ww]eights"):
        store.load_latest()


def test_config_change_from_nested_to_scalar_is_reported():
    with pytest.raises(ConfigChanged, match="ppo"):
        CheckpointStore.require_same_config({"ppo": {"gamma": 1}}, {"ppo": 1})


def test_nested_config_change_names_the_full_path():
    with pytest.raises(ConfigChanged, match=r"ppo\.gamma"):
        CheckpointStore.require_same_config({"ppo": {"gamma": 1}}, {"ppo": {"gamma": 2}})


def test_save_creates_missing_parent_directories(tmp_path):
    assert CheckpointStore(tmp_path / "run" / "checkpoints").save(_payload(1), env_steps=1).is_file()
