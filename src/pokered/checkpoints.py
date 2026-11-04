"""Atomic checkpoints on the network volume, with a fallback past damaged files.

A checkpoint becomes visible only through ``os.replace`` of a fully written, fsynced temp file, so a pod killed
mid-write leaves the previous one as latest. Should a final file still be damaged, its zip CRCs reveal it before
loading, and it is renamed aside so a resume falls back to the next one.
"""

from __future__ import annotations

import logging
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Final

import torch

from pokered.negative_space import require

log = logging.getLogger(__name__)

_NAME: Final = re.compile(r"^checkpoint_(\d{12})\.pt$")
_DIR_FLAGS: Final = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
# Settings that may differ on resume: run length and bookkeeping, and what depends on the pod. A new pod can have a
# different CPU limit, so it can have a different worker count; that changes the batch size, but stopping there would
# delete the pod and strand the run.
_RESUME_MAY_CHANGE: Final = frozenset(
    {
        "total_steps",
        "run_dir",
        "wandb_mode",
        "checkpoint_every_steps",
        "keep_checkpoints",
        "device",
        "torch_threads",
        "terminate_pod_on_exit",
        "pin_workers",
        "numa_node",
        "workers",
    }
)


class ConfigChanged(ValueError):
    """A resume asked for different training settings from those in the checkpoint."""


def _fsync(path: Path, flags: int) -> None:  # nsp: ignore NSP001 OS calls raise on failure
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _damage(path: Path) -> str | None:
    require(path.is_file(), f"{path} was listed but is not a file")
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
    except zipfile.BadZipFile as error:
        return f"BadZipFile: {error}"
    return None if bad_member is None else f"CRC mismatch in {bad_member}"


class CheckpointStore:
    def __init__(self, directory: Path, *, keep: int = 3) -> None:
        require(keep >= 1, f"must keep at least one checkpoint, got {keep}")
        self._directory = directory
        self._keep = keep

    def _existing(self) -> list[tuple[int, Path]]:  # nsp: ignore NSP001 a directory listing filtered by exact name
        """(env_steps, path), most steps first."""
        if not self._directory.is_dir():
            return []
        found = [(int(match[1]), entry) for entry in self._directory.iterdir() if (match := _NAME.match(entry.name))]
        return sorted(found, reverse=True)

    def save(self, payload: dict[str, Any], *, env_steps: int) -> Path:
        require(0 <= env_steps < 10**12, f"env_steps out of range: {env_steps}")
        require(payload.get("env_steps") == env_steps, "the file name and the payload must agree on env_steps")
        self._directory.mkdir(parents=True, exist_ok=True)
        final = self._directory / f"checkpoint_{env_steps:012d}.pt"
        temporary = final.with_name(f".{final.name}.tmp")
        try:
            torch.save(payload, temporary)
            _fsync(temporary, os.O_RDONLY)
            os.replace(temporary, final)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        _fsync(self._directory, _DIR_FLAGS)
        for _, old in self._existing()[self._keep :]:
            old.unlink()
        require(len(self._existing()) <= self._keep, f"more than {self._keep} checkpoints remain after pruning")
        return final

    def load_latest(self) -> dict[str, Any] | None:
        """The newest intact checkpoint on the CPU, or None. Damaged files are renamed ``<name>.corrupt``."""
        for steps, path in self._existing():
            damage = _damage(path)
            if damage is None:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                require(payload["env_steps"] == steps, f"{path} holds a payload for {payload['env_steps']} steps")
                return payload
            log.warning("setting aside damaged checkpoint %s: %s", path, damage)
            os.replace(path, path.with_name(path.name + ".corrupt"))
        return None

    @staticmethod
    def require_same_config(stored: dict[str, Any], current: dict[str, Any], prefix: str = "") -> None:
        """Raise ConfigChanged naming every setting that would change what is being trained."""
        for key in sorted(stored.keys() | current.keys()):
            if key in _RESUME_MAY_CHANGE:
                continue
            old, new = stored.get(key), current.get(key)
            if isinstance(old, dict) and isinstance(new, dict):
                CheckpointStore.require_same_config(old, new, f"{prefix}{key}.")
            elif old != new:
                raise ConfigChanged(f"{prefix}{key} was {old!r} in the checkpoint and is {new!r} now; use a new run_name")
