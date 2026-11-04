"""W worker processes x k envs per worker, in lockstep, with observations in shared memory.

Workers never import torch. Measured on the M4 Pro, k = 8 beats k = 1 by 53% because inference batches grow while
the per-worker message count doesn't (docs/CONFLICT_RESOLUTIONS.md).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Final

import numpy as np

from pokered.affinity import pin_process
from pokered.negative_space import check_shape, require

log = logging.getLogger(__name__)

_REPLY_TIMEOUT_S: Final[float] = 300.0
_STEP: Final[str] = "step"
_RESET: Final[str] = "reset"
_CLOSE: Final[str] = "close"
_OBS: Final[str] = "obs/"
_FINAL: Final[str] = "final/"


class WorkerFailure(RuntimeError):
    """A worker raised, died, or stopped replying."""


@dataclass(frozen=True, slots=True)
class StepResult:
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    final_obs: dict[str, np.ndarray]
    episodes: list[dict[str, float]]


def _layout(shapes: Mapping[str, tuple[int, ...]], num_envs: int) -> dict[str, tuple[tuple[int, ...], type]]:
    require(num_envs >= 1, f"need at least one env, got {num_envs}")
    fields: dict[str, tuple[tuple[int, ...], type]] = {
        "actions": ((num_envs,), np.int64),
        "rewards": ((num_envs,), np.float32),
        "terminated": ((num_envs,), np.bool_),
        "truncated": ((num_envs,), np.bool_),
    }
    for key, shape in shapes.items():
        fields[_OBS + key] = ((num_envs, *shape), np.uint8)
        fields[_FINAL + key] = ((num_envs, *shape), np.uint8)
    return fields


def _views(buffer: Any, layout: Mapping[str, tuple[tuple[int, ...], type]]) -> dict[str, np.ndarray]:
    require(layout, "the shared-memory layout is empty")
    views, offset = {}, 0
    for name, (shape, dtype) in layout.items():
        views[name] = np.ndarray(shape, dtype, buffer, offset)
        offset += int(np.prod(shape)) * np.dtype(dtype).itemsize
    require(offset <= len(buffer), f"layout needs {offset} bytes, the buffer holds {len(buffer)}")
    return views


def _size(layout: Mapping[str, tuple[tuple[int, ...], type]]) -> int:
    return sum(int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in layout.values())


def _write_obs(views: Mapping[str, np.ndarray], prefix: str, slot: int, obs: Mapping[str, np.ndarray]) -> None:  # nsp: ignore NSP001 numpy assignment raises on a shape mismatch
    for key, value in obs.items():
        views[prefix + key][slot] = value


def _worker(conn: Connection, factory: Callable[[], Any], memory_name: str, layout: Any, slots: range, seed: int, core: int | None) -> None:
    require(len(slots) >= 1, "a worker needs at least one env")
    if core is not None:
        pin_process({core})
    memory = SharedMemory(name=memory_name)
    views = _views(memory.buf, layout)
    envs = []
    try:
        envs = [factory() for _ in slots]
        while (command := conn.recv()) != _CLOSE:
            episodes: list[dict[str, float]] = []
            for slot, env in zip(slots, envs, strict=True):
                if command == _RESET:
                    _write_obs(views, _OBS, slot, env.reset(seed=seed + slot)[0])
                    continue
                obs, reward, terminated, truncated, info = env.step(views["actions"][slot])
                views["rewards"][slot] = reward
                views["terminated"][slot] = terminated
                views["truncated"][slot] = truncated
                if terminated or truncated:
                    _write_obs(views, _FINAL, slot, obs)
                    episodes.append(info["episode"])
                    obs = env.reset()[0]
                _write_obs(views, _OBS, slot, obs)
            conn.send(("ok", episodes))
    except Exception:  # Reported to the parent, which raises WorkerFailure: a worker has nowhere else to report.
        conn.send(("error", traceback.format_exc()))
    finally:
        for env in envs:
            env.close()
        views.clear()
        memory.close()


class VecEnv:
    def __init__(
        self,
        factory: Callable[[], Any],
        shapes: Mapping[str, tuple[int, ...]],
        *,
        workers: int,
        envs_per_worker: int,
        seed: int,
        worker_cores: Sequence[int] | None = None,
    ) -> None:
        if workers < 1 or envs_per_worker < 1:
            raise ValueError(f"need at least one worker and env, got {workers} x {envs_per_worker}")
        require(worker_cores is None or len(worker_cores) == workers, "worker_cores needs one core per worker")
        self.num_envs = workers * envs_per_worker
        self._shapes = dict(shapes)
        layout = _layout(self._shapes, self.num_envs)
        self._memory = SharedMemory(create=True, size=_size(layout))
        self.shared_memory_name = self._memory.name
        self._views = _views(self._memory.buf, layout)
        self.obs = {key: self._views[_OBS + key] for key in self._shapes}
        self._final = {key: self._views[_FINAL + key] for key in self._shapes}
        context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes = []
        for index in range(workers):
            ours, theirs = context.Pipe()
            slots = range(index * envs_per_worker, (index + 1) * envs_per_worker)
            process = context.Process(
                target=_worker,
                args=(theirs, factory, self._memory.name, layout, slots, seed, None if worker_cores is None else worker_cores[index]),
                daemon=True,
            )
            process.start()
            theirs.close()
            self._connections.append(ours)
            self._processes.append(process)
        self._closed = False

    def reset(self) -> dict[str, np.ndarray]:
        require(not self._closed, "reset after close")
        self._exchange(_RESET)
        return self.obs

    def step(self, actions: np.ndarray) -> StepResult:
        require(not self._closed, "step after close")
        check_shape(actions, (self.num_envs,), "actions")
        self._views["actions"][:] = actions
        episodes = self._exchange(_STEP)
        return StepResult(
            rewards=self._views["rewards"],
            terminated=self._views["terminated"],
            truncated=self._views["truncated"],
            final_obs=self._final,
            episodes=episodes,
        )

    def _exchange(self, command: str) -> list[dict[str, float]]:
        for connection in self._connections:
            connection.send(command)
        episodes: list[dict[str, float]] = []
        for connection in self._connections:
            if not connection.poll(_REPLY_TIMEOUT_S):
                raise WorkerFailure(f"no reply within {_REPLY_TIMEOUT_S} s")
            try:
                status, payload = connection.recv()
            except EOFError as error:
                raise WorkerFailure("a worker died without reporting") from error
            if status != "ok":
                raise WorkerFailure(payload)
            episodes.extend(payload)
        return episodes

    def close(self) -> None:  # nsp: ignore NSP001 teardown must always finish; a failed check here would leak memory
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.send(_CLOSE)
            except OSError as error:
                log.warning("a worker had already exited during close: %s", error)
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
        self.obs = {}
        self._final = {}
        self._views = {}
        self._memory.close()
        self._memory.unlink()
