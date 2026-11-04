"""Runpod self-termination: find our own pod id, delete the pod, confirm with a read.

Ported from the user's massively-parallel-rl project. The network volume survives pod deletion, so checkpoints do too.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import httpx

from pokered.negative_space import bounded, require


# Verified 2026-10-27 against RunPod's REST v2 docs:
#   DELETE {base}/pods/{id}: 204 no body; 401/403/404/409/429 errors
#     https://docs.runpod.io/api-reference-v2/pods/terminate-a-pod
#   GET {base}/pods/{id}: 200 JSON with `status` (PROVISIONING, STARTING, RUNNING, EXITED, ERROR,
#     TERMINATED); a terminated pod is gone from listPods, so 404 is expected
#     https://docs.runpod.io/api-reference-v2/pods/get-a-pod
# Auth is `Authorization: Bearer <key>` only.
class MissingSecret(RuntimeError):
    """A required API key is not set."""


class OwnPodUnknown(RuntimeError):
    """This process cannot tell which pod it runs on."""


class TerminateFailed(RuntimeError):
    """The pod could not be confirmed deleted, so it may still be billing."""


RUNPOD_REST_BASE: Final[str] = "https://api.runpod.io/v2"

_POD_ID_VAR = "RUNPOD_POD_ID"
_API_KEY_VAR = "RUNPOD_API_KEY"
_POD_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_OK = 200
_NO_CONTENT = 204
_NOT_FOUND = 404
_REFUSED = frozenset({401, 403})
_TOO_MANY_REQUESTS = 429
_SERVER_ERROR_FLOOR = 500
_EXCERPT_CHARS = 200


def own_pod_id(environ: Mapping[str, str], proc1_environ: Path = Path("/proc/1/environ")) -> str:  # nsp: ignore NSP001 every failure raises OwnPodUnknown
    """The id of the pod we run on.

    RunPod sets RUNPOD_POD_ID for the container's PID 1 only; SSH login shells do not get it, so
    fall back to the NUL-separated KEY=VALUE entries of /proc/1/environ.
    """
    value = environ.get(_POD_ID_VAR)
    if value:
        return value
    try:
        raw = proc1_environ.read_bytes()
    except OSError as error:
        raise OwnPodUnknown(f"{_POD_ID_VAR} is not set and {proc1_environ} is unreadable: {error}") from error
    prefix = f"{_POD_ID_VAR}=".encode()
    for entry in raw.split(b"\x00"):
        if entry.startswith(prefix) and len(entry) > len(prefix):
            return entry[len(prefix) :].decode()
    raise OwnPodUnknown(f"{_POD_ID_VAR} is not set in the environment or in {proc1_environ}")


def own_api_key(environ: Mapping[str, str], proc1_environ: Path = Path("/proc/1/environ")) -> str:  # nsp: ignore NSP001 every failure raises MissingSecret
    """The API key from the environment, else the pod-scoped key Runpod gives the container's PID 1."""
    value = environ.get(_API_KEY_VAR)
    if value:
        return value
    prefix = f"{_API_KEY_VAR}=".encode()
    try:
        raw = proc1_environ.read_bytes()
    except OSError as error:
        raise MissingSecret(f"{_API_KEY_VAR} is not set and {proc1_environ} is unreadable: {error}") from error
    for entry in raw.split(b"\x00"):
        if entry.startswith(prefix) and len(entry) > len(prefix):
            return entry[len(prefix) :].decode()
    raise MissingSecret(f"{_API_KEY_VAR} is not set in the environment or in {proc1_environ}")


def own_terminator(environ: Mapping[str, str], client: httpx.Client) -> PodTerminator:
    """A ``PodTerminator`` for the pod we run on; ``MissingSecret`` / ``OwnPodUnknown`` when it cannot be built.

    The pod id comes from outside the program, so a malformed one is an operating error here rather than
    the constructor's contract violation.
    """
    api_key = own_api_key(environ)
    pod_id = own_pod_id(environ)
    if _POD_ID_PATTERN.fullmatch(pod_id) is None:
        raise OwnPodUnknown(f"malformed pod id {pod_id!r}: expected [A-Za-z0-9_-]+")
    return PodTerminator(client, api_key, pod_id, base_url=RUNPOD_REST_BASE)


class PodTerminator:
    def __init__(
        self,
        client: httpx.Client,
        api_key: str,
        pod_id: str,
        *,
        base_url: str,
        max_attempts: int = 5,
        base_delay_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        require(bool(api_key), "api_key must not be empty")
        require(_POD_ID_PATTERN.fullmatch(pod_id) is not None, f"pod_id must match [A-Za-z0-9_-]+, got {pod_id!r}")
        require(max_attempts >= 1, f"max_attempts must be at least 1, got {max_attempts}")
        require(base_delay_s >= 0, f"base_delay_s must be non-negative, got {base_delay_s}")
        self._client = client
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._pod_id = pod_id
        self._url = f"{base_url.rstrip('/')}/pods/{pod_id}"
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._sleep = sleep

    @property
    def pod_id(self) -> str:
        return self._pod_id

    def status(self) -> str | None:
        """The pod's status (GET, retrying transient failures), or None when it does not exist (404)."""
        return self._read_status()

    def terminate(self) -> None:
        """Read the pod, delete it, then confirm with a read that it is gone (404 or status TERMINATED).

        The pre-read makes a mis-resolved pod id loud: 404 there means we never saw the pod, so a later
        404 would prove nothing. An already TERMINATED pod is success (idempotent) and sends no DELETE.
        """
        before = self._read_status()
        if before is None:
            raise TerminateFailed(f"pod {self._pod_id} not found — refusing to report success")
        if before == "TERMINATED":
            return
        deleted = self._send("DELETE")
        if deleted.status_code == _NOT_FOUND:
            # Possibly already terminated; only the read can tell.
            status = self._read_status()
            if status is not None and status != "TERMINATED":
                raise TerminateFailed(f"DELETE of pod {self._pod_id} returned 404 but the pod is {status}")
            return
        if deleted.status_code not in (_OK, _NO_CONTENT):
            raise self._failure("DELETE", deleted)
        status = None
        for attempt in bounded(range(self._max_attempts), self._max_attempts, "termination confirmation"):
            status = self._read_status(refused_means_gone=True)
            if status is None or status == "TERMINATED":
                return
            if attempt < self._max_attempts - 1:
                self._sleep(self._delay(attempt + 1))
        raise TerminateFailed(f"pod {self._pod_id} still {status} after {self._max_attempts} reads following DELETE")

    def _delay(self, retry: int) -> float:
        return self._base_delay_s * 2 ** (retry - 1)

    def _read_status(self, *, refused_means_gone: bool = False) -> str | None:
        """The pod's status, or None when the API says it no longer exists (404).

        A pod-scoped key dies with its pod, so a key that was just able to read and delete it is refused afterwards.
        """
        response = self._send("GET")
        if response.status_code == _NOT_FOUND or (refused_means_gone and response.status_code in _REFUSED):
            return None
        if response.status_code != _OK:
            raise self._failure("GET", response)
        try:
            return str(response.json()["status"])
        except (ValueError, KeyError, TypeError) as error:
            raise TerminateFailed(
                f"GET of pod {self._pod_id} returned no status: {response.text[:_EXCERPT_CHARS]}"
            ) from error

    def _failure(self, method: str, response: httpx.Response) -> TerminateFailed:
        return TerminateFailed(
            f"{method} of pod {self._pod_id} failed with HTTP {response.status_code}: {response.text[:_EXCERPT_CHARS]}"
        )

    def _send(self, method: str) -> httpx.Response:  # nsp: ignore NSP001 the loop is bounded; exhausting it raises TerminateFailed
        """One request, retrying 5xx, 429 and transport errors; returns the first non-transient reply."""
        last = "no attempt"
        for attempt in bounded(range(self._max_attempts), self._max_attempts, f"{method} attempts"):
            if attempt > 0:
                self._sleep(self._delay(attempt))
            try:
                response = self._client.request(method, self._url, headers=self._headers)
            except httpx.TransportError as error:
                last = f"{type(error).__name__}: {error}"
                continue
            if response.status_code != _TOO_MANY_REQUESTS and response.status_code < _SERVER_ERROR_FLOOR:
                return response
            last = f"HTTP {response.status_code}: {response.text[:_EXCERPT_CHARS]}"
        raise TerminateFailed(f"{method} of pod {self._pod_id} failed after {self._max_attempts} attempts: {last}")
