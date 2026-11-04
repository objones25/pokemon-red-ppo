"""Root conftest.py: put this at the top of the tests/ tree.

pytest imports conftest.py automatically, and fixtures defined here are visible
to every test below this directory with no import statement.

What it buys you:
  * determinism  : stdlib and NumPy global RNGs reseeded before every test
  * isolation    : outbound network calls fail loudly (loopback and Unix sockets allowed)
  * hygiene      : cwd and app environment variables cannot leak between tests
  * hypothesis   : dev / ci / nightly / crosshair profiles, selected with --hypothesis-profile
  * markers      : tests under integration/, e2e/, symbolic/ are marked by directory

Verified against pytest 9.2.0 and Hypothesis 6.170 (October 2026).
"""

from __future__ import annotations

import importlib.util
import ipaddress
import os
import random
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

# One fixed seed for the whole suite. Any value works; it must never change.
SEED = 20260826

# Environment variable prefixes owned by the project under test. Edit these.
APP_ENV_PREFIXES = ("POKERED_", "WANDB_", "RUNPOD_")

MARKERS = (
    "slow: takes more than ~1s; deselected by default",
    "integration: crosses a process, service, or network boundary; deselected by default",
    "e2e: full end-to-end workflow; deselected by default",
    "gpu: requires CUDA; deselected by default",
    "security: security regression test (path traversal, injection, input limits)",
    "symbolic: runs Hypothesis with the CrossHair solver backend; deselected by default",
)

DIRECTORY_MARKERS = ("integration", "e2e", "symbolic")


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _deterministic_rngs() -> None:
    """Reseed the global RNGs before each test, so a test reproduces when run alone.

    This covers random.* and the legacy np.random.* functions. It does NOT reach
    an unseeded np.random.default_rng() or random.Random() created by the code
    under test; those need a seed passed in explicitly.
    """
    random.seed(SEED)
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np

        np.random.seed(SEED)


@pytest.fixture
def rng() -> random.Random:
    """An explicit, independent RNG that nothing else in the process can perturb."""
    return random.Random(SEED)


# --------------------------------------------------------------------------
# Isolation
# --------------------------------------------------------------------------
class NetworkBlockedError(RuntimeError):
    """Raised when a unit test tries to reach a non-loopback host."""


def _is_local(host: object) -> bool:
    if not isinstance(host, str):
        return False
    if host in {"localhost", ""}:
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make outbound network access fail immediately in unit tests.

    Blocks at the Python socket layer, which also catches calls made deep inside
    dependencies. Loopback and Unix-domain sockets stay allowed so local test
    servers and multiprocessing keep working. Known gaps: subprocesses (curl, git)
    and C extensions that open sockets without the Python socket module.

    Tests marked integration or e2e are exempt.
    """
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("e2e"):
        return

    def blocked(target: object) -> NetworkBlockedError:
        return NetworkBlockedError(
            f"Test tried to reach {target!r}. Stub the boundary with monkeypatch, "
            "or mark the test @pytest.mark.integration."
        )

    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if self.family == getattr(socket, "AF_UNIX", None) or (isinstance(address, tuple) and _is_local(address[0])):
            return real_connect(self, address)
        raise blocked(address)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any):
        if _is_local(host):
            return real_getaddrinfo(host, *args, **kwargs)
        raise blocked(host)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture(autouse=True)
def _isolated_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run each test in its own empty directory; relative writes land somewhere disposable."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove app configuration from the environment.

    PYTHONHASHSEED is set for child processes only. The running interpreter fixed
    its hash seed at startup; to pin it for this process, set it before launching
    pytest (CI does this in the workflow env).
    """
    for key in list(os.environ):
        if key.startswith(APP_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTHONHASHSEED", str(SEED))


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------
@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """A controllable clock, so tests never sleep.

    Patches time.time/monotonic/perf_counter/sleep on the time module. Code that did
    `from time import time` keeps the real function, and datetime.now() is not
    affected. For those, inject a clock or use the time-machine package.

        def test_timeout(frozen_clock):
            start_job()
            frozen_clock[0] += 3600
            assert job_is_expired()
    """
    import time as _time

    now = [1_000_000.0]
    monkeypatch.setattr(_time, "time", lambda: now[0])
    monkeypatch.setattr(_time, "monotonic", lambda: now[0])
    monkeypatch.setattr(_time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(_time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    return now


# --------------------------------------------------------------------------
# Hypothesis profiles
# --------------------------------------------------------------------------
if importlib.util.find_spec("hypothesis") is not None:
    from hypothesis import settings

    settings.register_profile("dev", max_examples=100)
    settings.register_profile("ci", max_examples=300, deadline=None, print_blob=True)
    settings.register_profile("nightly", max_examples=5_000, deadline=None, print_blob=True)
    if importlib.util.find_spec("hypothesis_crosshair_provider") or importlib.util.find_spec("hypothesis_crosshair"):
        settings.register_profile("crosshair", backend="crosshair", deadline=None, max_examples=200)
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


# --------------------------------------------------------------------------
# Markers and reporting
# --------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    """Register markers in code too, so this conftest works before pyproject is updated."""
    for line in MARKERS:
        config.addinivalue_line("markers", line)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark tests by directory, so the default -m deselection cannot be forgotten."""
    for item in items:
        parts = {p.name for p in Path(str(item.path)).parents}
        for name in DIRECTORY_MARKERS:
            if name in parts:
                item.add_marker(getattr(pytest.mark, name))


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Print the seed in the run header; it is what lets you reproduce a CI failure."""
    return [f"pytest-expert: seed={SEED} python={sys.version.split()[0]}"]
