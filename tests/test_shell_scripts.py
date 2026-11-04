"""The pod scripts' refusals, run for real against fake mounts and tools."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

import pytest
from support import REPO

SETUP = REPO / "scripts" / "setup.sh"
TRAIN = REPO / "scripts" / "train.sh"


def _run(script: Path, env: dict[str, str], *args: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(script), *args],
        env={"PATH": os.environ["PATH"], "HOME": env.get("HOME", "/nonexistent"), **env},
        input=stdin, capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip


def _fake_tool(directory: Path, name: str, body: str = 'printf "%s\\n" "$*" >> "$TOOL_LOG"') -> None:
    tool = directory / name
    tool.write_text(f"#!/bin/bash\n{body}\n")
    tool.chmod(tool.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def pod(tmp_path):
    mounts = tmp_path / "mounts"
    mounts.write_text("overlay / overlay rw 0 0\nmfs /workspace fuse rw 0 0\n")
    rom = tmp_path / "rom.gb"
    rom.write_bytes(b"rom")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_tool(bin_dir, "uv")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    _fake_tool(venv / "bin", "python", 'printf "%s\\n" "$*" >> "$TOOL_LOG"; exit 0')
    _fake_tool(venv / "bin", "pokered-train")
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "POKERED_MOUNTS_FILE": str(mounts),
        "POKERED_VENV": str(venv),
        "POKERED_ENV_FILE": str(tmp_path / ".env"),
        "POKERED_ROM": str(rom),
        "POKERED_ROM_SHA1": hashlib.sha1(b"rom", usedforsecurity=False).hexdigest(),
        "POKERED_SYSTEM_PYTHON": "/bin/true",
        "TOOL_LOG": str(tmp_path / "tool.log"),
    }


ANSWERS = "wandb-key\nentity\nrunpod-key\n"


def test_setup_refuses_without_a_mounted_volume(pod, tmp_path):
    (tmp_path / "mounts").write_text("overlay / overlay rw 0 0\n")
    result = _run(SETUP, pod, stdin=ANSWERS)
    assert result.returncode == 1
    assert "/workspace is not mounted" in result.stderr


def test_setup_refuses_a_wrong_rom(pod):
    result = _run(SETUP, {**pod, "POKERED_ROM_SHA1": "0" * 40}, stdin=ANSWERS)
    assert result.returncode == 1
    assert "SHA-1" in result.stderr


def test_setup_refuses_a_missing_rom(pod, tmp_path):
    (tmp_path / "rom.gb").unlink()
    result = _run(SETUP, pod, stdin=ANSWERS)
    assert result.returncode == 1
    assert "ROM not found" in result.stderr


def test_setup_writes_env_file_only_the_owner_can_read(pod, tmp_path):
    assert _run(SETUP, pod, stdin=ANSWERS).returncode == 0
    assert stat.S_IMODE((tmp_path / ".env").stat().st_mode) == 0o600


def test_setup_stores_the_keys_it_was_given(pod, tmp_path):
    _run(SETUP, pod, stdin=ANSWERS)
    assert "WANDB_API_KEY=wandb-key" in (tmp_path / ".env").read_text()


def test_setup_keeps_existing_values_when_answer_is_empty(pod, tmp_path):
    _run(SETUP, pod, stdin=ANSWERS)
    _run(SETUP, pod, stdin="\n\n\n")
    assert "RUNPOD_API_KEY=runpod-key" in (tmp_path / ".env").read_text()


def test_setup_syncs_dependencies_into_the_venv(pod, tmp_path):
    _run(SETUP, pod, stdin=ANSWERS)
    assert "sync --frozen" in (tmp_path / "tool.log").read_text()


def test_setup_never_prints_a_secret(pod):
    result = _run(SETUP, pod, stdin=ANSWERS)
    assert "wandb-key" not in result.stdout + result.stderr


def test_train_refuses_before_setup(pod):
    result = _run(TRAIN, pod)
    assert result.returncode == 1
    assert "run scripts/setup.sh" in result.stderr


def test_train_runs_with_pod_defaults_after_setup(pod, tmp_path):
    _run(SETUP, pod, stdin=ANSWERS)
    assert _run(TRAIN, pod, "--total-steps", "1000").returncode == 0
    last = (tmp_path / "tool.log").read_text().splitlines()[-1]
    assert "--pin-workers" in last
    assert "--terminate-pod-on-exit" in last
    assert "--total-steps 1000" in last


def test_train_passes_the_rom_path(pod, tmp_path):
    _run(SETUP, pod, stdin=ANSWERS)
    _run(TRAIN, pod)
    assert str(tmp_path / "rom.gb") in (tmp_path / "tool.log").read_text().splitlines()[-1]


def _workers_flag(tmp_path, pod, cpu_max: str) -> str:
    limit = tmp_path / "cpu.max"
    limit.write_text(cpu_max)
    _run(SETUP, pod, stdin=ANSWERS)
    _run(TRAIN, {**pod, "POKERED_CPU_MAX_FILE": str(limit)})
    return (tmp_path / "tool.log").read_text().splitlines()[-1]


def test_train_leaves_one_core_of_the_cgroup_limit_for_the_main_process(pod, tmp_path):
    assert "--workers 26 " in _workers_flag(tmp_path, pod, "2720000 100000")


def test_train_falls_back_to_cpu_count_without_a_quota(pod, tmp_path):
    expected = max((os.cpu_count() or 2) - 1, 1)
    assert f"--workers {expected} " in _workers_flag(tmp_path, pod, "max 100000")
