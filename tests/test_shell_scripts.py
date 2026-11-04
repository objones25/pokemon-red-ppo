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


def test_train_defaults_to_the_rollout_length_the_paper_preferred(pod, tmp_path):
    """On the pod T = 512 and T = 2048 ran at the same speed, within noise, so nothing argues against the longer one."""
    _run(SETUP, pod, stdin=ANSWERS)
    _run(TRAIN, pod)
    last = (tmp_path / "tool.log").read_text().splitlines()[-1]
    assert "--envs-per-worker 8 " in last
    assert "--ppo.rollout-steps 2048 " in last


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


def _workers_with_v1_quota(tmp_path, pod, quota: str, period: str = "100000") -> str:
    """A cgroup v1 host, as Runpod's CA-MTL-3 A100 PCIe was: no cpu.max, and the limit in two other files."""
    (tmp_path / "quota").write_text(f"{quota}\\n")
    (tmp_path / "period").write_text(f"{period}\\n")
    _run(SETUP, pod, stdin=ANSWERS)
    _run(TRAIN, {
        **pod,
        "POKERED_CPU_MAX_FILE": str(tmp_path / "missing"),
        "POKERED_CFS_QUOTA_FILE": str(tmp_path / "quota"),
        "POKERED_CFS_PERIOD_FILE": str(tmp_path / "period"),
    })  # fmt: skip
    return (tmp_path / "tool.log").read_text().splitlines()[-1]


def test_train_reads_the_cgroup_v1_limit_when_there_is_no_cpu_max(pod, tmp_path):
    assert "--workers 25 " in _workers_with_v1_quota(tmp_path, pod, "2635000")


def test_train_falls_back_to_cpu_count_when_the_v1_quota_is_unlimited(pod, tmp_path):
    expected = max((os.cpu_count() or 2) - 1, 1)
    assert f"--workers {expected} " in _workers_with_v1_quota(tmp_path, pod, "-1")


def _numa_flag(tmp_path, pod, node: str, bus_id: str = "00000000:81:00.0") -> str:
    _fake_tool(tmp_path / "bin", "nvidia-smi", f'printf "%s\\n" "{bus_id}"')
    sysfs = tmp_path / "pci"
    (sysfs / "0000:81:00.0").mkdir(parents=True)
    (sysfs / "0000:81:00.0" / "numa_node").write_text(f"{node}\n")
    _run(SETUP, pod, stdin=ANSWERS)
    _run(TRAIN, {**pod, "POKERED_PCI_SYSFS": str(sysfs)})
    return (tmp_path / "tool.log").read_text().splitlines()[-1]


def test_train_pins_to_the_numa_node_the_gpu_is_on(pod, tmp_path):
    assert "--numa-node 1 " in _numa_flag(tmp_path, pod, "1")


def test_train_uses_node_zero_when_the_kernel_reports_no_node(pod, tmp_path):
    assert "--numa-node 0 " in _numa_flag(tmp_path, pod, "-1")


def test_train_uses_node_zero_without_nvidia_smi(pod, tmp_path):
    _run(SETUP, pod, stdin=ANSWERS)
    _run(TRAIN, {**pod, "POKERED_PCI_SYSFS": str(tmp_path / "missing")})
    assert "--numa-node 0 " in (tmp_path / "tool.log").read_text().splitlines()[-1]


def test_train_uses_node_zero_when_the_bus_id_has_no_sysfs_entry(pod, tmp_path):
    assert "--numa-node 0 " in _numa_flag(tmp_path, pod, "1", bus_id="00000000:01:00.0")


def test_train_logs_offline_when_setup_was_given_no_wandb_key(pod, tmp_path):
    """Online W&B with no key raises at start-up, and on a pod that exit deletes the pod."""
    _run(SETUP, pod, stdin="\n\nrunpod-key\n")
    result = _run(TRAIN, pod)
    assert "--wandb-mode offline" in (tmp_path / "tool.log").read_text().splitlines()[-1]
    assert "no WANDB_API_KEY" in result.stderr


def test_train_leaves_wandb_mode_alone_when_it_has_a_key(pod, tmp_path):
    _run(SETUP, pod, stdin=ANSWERS)
    _run(TRAIN, pod)
    assert "--wandb-mode" not in (tmp_path / "tool.log").read_text().splitlines()[-1]


def test_train_lets_a_caller_choose_wandb_mode_after_the_default(pod, tmp_path):
    _run(SETUP, pod, stdin="\n\nrunpod-key\n")
    _run(TRAIN, pod, "--wandb-mode", "disabled")
    assert (tmp_path / "tool.log").read_text().splitlines()[-1].endswith("--wandb-mode disabled")
