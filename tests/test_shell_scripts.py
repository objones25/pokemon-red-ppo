"""The pod scripts' refusals, run for real against fake mounts and tools."""

from __future__ import annotations

import hashlib
import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest
import tyro
from support import REPO

from pokered.config import TrainConfig

SETUP = REPO / "scripts" / "setup.sh"
TRAIN = REPO / "scripts" / "train.sh"
START = REPO / "scripts" / "start.sh"
LAUNCH = REPO / "scripts" / "launch_pod.sh"


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
    # Stands in for pokered.wandb_check too: it accepts every key but the one named bad-key.
    _fake_tool(venv / "bin", "python", 'printf "%s\\n" "$*" >> "$TOOL_LOG"; [[ ${WANDB_API_KEY:-} != bad-key ]]')
    _fake_tool(venv / "bin", "pokered-train")
    _fake_tool(venv / "bin", "pokered-delete-pod", 'printf "delete-pod\\n" >> "$TOOL_LOG"')
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


ANSWERS = "wandb-key\n"


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


def test_setup_keeps_other_values_already_in_the_env_file(pod, tmp_path):
    (tmp_path / ".env").write_text("WANDB_ENTITY=team\nRUNPOD_API_KEY=runpod-key\n")
    _run(SETUP, pod, stdin=ANSWERS)
    assert (tmp_path / ".env").read_text() == "WANDB_API_KEY=wandb-key\nWANDB_ENTITY=team\nRUNPOD_API_KEY=runpod-key\n"


def test_setup_asks_only_for_the_wandb_key(pod):
    result = _run(SETUP, pod, stdin=ANSWERS)
    assert result.stderr.count("(hidden") == 1


def test_setup_does_not_ask_again_when_the_stored_key_still_works(pod):
    _run(SETUP, pod, stdin=ANSWERS)
    assert "(hidden" not in _run(SETUP, pod).stderr


def test_setup_uses_a_key_already_in_the_environment_without_asking(pod, tmp_path):
    result = _run(SETUP, {**pod, "WANDB_API_KEY": "from-secret"})
    assert "(hidden" not in result.stderr
    assert "WANDB_API_KEY=from-secret" in (tmp_path / ".env").read_text()


def test_setup_asks_again_after_a_rejected_key(pod, tmp_path):
    result = _run(SETUP, pod, stdin="bad-key\nwandb-key\n")
    assert result.returncode == 0
    assert "WANDB_API_KEY=wandb-key" in (tmp_path / ".env").read_text()


def test_setup_replaces_a_stored_key_that_stopped_working(pod, tmp_path):
    (tmp_path / ".env").write_text("WANDB_API_KEY=bad-key\n")
    _run(SETUP, pod, stdin=ANSWERS)
    assert "WANDB_API_KEY=wandb-key" in (tmp_path / ".env").read_text()


def test_setup_gives_up_after_three_rejected_keys(pod):
    result = _run(SETUP, pod, stdin="bad-key\n" * 3)
    assert result.returncode == 1
    assert "3 tries" in result.stderr


def test_setup_strips_whitespace_a_paste_brings_along(pod, tmp_path):
    _run(SETUP, pod, stdin="  wandb-key \n")
    assert "WANDB_API_KEY=wandb-key\n" in (tmp_path / ".env").read_text()


def test_setup_says_an_empty_answer_means_offline(pod, tmp_path):
    result = _run(SETUP, pod, stdin="\n")
    assert "offline" in result.stderr
    assert "WANDB_API_KEY" not in (tmp_path / ".env").read_text()


def test_setup_checks_the_key_without_putting_it_on_a_command_line(pod, tmp_path):
    _run(SETUP, pod, stdin=ANSWERS)
    log = (tmp_path / "tool.log").read_text()
    assert "-m pokered.wandb_check" in log
    assert "wandb-key" not in log


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


def test_train_uses_fast_mode_on_a_pod(pod, tmp_path):
    _run(SETUP, pod, stdin=ANSWERS)
    _run(TRAIN, pod)
    last = (tmp_path / "tool.log").read_text().splitlines()[-1]
    assert tyro.cli(TrainConfig, args=shlex.split(last)).env.fast_mode is True


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
    _run(SETUP, pod, stdin="\n")
    result = _run(TRAIN, pod)
    assert "--wandb-mode offline" in (tmp_path / "tool.log").read_text().splitlines()[-1]
    assert "no WANDB_API_KEY" in result.stderr


def test_train_leaves_wandb_mode_alone_when_it_has_a_key(pod, tmp_path):
    _run(SETUP, pod, stdin=ANSWERS)
    _run(TRAIN, pod)
    assert "--wandb-mode" not in (tmp_path / "tool.log").read_text().splitlines()[-1]


def test_train_lets_a_caller_choose_wandb_mode_after_the_default(pod, tmp_path):
    _run(SETUP, pod, stdin="\n")
    _run(TRAIN, pod, "--wandb-mode", "disabled")
    assert (tmp_path / "tool.log").read_text().splitlines()[-1].endswith("--wandb-mode disabled")


@pytest.fixture
def start_env(pod, tmp_path):
    _fake_tool(tmp_path / "bin", "nohup")  # Records the launch; a real one would race the assertions.
    return {**pod, "POKERED_RUN_DIR": str(tmp_path / "run"), "POKERED_NO_FOLLOW": "1"}


def test_start_runs_setup_before_launching(start_env, tmp_path):
    assert _run(START, start_env, stdin=ANSWERS).returncode == 0
    assert "WANDB_API_KEY=wandb-key" in (tmp_path / ".env").read_text()


def test_start_launches_the_supervisor_so_it_outlives_the_ssh_session(start_env, tmp_path):
    _run(START, start_env, "--total-steps", "1000", stdin=ANSWERS)
    last = (tmp_path / "tool.log").read_text().splitlines()[-1]
    assert last.endswith("start.sh --supervise --total-steps 1000")


def test_start_stops_when_setup_refuses(start_env, tmp_path):
    (tmp_path / "rom.gb").unlink()
    result = _run(START, start_env, stdin=ANSWERS)
    assert result.returncode == 1
    assert not (tmp_path / "tool.log").exists()


def test_start_does_not_launch_a_second_trainer(start_env, tmp_path):
    (tmp_path / "bin" / "ps").write_text('#!/bin/bash\nprintf "/opt/venv/bin/pokered-train --env.rom x\\n"\n')
    (tmp_path / "bin" / "ps").chmod(0o755)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "train.pid").write_text("4242\n")
    result = _run(START, start_env, stdin=ANSWERS)
    assert "already running" in result.stderr
    assert not (tmp_path / ".env").exists()


def test_start_launches_when_the_recorded_pid_is_something_else(start_env, tmp_path):
    (tmp_path / "bin" / "ps").write_text('#!/bin/bash\nprintf "/usr/sbin/sshd\\n"\n')
    (tmp_path / "bin" / "ps").chmod(0o755)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "train.pid").write_text("4242\n")
    _run(START, start_env, stdin=ANSWERS)
    assert "--supervise" in (tmp_path / "tool.log").read_text()


def _supervise(tmp_path, env, train_body: str, *args: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / ".env").write_text("WANDB_API_KEY=k\n")
    _fake_tool(tmp_path / "venv" / "bin", "pokered-train", train_body)
    return _run(START, env, "--supervise", *args)


def test_supervisor_deletes_the_pod_when_the_trainer_is_killed(start_env, tmp_path):
    """SIGKILL, as from the OOM killer, skips the trainer's own cleanup, and an idle A100 keeps billing."""
    result = _supervise(tmp_path, start_env, "kill -KILL $$")
    assert result.returncode == 137
    assert "delete-pod" in (tmp_path / "tool.log").read_text()


def test_supervisor_leaves_a_normal_exit_to_the_trainer(start_env, tmp_path):
    result = _supervise(tmp_path, start_env, "printf 'trained\\n' >> \"$TOOL_LOG\"")
    assert result.returncode == 0
    assert "delete-pod" not in (tmp_path / "tool.log").read_text()


def test_supervisor_keeps_the_pod_after_an_ordinary_failure_like_a_mistyped_option(start_env, tmp_path):
    result = _supervise(tmp_path, start_env, "printf 'bad option\\n' >> \"$TOOL_LOG\"; exit 2")
    assert result.returncode == 2
    assert "delete-pod" not in (tmp_path / "tool.log").read_text()


def test_supervisor_keeps_the_pod_when_asked_to(start_env, tmp_path):
    _supervise(tmp_path, start_env, "printf 'trained\\n' >> \"$TOOL_LOG\"; kill -KILL $$", "--no-terminate-pod-on-exit")
    assert "delete-pod" not in (tmp_path / "tool.log").read_text()


def test_supervisor_removes_its_pid_file_when_training_ends(start_env, tmp_path):
    _supervise(tmp_path, start_env, "exit 0")
    assert not (tmp_path / "run" / "train.pid").exists()


SSH_BODY = """
printf '%s\\n' "$*" >> "$TOOL_LOG"
case "$*" in
  *"test -f"*) exit "${ROM_ON_POD_STATUS:-0}" ;;
  *.part*) cat > "$UPLOADED" ;;
esac
"""


@pytest.fixture
def mac(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_tool(bin_dir, "ssh", SSH_BODY)
    rom = tmp_path / "local.gb"
    rom.write_bytes(b"rom bytes")
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "POKERED_ROM": str(rom),
        "POKERED_SSH_WAIT_SECONDS": "0",
        "POKERED_SSH_WAIT_ATTEMPTS": "2",
        "TOOL_LOG": str(tmp_path / "ssh.log"),
        "UPLOADED": str(tmp_path / "uploaded"),
    }


def test_launch_needs_a_host_and_port(mac):
    result = _run(LAUNCH, mac)
    assert result.returncode == 1
    assert "Usage" in result.stderr


def test_launch_refuses_before_contacting_the_pod_when_the_rom_is_missing(mac, tmp_path):
    (tmp_path / "local.gb").unlink()
    result = _run(LAUNCH, mac, "1.2.3.4", "2222")
    assert result.returncode == 1
    assert not (tmp_path / "ssh.log").exists()


def test_launch_gives_up_when_the_pod_never_answers(mac, tmp_path):
    _fake_tool(tmp_path / "bin", "ssh", "exit 255")
    result = _run(LAUNCH, mac, "1.2.3.4", "2222")
    assert result.returncode == 1
    assert "can't reach" in result.stderr


def test_launch_connects_to_the_given_host_and_port(mac, tmp_path):
    _run(LAUNCH, mac, "1.2.3.4", "2222")
    assert "-p 2222 root@1.2.3.4" in (tmp_path / "ssh.log").read_text()


def test_launch_clones_or_updates_the_repo_on_the_volume(mac, tmp_path):
    _run(LAUNCH, mac, "1.2.3.4", "2222")
    log = (tmp_path / "ssh.log").read_text()
    assert "git clone https://github.com/objones25/pokemon-red-ppo.git /workspace/pokemon-red-ppo" in log
    assert "pull --ff-only" in log


def test_launch_uploads_the_rom_when_the_volume_has_none(mac, tmp_path):
    _run(LAUNCH, {**mac, "ROM_ON_POD_STATUS": "1"}, "1.2.3.4", "2222")
    assert (tmp_path / "uploaded").read_bytes() == b"rom bytes"


def test_launch_skips_the_upload_when_the_volume_has_the_rom(mac, tmp_path):
    _run(LAUNCH, mac, "1.2.3.4", "2222")
    assert not (tmp_path / "uploaded").exists()


def test_launch_ends_in_a_terminal_running_start_with_the_extra_options(mac, tmp_path):
    _run(LAUNCH, mac, "1.2.3.4", "2222", "--run-name", "two words")
    last = (tmp_path / "ssh.log").read_text().splitlines()[-1]
    assert last.startswith("-t ")
    assert "scripts/start.sh --run-name two\\ words" in last


def test_launch_explains_that_a_dropped_connection_does_not_stop_training(mac, tmp_path):
    _fake_tool(tmp_path / "bin", "ssh", 'case "$*" in *start.sh*) exit 255 ;; *) cat >/dev/null ;; esac')
    result = _run(LAUNCH, mac, "1.2.3.4", "2222")
    assert "keeps running" in result.stderr


def test_launch_passes_the_run_name_to_start(mac, tmp_path):
    _run(LAUNCH, {**mac, "POKERED_RUN_NAME": "run3"}, "1.2.3.4", "2222")
    last = (tmp_path / "ssh.log").read_text().splitlines()[-1]
    assert "POKERED_RUN_NAME=run3 scripts/start.sh" in last


def test_launch_without_a_run_name_leaves_start_to_its_default(mac, tmp_path):
    _run(LAUNCH, mac, "1.2.3.4", "2222")
    assert "POKERED_RUN_NAME" not in (tmp_path / "ssh.log").read_text()


def test_launch_refuses_a_run_name_that_is_not_a_plain_word_before_contacting_the_pod(mac, tmp_path):
    result = _run(LAUNCH, {**mac, "POKERED_RUN_NAME": "run 3;rm"}, "1.2.3.4", "2222")
    assert result.returncode == 1
    assert "run name" in result.stderr
    assert not (tmp_path / "ssh.log").exists()
