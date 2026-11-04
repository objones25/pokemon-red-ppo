#!/usr/bin/env bash
#
# train.sh: train on a pod, resuming from the newest checkpoint on the volume, then delete the pod when training ends.
# The worker count comes from the cgroup CPU limit, which can be below the vCPU count Runpod reports.
#
# Usage: scripts/train.sh [pokered-train options, which override these defaults]
# Requires: bash 3.2+, awk; run scripts/setup.sh on this pod first.
# Exit status: that of pokered-train.
# Test-only overrides: POKERED_VENV, POKERED_ENV_FILE, POKERED_CPU_MAX_FILE, POKERED_PCI_SYSFS.
# bash-target: 3.2

set -Eeuo pipefail

readonly SCRIPT_NAME=${0##*/}
readonly VOLUME=/workspace
readonly VENV=${POKERED_VENV:-/opt/venv}
readonly ROM=${POKERED_ROM:-$VOLUME/rom/Pokemon Red.gb}
readonly RUN_NAME=${POKERED_RUN_NAME:-main}
readonly CPU_MAX_FILE=${POKERED_CPU_MAX_FILE:-/sys/fs/cgroup/cpu.max}
readonly PCI_SYSFS=${POKERED_PCI_SYSFS:-/sys/bus/pci/devices}
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
readonly REPO
readonly ENV_FILE=${POKERED_ENV_FILE:-$REPO/.env}

die() {
  printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

# Usable CPUs minus one for the main process, and never below one.
worker_count() {
  local cpus=""
  if [[ -r $CPU_MAX_FILE ]]; then
    cpus=$(awk '$1 != "max" { print int($1 / $2) }' "$CPU_MAX_FILE")
  fi
  if [[ -z $cpus ]]; then
    cpus=$(getconf _NPROCESSORS_ONLN)
  fi
  if ((cpus > 1)); then printf '%s\n' "$((cpus - 1))"; else printf '1\n'; fi
}

# The NUMA node of the first GPU, or 0 when it can't be told. nvidia-smi prints an 8-digit PCI domain, sysfs 4 digits.
gpu_numa_node() {
  local bus_id="" node=""
  bus_id=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | awk 'NR == 1 { print tolower($1) }') || true
  bus_id=${bus_id#0000}
  if [[ -n $bus_id && -r $PCI_SYSFS/$bus_id/numa_node ]]; then
    node=$(<"$PCI_SYSFS/$bus_id/numa_node")
  fi
  if [[ $node =~ ^[0-9]+$ ]]; then printf '%s\n' "$node"; else printf '0\n'; fi
}

main() {
  [[ -f $ENV_FILE ]] || die "$ENV_FILE is missing: run scripts/setup.sh first"
  [[ -x $VENV/bin/pokered-train ]] || die "$VENV has no pokered-train: run scripts/setup.sh first"
  set -a
  # shellcheck disable=SC1090 # written by setup.sh
  . "$ENV_FILE"
  set +a
  local workers
  workers=$(worker_count)
  exec "$VENV/bin/pokered-train" \
    --env.rom "$ROM" \
    --workers "$workers" \
    --envs-per-worker 8 \
    --ppo.rollout-steps 512 \
    --pin-workers \
    --numa-node "$(gpu_numa_node)" \
    --terminate-pod-on-exit \
    --run-name "$RUN_NAME" \
    --run-dir "$VOLUME/pokered/runs/$RUN_NAME" \
    "$@"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
