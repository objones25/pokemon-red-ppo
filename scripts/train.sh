#!/usr/bin/env bash
#
# train.sh: train on a pod, resuming from the newest checkpoint on the volume, then delete the pod when training ends.
# The worker count comes from the cgroup CPU limit, which can be below the vCPU count Runpod reports.
#
# Usage: scripts/train.sh [pokered-train options, which override these defaults]
# Requires: bash 3.2+, awk; run scripts/setup.sh on this pod first.
# Exit status: that of pokered-train.
# Test-only overrides: POKERED_VENV, POKERED_ENV_FILE, POKERED_CPU_MAX_FILE.
# bash-target: 3.2

set -Eeuo pipefail

readonly SCRIPT_NAME=${0##*/}
readonly VOLUME=/workspace
readonly VENV=${POKERED_VENV:-/opt/venv}
readonly ROM=${POKERED_ROM:-$VOLUME/rom/Pokemon Red.gb}
readonly RUN_NAME=${POKERED_RUN_NAME:-main}
readonly CPU_MAX_FILE=${POKERED_CPU_MAX_FILE:-/sys/fs/cgroup/cpu.max}
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
    --terminate-pod-on-exit \
    --run-name "$RUN_NAME" \
    --run-dir "$VOLUME/pokered/runs/$RUN_NAME" \
    "$@"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
