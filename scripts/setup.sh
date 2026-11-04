#!/usr/bin/env bash
#
# setup.sh: prepare a Runpod pod for training. Refuses unless the network volume is mounted at /workspace and the ROM
# is on it. Installs into a venv on the container disk that reuses the image's torch, then asks for a W&B key unless
# a working one is already stored, and checks it with W&B.
# Run it on every new pod: the container disk doesn't survive one, and the repo, ROM and .env on the volume do.
#
# Usage: scripts/setup.sh
# Requires: bash 3.2+, uv, awk, shasum or sha1sum. The key is read from stdin without echo.
# Exit status: 1 on any refusal or failed step.
# Test-only overrides: POKERED_MOUNTS_FILE, POKERED_VENV, POKERED_ENV_FILE, POKERED_ROM_SHA1, POKERED_SYSTEM_PYTHON.
# bash-target: 3.2

set -Eeuo pipefail

readonly SCRIPT_NAME=${0##*/}
readonly VOLUME=/workspace
readonly MOUNTS_FILE=${POKERED_MOUNTS_FILE:-/proc/mounts}
readonly VENV=${POKERED_VENV:-/opt/venv}
readonly ROM=${POKERED_ROM:-$VOLUME/rom/Pokemon Red.gb}
readonly ROM_SHA1=${POKERED_ROM_SHA1:-ea9bcae617fdf159b045185467ae58b2e4a48b9a}
readonly SYSTEM_PYTHON=${POKERED_SYSTEM_PYTHON:-/usr/bin/python3.12}
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
readonly REPO
readonly ENV_FILE=${POKERED_ENV_FILE:-$REPO/.env}

die() {
  printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

sha1_of() {
  if command -v sha1sum >/dev/null 2>&1; then
    sha1sum -- "$1" | awk '{print $1}'
  else
    shasum -a 1 -- "$1" | awk '{print $1}'
  fi
}

existing_value() {
  [[ -f $ENV_FILE ]] || return 0
  awk -F= -v key="$1" '$1 == key { print substr($0, length(key) + 2) }' "$ENV_FILE"
}

key_works() {
  WANDB_API_KEY=$1 WANDB_SILENT=true "$VENV/bin/python" -m pokered.wandb_check
}

# Prints the W&B key to store, or nothing to log offline. Asks only when no working key is already known.
wandb_key() {
  local key answer attempt
  key=${WANDB_API_KEY:-$(existing_value WANDB_API_KEY)}
  if [[ -n $key ]] && key_works "$key"; then
    printf '%s\n' "$key"
    return 0
  fi
  for attempt in 1 2 3; do
    printf 'W&B API key from https://wandb.ai/authorize (hidden; Enter logs offline, try %s of 3): ' "$attempt" >&2
    answer=""
    IFS= read -r -s answer || true
    printf '\n' >&2
    answer=${answer//[[:space:]]/}
    if [[ -z $answer ]]; then
      printf '%s: no key, so this run logs offline; upload it later with wandb sync\n' "$SCRIPT_NAME" >&2
      return 0
    fi
    if [[ $answer =~ ^[A-Za-z0-9_.:-]+$ ]] && key_works "$answer"; then
      printf '%s\n' "$answer"
      return 0
    fi
  done
  die "no working W&B key after 3 tries; rerun, or press Enter at the prompt to log offline"
}

main() {
  awk -v point="$VOLUME" '$2 == point { found = 1 } END { exit !found }' "$MOUNTS_FILE" \
    || die "$VOLUME is not mounted: create the pod with a network volume there, or checkpoints won't survive it"
  [[ -f $ROM ]] || die "ROM not found at $ROM: copy your own there; it isn't distributed"
  [[ $(sha1_of "$ROM") == "$ROM_SHA1" ]] || die "$ROM has the wrong SHA-1; these RAM addresses only fit $ROM_SHA1"

  cd -- "$REPO"
  # On the container disk: a venv or uv cache on the volume uses up its quota.
  export UV_PROJECT_ENVIRONMENT=$VENV UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv} UV_LINK_MODE=copy
  if [[ ! -x $VENV/bin/python ]]; then
    uv venv --system-site-packages --python "$SYSTEM_PYTHON" "$VENV" || die "uv venv failed"
  fi
  uv sync --frozen --no-dev || die "uv sync failed"
  "$VENV/bin/python" -c 'import pokered.train, torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())' \
    || die "the environment can't import the project and torch"

  local key temporary name value
  key=$(wandb_key)
  temporary=$(mktemp "$ENV_FILE.XXXXXX") || die "cannot create a file next to $ENV_FILE"
  chmod 0600 "$temporary"
  {
    if [[ -n $key ]]; then printf 'WANDB_API_KEY=%s\n' "$key"; fi
    for name in WANDB_ENTITY RUNPOD_API_KEY; do # Optional, and never asked for: the pod has its own Runpod key.
      value=$(existing_value "$name")
      if [[ -n $value ]]; then printf '%s=%s\n' "$name" "$value"; fi
    done
  } >"$temporary"
  mv -f -- "$temporary" "$ENV_FILE"
  printf '%s: ready; start training with scripts/start.sh\n' "$SCRIPT_NAME" >&2
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
