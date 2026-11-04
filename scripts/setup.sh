#!/usr/bin/env bash
#
# setup.sh: prepare a Runpod pod for training. Refuses unless the network volume is mounted at /workspace and the ROM
# is on it. Installs into a venv on the container disk that reuses the image's torch, then asks for API keys.
# Run it on every new pod: the container disk doesn't survive one, and the repo, ROM and .env on the volume do.
#
# Usage: scripts/setup.sh
# Requires: bash 3.2+, uv, awk, shasum or sha1sum. Keys are read from stdin without echo.
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

# Asks for one key on stdin; an empty answer keeps the current value. Never echoes a secret.
ask() {
  local name=$1 flags=$2 current answer=""
  current=$(existing_value "$name")
  printf '%s%s: ' "$name" "${current:+ [set; Enter keeps it]}" >&2
  if [[ $flags == -s ]]; then
    IFS= read -r -s answer || true
  else
    IFS= read -r answer || true
  fi
  printf '\n' >&2
  answer=${answer:-$current}
  [[ $answer =~ ^[A-Za-z0-9_.:-]*$ ]] || die "$name has characters this .env format can't hold"
  printf '%s=%s\n' "$name" "$answer"
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

  local temporary
  temporary=$(mktemp "$ENV_FILE.XXXXXX") || die "cannot create a file next to $ENV_FILE"
  chmod 0600 "$temporary"
  {
    ask WANDB_API_KEY -s
    ask WANDB_ENTITY ""
    ask RUNPOD_API_KEY -s
  } >"$temporary" || {
    rm -f -- "$temporary"
    exit 1
  }
  mv -f -- "$temporary" "$ENV_FILE"
  printf '%s: ready; start training with scripts/train.sh\n' "$SCRIPT_NAME" >&2
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
