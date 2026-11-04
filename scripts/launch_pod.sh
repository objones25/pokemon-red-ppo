#!/usr/bin/env bash
#
# launch_pod.sh: from the Mac, put everything on a fresh pod and start training. Waits for SSH, clones or updates the repo
# on the volume, uploads the ROM if the volume has none, then runs scripts/start.sh in a terminal, which asks for a W&B key.
# Safe to rerun: it reattaches to a running trainer. Once training has started, the Mac can sleep.
#
# Usage: scripts/launch_pod.sh HOST PORT [pokered-train options]
#   HOST and PORT are the pod's public IP and the external port mapped to 22/tcp.
# Requires: bash 3.2+, ssh with a key the pod accepts, and the ROM at the repo root.
# Exit status: 1 on a refusal or if the pod never answers; otherwise that of the remote command.
# Test-only overrides: POKERED_ROM, POKERED_SSH_WAIT_SECONDS, POKERED_SSH_WAIT_ATTEMPTS.
# bash-target: 3.2

set -Eeuo pipefail

readonly SCRIPT_NAME=${0##*/}
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
readonly REPO
readonly ROM=${POKERED_ROM:-$REPO/Pokemon Red.gb}
readonly REPO_URL=https://github.com/objones25/pokemon-red-ppo.git
readonly POD_REPO=/workspace/pokemon-red-ppo
readonly POD_ROM="/workspace/rom/Pokemon Red.gb"
readonly WAIT_SECONDS=${POKERED_SSH_WAIT_SECONDS:-5}
readonly WAIT_ATTEMPTS=${POKERED_SSH_WAIT_ATTEMPTS:-60}
readonly SSH_LOST=255

die() {
  printf '%s: error: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

main() {
  if (($# < 2)); then
    printf 'Usage: %s HOST PORT [pokered-train options]\n' "$SCRIPT_NAME" >&2
    exit 1
  fi
  local host=$1 port=$2
  shift 2
  [[ -f $ROM ]] || die "ROM not found at $ROM"

  # Pods reuse addresses, so remember this one's host key for this run only rather than in ~/.ssh/known_hosts.
  known_hosts=$(mktemp "${TMPDIR:-/tmp}/pokered-known-hosts.XXXXXX") || die "cannot create a temporary file"
  trap 'rm -f -- "$known_hosts"' EXIT
  local options=(-o "UserKnownHostsFile=$known_hosts" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10
    -o ServerAliveInterval=30 -p "$port" "root@$host")

  local attempt reached=""
  for ((attempt = 1; attempt <= WAIT_ATTEMPTS; attempt++)); do
    if ssh -o BatchMode=yes "${options[@]}" true </dev/null; then
      reached=yes
      break
    fi
    /bin/sleep "$WAIT_SECONDS"
  done
  [[ -n $reached ]] || die "can't reach root@$host:$port after $WAIT_ATTEMPTS tries; is the pod running, with 22/tcp exposed?"

  ssh -o BatchMode=yes "${options[@]}" \
    "if [ -d $POD_REPO/.git ]; then git -C $POD_REPO pull --ff-only; else git clone $REPO_URL $POD_REPO; fi" </dev/null \
    || die "couldn't put the repo on the volume; is one mounted at /workspace?"

  if ! ssh -o BatchMode=yes "${options[@]}" "test -f \"$POD_ROM\"" </dev/null; then
    printf '%s: uploading the ROM\n' "$SCRIPT_NAME" >&2
    ssh -o BatchMode=yes "${options[@]}" \
      "mkdir -p \"${POD_ROM%/*}\" && cat > \"$POD_ROM.part\" && mv -f \"$POD_ROM.part\" \"$POD_ROM\"" <"$ROM" \
      || die "the ROM upload failed"
  fi

  local remote="cd $POD_REPO && scripts/start.sh"
  if (($# > 0)); then remote+=$(printf ' %q' "$@"); fi
  local status=0
  ssh -t "${options[@]}" "$remote" || status=$?
  if ((status == SSH_LOST)); then
    printf '%s: the connection ended. Training keeps running on the pod unless it finished and deleted the pod; rerun this command to watch it.\n' "$SCRIPT_NAME" >&2
  fi
  return "$status"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
