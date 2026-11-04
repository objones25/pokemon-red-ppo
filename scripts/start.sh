#!/usr/bin/env bash
#
# start.sh: the one command to run on a pod. Sets up, asks for a W&B key if none works, then starts training detached,
# so closing SSH or sleeping the Mac doesn't stop it, and shows the log. Run it again to reattach: it won't start a
# second trainer. If the trainer is killed outright (the OOM killer, say), this deletes the pod, since it can't.
#
# Usage: scripts/start.sh [pokered-train options]
# Requires: bash 3.2+; see scripts/setup.sh and scripts/train.sh.
# Exit status: 1 if setup refuses; otherwise 0 once the log view ends, whatever training is doing.
# Test-only overrides: POKERED_RUN_DIR, POKERED_NO_FOLLOW, and those of setup.sh and train.sh.
# bash-target: 3.2

set -Eeuo pipefail

readonly SCRIPT_NAME=${0##*/}
readonly VENV=${POKERED_VENV:-/opt/venv}
readonly RUN_DIR=${POKERED_RUN_DIR:-/workspace/pokered/runs/${POKERED_RUN_NAME:-main}}
readonly LOG=$RUN_DIR/train.log
readonly PID_FILE=$RUN_DIR/train.pid
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
readonly REPO
readonly SELF=$REPO/scripts/start.sh

# The trainer's pid if it is alive. The file is on the volume and can outlive its pod, so check what that pid really is.
running_pid() {
  local pid
  [[ -r $PID_FILE ]] || return 1
  pid=$(<"$PID_FILE")
  [[ $pid =~ ^[0-9]+$ ]] || return 1
  ps -p "$pid" -o command= 2>/dev/null | grep -q pokered-train || return 1
  printf '%s\n' "$pid"
}

keeps_pod() {
  local option
  for option in "$@"; do
    if [[ $option == --no-terminate-pod-on-exit ]]; then return 0; fi
  done
  return 1
}

# Runs detached. Any exit Python sees already deletes the pod in pokered.train; a status above 128 means a signal ended
# it first. Other failures, such as a mistyped option, leave the pod up for whoever is watching.
supervise() {
  local status=0
  "$REPO/scripts/train.sh" "$@" &
  printf '%s\n' "$!" >"$PID_FILE"
  wait "$!" || status=$?
  rm -f -- "$PID_FILE"
  if ((status > 128)) && ! keeps_pod "$@"; then
    printf '%s: the trainer was killed (status %s); deleting the pod\n' "$SCRIPT_NAME" "$status" >&2
    "$VENV/bin/pokered-delete-pod" || true # It reports its own failure, and this exit status should stay the trainer's.
  fi
  exit "$status"
}

main() {
  mkdir -p -- "$RUN_DIR"
  if [[ ${1:-} == --supervise ]]; then
    shift
    supervise "$@"
  fi
  local pid
  if pid=$(running_pid); then
    printf '%s: already running as pid %s; showing its log\n' "$SCRIPT_NAME" "$pid" >&2
  else
    "$REPO/scripts/setup.sh"
    nohup "$SELF" --supervise "$@" </dev/null >>"$LOG" 2>&1 &
    printf '%s: training started in the background\n' "$SCRIPT_NAME" >&2
  fi
  printf '%s: Ctrl-C or a dropped connection only stops this view. The pod deletes itself when training ends.\n' "$SCRIPT_NAME" >&2
  printf "%s: to stop early, saving a checkpoint and deleting the pod: kill \"\$(cat %s)\"\n" "$SCRIPT_NAME" "$PID_FILE" >&2
  if [[ -n ${POKERED_NO_FOLLOW:-} ]]; then
    wait
    return 0
  fi
  touch -- "$LOG"
  exec tail -n 100 -f -- "$LOG"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
