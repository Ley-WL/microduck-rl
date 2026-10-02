#!/usr/bin/env bash
# Detached launcher with PID, console log and completion status.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_id="${1:?Usage: launch.sh RUN_ID TASK_ID [train arguments]}"
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Use letters, digits, _ or - for RUN_ID' >&2; exit 1; }
shift
[[ $# -gt 0 ]] || { echo 'Missing task ID' >&2; exit 1; }
run_dir="${MICRODUCK_RUN_DIR:-$here/../../../runs}"
mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
[[ ! -e "$run_dir/$run_id.pid" ]] || { echo 'RUN_ID already exists; choose a new one' >&2; exit 1; }
export MICRODUCK_EXIT_FILE="$run_dir/$run_id.exit"
nohup bash -c '"$@"; result=$?; printf "%s\n" "$result" > "$MICRODUCK_EXIT_FILE"; exit "$result"' \
  bash "$here/train.sh" "$@" > "$run_dir/$run_id.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run_dir/$run_id.pid"
printf 'PID: %s\nLog: %s\nExit status: %s\n' "$!" "$run_dir/$run_id.log" "$MICRODUCK_EXIT_FILE"
