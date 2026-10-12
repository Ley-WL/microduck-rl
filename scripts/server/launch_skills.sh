#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_id="${1:?Specify unique run ID}"
shift
[[ "$run_id" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 1
run_dir="$(cd "$here/../../.." && pwd)/runs"
mkdir -p "$run_dir"
[[ ! -e "$run_dir/$run_id.pid" ]] || { echo "Run ID exists" >&2; exit 1; }
export MICRODUCK_EXIT_FILE="$run_dir/$run_id.exit"
nohup bash -c 'bash "$@"; result=$?; printf "%s\n" "$result" > "$MICRODUCK_EXIT_FILE"; exit "$result"' \
    bash "$here/train_skills.sh" "$@" > "$run_dir/$run_id.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$run_dir/$run_id.pid"
printf 'PID: %s\nLog: %s\n' "$!" "$run_dir/$run_id.log"
