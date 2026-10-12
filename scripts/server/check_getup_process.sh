#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OMP_NUM_THREADS=4
job=evidence/getup-process
mkdir -p "$job"
trap 'result=$?; printf "%s\n" "$result" > "$job/job.exit"' EXIT
python=../microduck-rl/.venv/bin/python
"$python" scripts/evaluate_skills_cpu.py --onnx evidence/xgoduck_getup.onnx --output "$job/nominal" --trials 4 --seeds 8111 9001 --capture-policy-frames > "$job/nominal.log" 2>&1
"$python" scripts/evaluate_skills_cpu.py --onnx evidence/xgoduck_getup.onnx --output "$job/stress" --trials 4 --seeds 8111 9001 --capture-policy-frames --transition-stress --joint-velocity-lag-steps 1 > "$job/stress.log" 2>&1
printf 'complete\n' > "$job/completed.txt"
