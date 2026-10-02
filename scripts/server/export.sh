#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${MICRODUCK_GPU:-7}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
task="${1:?Usage: export.sh TASK_ID CHECKPOINT ONNX_FILE}"
checkpoint="${2:?Missing checkpoint}"
output="${3:?Missing output path}"
# Use the upstream exporter, including its observation normalizer.
exec uv run --locked scripts/export.py "$task" \
  --checkpoint-file "$checkpoint" --onnx-file "$output" --num-envs 1
