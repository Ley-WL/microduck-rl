#!/usr/bin/env bash
# Run from anywhere; keep the CUDA device and offline logs explicit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${MICRODUCK_GPU:-7}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE=offline
export UV_HTTP_TIMEOUT=600
task="${1:?Usage: train.sh TASK_ID [train arguments]}"
shift
exec uv run --locked train "$task" --agent.logger tensorboard "$@"
