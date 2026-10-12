#!/usr/bin/env bash
# Isolated source snapshot, reusing the already locked server dependency env.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES="${MICRODUCK_GPU:?Select a verified free physical GPU}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
venv="${MICRODUCK_SKILLS_VENV:-$PWD/../microduck-rl/.venv}"
exec "$venv/bin/train" "$@" --agent.logger tensorboard
