#!/usr/bin/env bash
# Finish a named stand run after its detached trainer exits successfully.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
run="${1:?Usage: finalize_stand.sh RUN_ID CHECKPOINT_PATH}"
checkpoint="${2:?Missing final checkpoint path}"
task=Mjlab-StandHold-Flat-MicroDuck-HD1910
export CUDA_VISIBLE_DEVICES="${MICRODUCK_GPU:-7}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
export PATH="$HOME/.local/bin:$PATH"
until [[ -f "../runs/$run.exit" ]]; do sleep 15; done
[[ "$(cat "../runs/$run.exit")" == 0 ]] || { echo 'Training failed; inspect trainer log' >&2; exit 1; }
[[ -f "$checkpoint" ]] || { echo 'Final checkpoint missing' >&2; exit 1; }
scripts/server/export.sh "$task" "$checkpoint" "../exports/$run.onnx"
uv run --locked python scripts/server/verify_onnx.py "../exports/$run.onnx"
uv run --locked python scripts/server/evaluate_stand.py --onnx "../exports/$run.onnx" \
  --seconds 20 --envs 32 --push --video "../exports/$run.mp4"
uv run --locked python scripts/server/write_policy_metadata.py "../exports/$run.onnx"
printf '0\n' > "../runs/$run.finalize-exit"
