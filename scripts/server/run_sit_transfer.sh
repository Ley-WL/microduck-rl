#!/usr/bin/env bash
# Finite offline transfer job. No robot access, no automatic deployment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="${MICRODUCK_GPU:?Verified free GPU required}"
export OMP_NUM_THREADS=8 WANDB_MODE=offline MUJOCO_GL=egl
export MICRODUCK_WARM_START=1
venv="$PWD/../microduck-rl/.venv"
task=Mjlab-SitStandTransfer-Flat-MicroDuck-HD1910
mode="${1:-train}"
if [[ "$mode" == smoke ]]; then
  iterations=5;worlds=64;run=transfer-smoke-v2
elif [[ "$mode" == train ]]; then
  iterations=1000;worlds=4096;run=retained-smooth1000
else
  exit 2
fi
evidence="evidence/$mode"
mkdir -p "$evidence"
trap 'result=$?; printf "%s\n" "$result" > "$evidence/job.exit"' EXIT
"$venv/bin/train" "$task" --env.scene.num-envs "$worlds" --agent.max-iterations "$iterations" \
  --agent.resume True --agent.load-run transfer_seed --agent.load-checkpoint seed.pt \
  --agent.run-name "$run" --agent.logger tensorboard > "$evidence/train.log" 2>&1
checkpoint=$(find logs/rsl_rl/hd1910_sitstand_transfer_v1 -path "*${run}/model_$((iterations-1)).pt" -print -quit)
test -n "$checkpoint"
"$venv/bin/python" scripts/export.py "$task" --checkpoint-file "$checkpoint" \
  --onnx-file "$PWD/$evidence/policy.onnx" --num-envs 1 --device cuda:0 > "$evidence/export.log" 2>&1
"$venv/bin/python" scripts/server/write_policy_metadata.py "$evidence/policy.onnx" "$task" > "$evidence/metadata.log" 2>&1
if [[ "$mode" == train ]]; then
  "$venv/bin/python" scripts/server/finalize_sit_transfer.py > "$evidence/evaluation.log" 2>&1
fi
printf 'complete\n' > "$evidence/completed.txt"
