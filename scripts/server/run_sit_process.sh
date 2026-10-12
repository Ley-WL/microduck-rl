#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES="${MICRODUCK_GPU:?Verified free GPU required}"
export OMP_NUM_THREADS=8 WANDB_MODE=offline MUJOCO_GL=egl MICRODUCK_WARM_START=1
task=Mjlab-SitStandProcess-Flat-MicroDuck-HD1910
mode="${1:?smoke or train}"
case "$mode" in
  smoke) iterations=5; worlds=64; run=process-smoke-v2 ;;
  train) iterations=1000; worlds=4096; run=process1000 ;;
  *) exit 2 ;;
esac
evidence="evidence/$mode"
mkdir -p "$evidence"
trap 'result=$?; printf "%s\n" "$result" > "$evidence/job.exit"' EXIT
venv=../microduck-rl/.venv
"$venv/bin/train" "$task" --env.scene.num-envs "$worlds" --agent.max-iterations "$iterations" \
  --agent.resume True --agent.load-run transfer_seed --agent.load-checkpoint seed.pt \
  --agent.run-name "$run" --agent.logger tensorboard > "$evidence/train.log" 2>&1
checkpoint=$(find logs/rsl_rl/hd1910_sitstand_process_v2 -path "*${run}/model_$((iterations-1)).pt" -print -quit)
test -n "$checkpoint"
"$venv/bin/python" scripts/export.py "$task" --checkpoint-file "$checkpoint" \
  --onnx-file "$PWD/$evidence/policy.onnx" --num-envs 1 --device cuda:0 > "$evidence/export.log" 2>&1
"$venv/bin/python" scripts/server/write_policy_metadata.py "$evidence/policy.onnx" "$task" > "$evidence/metadata.log" 2>&1
if [[ "$mode" == smoke ]]; then
  export MICRODUCK_CHECK_TASK="$task" MICRODUCK_CHECK_EXPERIMENT=hd1910_sitstand_process_v2 MICRODUCK_CHECK_RUN=process-smoke-v2
  "$venv/bin/python" scripts/server/smoke_sit_transfer.py > "$evidence/contract.log" 2>&1
else
  "$venv/bin/python" scripts/server/finalize_sit_process.py > "$evidence/evaluation.log" 2>&1
fi
printf 'complete\n' > "$evidence/completed.txt"
