#!/usr/bin/env bash
# Finite first-look job: train 1000, export, matched GPU/CPU diagnostics, stop.
# Does not deploy a model or access the robot.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES="${MICRODUCK_GPU:?Choose a verified free GPU}"
export MUJOCO_GL=egl OMP_NUM_THREADS=8 WANDB_MODE=offline
export MUJOCO_EGL_DEVICE_ID="$MICRODUCK_GPU"
venv="$PWD/../microduck-rl/.venv"
task=Mjlab-SitStandContact-Flat-MicroDuck-HD1910
run_name="${MICRODUCK_RUN_NAME:-contact-firstlook1000}"
seed_options=()
if [[ -n "${MICRODUCK_SEED_RUN:-}" ]]; then
  export MICRODUCK_WARM_START=1
  seed_options=(--agent.resume True --agent.load-run "$MICRODUCK_SEED_RUN" --agent.load-checkpoint seed.pt)
fi
mkdir -p evidence/firstlook
trap 'result=$?; printf "%s\n" "$result" > evidence/firstlook/job.exit' EXIT
"$venv/bin/train" "$task" --env.scene.num-envs 4096 --agent.max-iterations 1000 \
  --agent.run-name "$run_name" --agent.logger tensorboard "${seed_options[@]}" > evidence/firstlook/train.log 2>&1
checkpoint=$(find logs/rsl_rl/hd1910_sitstand_contact_v3 -path "*${run_name}/model_999.pt" -print -quit)
test -n "$checkpoint"
"$venv/bin/python" scripts/export.py "$task" --checkpoint-file "$checkpoint" \
  --onnx-file "$PWD/evidence/firstlook/policy.onnx" --num-envs 1 --device cuda:0 > evidence/firstlook/export.log 2>&1
"$venv/bin/python" scripts/server/write_policy_metadata.py evidence/firstlook/policy.onnx "$task" > evidence/firstlook/metadata.log 2>&1
"$venv/bin/python" scripts/server/evaluate_skills.py --kind sitstand --onnx evidence/firstlook/policy.onnx \
  --output evidence/firstlook/gpu-nominal --envs 32 --seeds 1009 2027 --profile nominal --video > evidence/firstlook/gpu-nominal.log 2>&1
"$venv/bin/python" scripts/server/evaluate_skills.py --kind sitstand --onnx evidence/firstlook/policy.onnx \
  --output evidence/firstlook/gpu-robust --envs 32 --seeds 1009 2027 --profile robust > evidence/firstlook/gpu-robust.log 2>&1
"$venv/bin/python" scripts/evaluate_skills_cpu.py --onnx evidence/firstlook/policy.onnx \
  --output evidence/firstlook/cpu-runtime --trials 8 --seeds 1009 2027 --profile runtime --video > evidence/firstlook/cpu-runtime.log 2>&1
printf 'complete\n' > evidence/firstlook/completed.txt
