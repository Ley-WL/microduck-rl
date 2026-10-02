#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$here/train.sh" "${1:-Mjlab-StandUp-Flat-MicroDuck}" \
  --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.run-name "server-smoke-$(date +%Y%m%d-%H%M%S)"
