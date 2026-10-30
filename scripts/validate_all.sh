#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/mrl_6534/miniconda3/envs/isaaclab/bin/python
export PYTHONDONTWRITEBYTECODE=1
"$PY" scripts/validate_math.py
"$PY" scripts/run.py --headless --device cuda:0 --physics isaacsim_physx --num_envs 1 --validate --output results/physx_1
"$PY" scripts/run.py --headless --device cuda:0 --physics isaacsim_physx --num_envs 16 --validate --output results/physx_16
"$PY" scripts/compare_runs.py
"$PY" scripts/run.py --headless --device cuda:0 --physics isaacsim_physx --num_envs 16 --profile --output results/profile
"$PY" scripts/summarize.py
