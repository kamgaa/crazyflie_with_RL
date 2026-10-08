#!/bin/bash
set -euo pipefail
cd /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2
render_dir=$(mktemp -d artifacts/runs/motor-estimation-render-XXXXXX)
MUJOCO_GL=egl python render_motor_estimation.py --run-dir /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/motor-estimation-g43e4hbx --output-dir "$render_dir"
