#!/bin/bash
set -euo pipefail
cd /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2
run_dir=$(mktemp -d artifacts/runs/motor-estimation-XXXXXX)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python compare_motor_estimation.py --stage develop --output-dir "$run_dir"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python compare_motor_estimation.py --stage evaluate --output-dir "$run_dir"
MUJOCO_GL=egl python render_motor_estimation.py --run-dir "$run_dir"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_motor_estimation.py --run-dir "$run_dir"
