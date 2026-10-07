#!/bin/bash
set -euo pipefail
cd /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python compare_motor_layouts.py --legacy-config /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/configs/eval_velocity_ab.yaml --user-config /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/configs/eval_velocity_ab_user_frd.yaml --previous-results /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/oracle-recovery-m3dh3vij --output-dir /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs
