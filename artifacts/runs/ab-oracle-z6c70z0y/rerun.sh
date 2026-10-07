#!/bin/bash
set -euo pipefail
cd /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python compare_ab_oracle.py --config /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/configs/eval_velocity_ab.yaml --previous-results /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/ab-integral-validation-lymyulpo --output-dir /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs
