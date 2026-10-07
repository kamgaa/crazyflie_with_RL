#!/bin/bash
set -euo pipefail
cd /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python compare_ab_integral.py --config /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/configs/eval_velocity_ab.yaml --record /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/velocity-ab-training-comparison-isygddsb/completion.json --previous-results /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/ab-payload-motor-791j3j2g --output-dir /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs --xi-xy-limit 0.4 --xi-z-limit 0.15
