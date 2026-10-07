#!/bin/bash
set -euo pipefail
cd /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python audit_coordinate_contracts.py --config /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/configs/eval_velocity_ab.yaml --output-dir /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs
