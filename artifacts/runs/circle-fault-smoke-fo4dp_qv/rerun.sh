#!/bin/bash
set -euo pipefail
cd /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2
run_dir=$(mktemp -d artifacts/runs/circle-fault-XXXXXX)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python compare_circle_faults.py --config /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/configs/eval_circle_fault_0289.yaml --output-dir "$run_dir" --smoke
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_circle_faults.py --run-dir "$run_dir"
