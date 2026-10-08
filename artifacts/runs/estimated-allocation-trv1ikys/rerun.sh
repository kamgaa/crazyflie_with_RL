#!/bin/bash
set -euo pipefail
cd /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2
run_dir=$(mktemp -d artifacts/runs/estimated-allocation-XXXXXX)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl python compare_estimated_allocation.py --output-dir "$run_dir"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_estimated_allocation.py --run-dir "$run_dir"
