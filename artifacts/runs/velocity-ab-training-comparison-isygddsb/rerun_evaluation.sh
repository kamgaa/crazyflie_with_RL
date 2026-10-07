#!/bin/bash
set -euo pipefail
cd /home/kenneth/RL_src/crazyflie_with_RL_refactor_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/crazyflie-dr-mpl \
/home/kenneth/miniconda3/envs/crazyflie_rl/bin/python compare_dr_policies.py \
  --config configs/eval_velocity_ab.yaml \
  --model A_final=/home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_final_20261002-140542.zip \
  --model A_best=/home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-a-absolute-observation-absolute-reward_seed42_best_20261002-140542-10.zip \
  --model B_final=/home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_final_20261002-140542.zip \
  --model B_best=/home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_20261002-140542/models/ppo_e2e_hover_velocity-ab-b-absolute-observation-error-reward_seed42_best_20261002-140542-10.zip \
  --model historical_nominal=/home/kenneth/RL_src/crazyflie_with_RL_refactor_v2/artifacts/runs/ppo_e2e_hover_nominal_seedunset_20260928-120033/models/ppo_e2e_hover_nominal_seedunset_best_20260928-120033-13.zip \
  --cases hover step-005 --seed 42
