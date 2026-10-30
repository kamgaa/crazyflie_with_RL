"""Phase10 continuation from phase9 final; unchanged task and PPO."""
import copy,json,types
from pathlib import Path
import torch
from tensordict import TensorDict
from ppo_common import dump_json,model_hash,sha256
from phase5_common import seed_all,rng_state,equal_state,memory,TERMS,REASONS,install_probe,loss_metrics
from phase6_common import jsonable,differences,tensor_sha,restore_rng,rng_fingerprints,adam_steps,preserved
from phase14z_metrics import Telemetry,TERMS,REASONS
BASE_IT=0
ADD_IT=512
FINAL_IT=512
NUM_ENVS=1024
ROLLOUT=32
BATCH=32768
BASE_TRANSITIONS=0
ADD_TRANSITIONS=16777216
FINAL_TRANSITIONS=16777216
BASE_UPDATES=0
INTERVAL=2097152

from phase14z_reward import apply_reward

