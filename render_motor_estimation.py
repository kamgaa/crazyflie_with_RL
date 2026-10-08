"""Render the specified saved A-best/user-motor-1 fault rollout as one MP4."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('MESA_SHADER_CACHE_DIR','/tmp/crazyflie-mesa')
from crazyflie_rl.fault_estimation_video import main

if __name__ == '__main__':
    main()
