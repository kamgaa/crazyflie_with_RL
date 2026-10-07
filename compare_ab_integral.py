#!/usr/bin/env python
"""Non-learning A/B evaluation: frozen PPO + external integral compensator."""
if __name__ == '__main__':
    from crazyflie_rl.integral_eval import main
    raise SystemExit(main())
