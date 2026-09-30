"""Compare frozen PPO policies; importing this entrypoint never starts a run."""
from __future__ import annotations


def main(argv=None):
    from crazyflie_rl.dr_transfer import main as run
    return run(argv)


if __name__ == '__main__':
    raise SystemExit(main())
