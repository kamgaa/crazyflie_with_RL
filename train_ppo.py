"""Legacy PPO entrypoint retained only for historical discoverability.

This module is not a source of current defaults and does not run the former
smoke-test pipeline. Use ``train_ppo_02.py`` with an explicit config profile.
Importing or executing this file never imports MuJoCo, NumPy, Torch or SB3.
"""

from __future__ import annotations

from typing import Sequence


LEGACY_NOTICE = """\
train_ppo.py is a legacy entrypoint and is not used for current training.

Use the official config-driven entrypoint instead:
  python train_ppo_02.py --config configs/residual_train.yaml
  python train_ppo_02.py --config configs/e2e_train.yaml

The former train_ppo.py hyperparameters, evaluation settings, and residual
scale are historical values and must not be used as current defaults.
"""


def main(argv: Sequence[str] | None = None) -> None:
    # ``argv`` is accepted to keep a conventional entrypoint signature. No
    # legacy options are interpreted because this command cannot start a run.
    del argv
    print(LEGACY_NOTICE, end="")


if __name__ == "__main__":
    main()
