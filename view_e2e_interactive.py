"""Import-safe single E2E policy interactive evaluation."""


def main(argv=None):
    from crazyflie_rl.interactive_eval import main as run
    return run(argv)


if __name__ == '__main__':
    raise SystemExit(main())
