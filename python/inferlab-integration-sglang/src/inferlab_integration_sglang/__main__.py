from inferlab_adapter_sdk import run_adapter

from . import plan_serve, render_serve


def main() -> None:
    raise SystemExit(run_adapter(plan_serve, render_serve=render_serve))


if __name__ == "__main__":
    main()
