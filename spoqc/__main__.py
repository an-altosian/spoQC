from __future__ import annotations

from .cli_args import build_parser
from .core import threads


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    threads.configure(8 if args.dev_test or args.step == "unittest" else args.threads)
    from .cli import main as run  # heavy imports only after the thread budget is set

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
