from __future__ import annotations


def main():
    # Imported here, not at module level: spawned figure workers import this module as their
    # __main__, and spoqc.cli takes ~20 s and ~700 MB to import.
    from .cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
