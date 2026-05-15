"""Application launcher selecting CLI or GUI interface."""

from __future__ import annotations

import argparse

from cli.main import _bootstrap_session, build_command_registry, run_interactive_cli
from core.session import Session
from gui.app import run_app


def build_session() -> Session:
    return _bootstrap_session(Session(subject_id="default", description="TMSCoords session"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tmscoords")
    parser.add_argument("--cli", action="store_true", help="Start in interactive CLI mode")
    parser.add_argument("--gui", action="store_true", help="Start in GUI mode (default)")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    session = build_session()
    command_registry = build_command_registry()

    if args.cli and args.gui:
        raise SystemExit("Choose either --cli or --gui, not both")

    if args.cli:
        return run_interactive_cli(session=session, registry=command_registry)

    return run_app(session=session, command_registry=command_registry)


if __name__ == "__main__":
    raise SystemExit(main())
