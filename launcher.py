"""Application launcher selecting CLI or GUI interface for LiNCoT."""

from __future__ import annotations

import argparse

from cli.main import build_command_registry, run_interactive_cli
from core.project import ProjectMetadata
from core.session import Session, SubjectMetadata
from gui.app import run_app


from core.project_manager import ProjectManager


def build_cli_startup_session() -> Session:
    """Create the initial session, recovering autosaved work if available."""

    manager = ProjectManager()

    if manager.has_recovery():
        try:
            return manager.load(manager.default_recovery_path())
        except Exception:
            # Recovery failed; start with a clean workspace.
            pass

    return Session.create_empty_session()

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LiNCoT", description="Launch LiNCoT in CLI or GUI mode")
    parser.add_argument("--cli", action="store_true", help="Start in interactive CLI mode")
    parser.add_argument("--gui", action="store_true", help="Start in GUI mode (default)")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    command_registry = build_command_registry()

    if args.cli and args.gui:
        raise SystemExit("Choose either --cli or --gui, not both")

    if args.cli:
        session = build_cli_startup_session()
        return run_interactive_cli(
            session=session,
            registry=command_registry,
        )
    return run_app(
        command_registry=command_registry,
    )

if __name__ == "__main__":
    raise SystemExit(main())
