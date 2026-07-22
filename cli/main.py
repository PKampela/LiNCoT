"""CLI entry point for TMSLabs."""

from __future__ import annotations

import json
from typing import Optional, Sequence

from cli.console import Console
from cli.parser import CommandParseError, parse_command
from core.session import Session
from registry.command_registry import (
    CommandRegistry,
    register_default_commands,
)


def _bootstrap_session(base_session: Optional[Session] = None) -> Session:
    session = base_session or Session(subject_id="default", description="Interactive TMSLabs session")
    return session


def build_command_registry() -> CommandRegistry:
    registry = CommandRegistry()
    register_default_commands(registry)
    return registry


def run_interactive_cli(
    session: Session,
    registry: Optional[CommandRegistry] = None,
    console: Optional[Console] = None,
) -> int:
    runtime_registry = registry or build_command_registry()
    runtime_console = console or Console()

    runtime_console.print_lines(
        [
            "Interactive CLI ready. Type 'help' for commands, 'quit' to exit.",
        ]
    )

    while True:
        try:
            raw = runtime_console.read_input()
        except EOFError:
            runtime_console.print_lines(["Exiting CLI."])
            return 0

        stripped = raw.strip()
        if not stripped:
            continue

        if stripped.lower() in {"quit", "exit"}:
            runtime_console.print_lines(["Exiting CLI."])
            return 0

        try:
            parsed = parse_command(stripped)
            result = runtime_registry.execute(session, parsed.command, parsed.args, parsed.kwargs)
            
            # Handle JSON output format
            if result.output_format == "json":
                print(json.dumps(result.data, indent=2))
            else:
                runtime_console.print_lines(result.message.splitlines())
        except (CommandParseError, Exception) as exc:
            runtime_console.print_error(str(exc))


def main(argv: Optional[Sequence[str]] = None, session: Optional[Session] = None) -> int:
    """Main CLI entry point."""
    runtime_session = _bootstrap_session(session)
    
    # Interactive CLI is default
    return run_interactive_cli(runtime_session)


if __name__ == "__main__":
    raise SystemExit(main())
