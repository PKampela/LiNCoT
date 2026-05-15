"""Command-line interface."""

from . import main
from .main import build_command_registry, run_interactive_cli
from .parser import ParsedCommand, parse_command

__all__ = ["ParsedCommand", "build_command_registry", "main", "parse_command", "run_interactive_cli"]
