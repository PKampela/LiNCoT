"""Parser for interactive command strings."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ParsedCommand:
    command: str
    args: List[str]
    kwargs: Dict[str, Any]


class CommandParseError(ValueError):
    """Raised when interactive command text is invalid."""


def parse_command(text: str) -> ParsedCommand:
    """Parse raw command text into command id, positional args, and optional flags.

    Supports positional arguments and optional flags (boolean or key-value).

    Examples:
        "point add p1 0 0 0 head"
            -> command="point.add", args=["p1", "0", "0", "0", "head"], kwargs={}

        "transform head scanner 10 20 30 --show-matrix --explain"
            -> command="transform", args=["head", "scanner", "10", "20", "30"],
               kwargs={"show_matrix": True, "explain": True}

        "volume load path/to/file.nii --name brain --register-transform"
            -> command="volume.load", args=["path/to/file.nii"],
               kwargs={"name": "brain", "register_transform": True}

        "transform head scanner 10 20 30 --chain mat1 mat2"
            -> command="transform", args=["head", "scanner", "10", "20", "30"],
               kwargs={"chain": ["mat1", "mat2"]}
    """

    stripped = text.strip()
    if not stripped:
        raise CommandParseError("Empty command")

    try:
        tokens = shlex.split(stripped)
    except ValueError as exc:
        raise CommandParseError(f"Invalid command syntax: {exc}") from exc

    if len(tokens) < 2:
        raise CommandParseError(
            "Command must include a group and an action, e.g. 'point add ...'"
        )

    # Separate positional args and flags
    args: List[str] = []
    kwargs: Dict[str, Any] = {}
    i = 2  # Skip group and action

    while i < len(tokens):
        token = tokens[i]

        # Check if it's a flag (starts with --)
        if token.startswith("--"):
            flag_name = token[2:].replace("-", "_")  # --show-matrix -> show_matrix

            # Check if next token is a value (doesn't start with --)
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                # Collect all non-flag values until next flag
                values = []
                i += 1
                while i < len(tokens) and not tokens[i].startswith("--"):
                    values.append(tokens[i])
                    i += 1

                # If single value, store as string; if multiple, store as list
                kwargs[flag_name] = values[0] if len(values) == 1 else values
                i -= 1  # Back up one since we'll increment at loop end
            else:
                # Boolean flag (no value)
                kwargs[flag_name] = True

        else:
            # Positional argument
            args.append(token)

        i += 1

    group, action = tokens[0], tokens[1]
    command = f"{group}.{action}".lower()

    return ParsedCommand(command=command, args=args, kwargs=kwargs)
