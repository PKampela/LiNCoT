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


_TOP_LEVEL_COMMANDS = {"point", "frame", "transform", "volume", "surface", "view", "session", "help"}


def parse_command(text: str) -> ParsedCommand:
    """Parse raw command text into command id, positional args, and optional flags.

    Supports positional arguments and optional flags (boolean or key-value).

    Examples:
        "point add p1 0 0 0 head"
            -> command="point.add", args=["p1", "0", "0", "0", "head"], kwargs={}

        "transform p1 T1_voxel --show-matrix --explain"
            -> command="transform", args=["p1", "T1_voxel"],
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

    def _split_args_and_flags(remaining_tokens: List[str]) -> tuple[List[str], Dict[str, Any]]:
        args: List[str] = []
        kwargs: Dict[str, Any] = {}
        i = 0

        while i < len(remaining_tokens):
            token = remaining_tokens[i]

            if token.startswith("--"):
                flag_name = token[2:].replace("-", "_")
                if i + 1 < len(remaining_tokens) and not remaining_tokens[i + 1].startswith("--"):
                    values = []
                    i += 1
                    while i < len(remaining_tokens) and not remaining_tokens[i].startswith("--"):
                        values.append(remaining_tokens[i])
                        i += 1
                    kwargs[flag_name] = values[0] if len(values) == 1 else values
                    i -= 1
                else:
                    kwargs[flag_name] = True
            else:
                args.append(token)

            i += 1

        return args, kwargs

    if tokens[0].lower() == "transform":
        args, kwargs = _split_args_and_flags(tokens[1:])
        if not args:
            raise CommandParseError("Transform command requires at least a point or coordinate inputs")
        return ParsedCommand(command="transform", args=args, kwargs=kwargs)

    if len(tokens) >= 2 and tokens[1].lower() == "transform" and tokens[0].lower() not in _TOP_LEVEL_COMMANDS:
        synthetic_tokens = ["transform", tokens[0], *tokens[2:]]
        args, kwargs = _split_args_and_flags(synthetic_tokens[1:])
        return ParsedCommand(command="transform", args=args, kwargs=kwargs)

    if len(tokens) < 2:
        raise CommandParseError(
            "Command must include a group and an action, e.g. 'point add ...'"
        )

    args, kwargs = _split_args_and_flags(tokens[2:])
    group, action = tokens[0], tokens[1]
    command = f"{group}.{action}".lower()

    return ParsedCommand(command=command, args=args, kwargs=kwargs)
