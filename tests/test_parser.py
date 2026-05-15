"""Tests for CLI command parser."""

import pytest

from cli.parser import CommandParseError, parse_command


def test_parser_simple_command():
    """Parse simple command with positional arguments."""
    parsed = parse_command("point add p1 10 20 30 head")
    assert parsed.command == "point.add"
    assert parsed.args == ["p1", "10", "20", "30", "head"]
    assert parsed.kwargs == {}


def test_parser_with_boolean_flags():
    """Parse command with boolean flags."""
    parsed = parse_command("transform head mni 0 0 0 --json --show-matrix")
    assert parsed.command == "transform.head"
    assert parsed.args == ["mni", "0", "0", "0"]
    assert parsed.kwargs["json"] is True
    assert parsed.kwargs["show_matrix"] is True


def test_parser_with_value_flags():
    """Parse command with key-value flags."""
    parsed = parse_command("volume load path.nii --name brain --frame mri")
    assert parsed.command == "volume.load"
    assert parsed.args == ["path.nii"]
    assert parsed.kwargs["name"] == "brain"
    assert parsed.kwargs["frame"] == "mri"


def test_parser_with_list_values():
    """Parse command with flag accepting multiple values."""
    parsed = parse_command("transform head mni 0 0 0 --chain mat1 mat2 mat3")
    assert parsed.command == "transform.head"
    assert parsed.args == ["mni", "0", "0", "0"]
    assert parsed.kwargs["chain"] == ["mat1", "mat2", "mat3"]


def test_parser_mixed_flags_and_values():
    """Parse command with mix of boolean and value flags."""
    parsed = parse_command("transform head mni 10 20 30 --show-chain --frame scanner --json")
    assert parsed.command == "transform.head"
    assert parsed.args == ["mni", "10", "20", "30"]
    assert parsed.kwargs["show_chain"] is True
    assert parsed.kwargs["frame"] == "scanner"
    assert parsed.kwargs["json"] is True


def test_parser_invalid_syntax_missing_group():
    """Raise error when group and action missing (only one token)."""
    with pytest.raises(CommandParseError):
        parse_command("transform")  # Only group, no action


def test_parser_invalid_syntax_only_group():
    """Raise error when only group provided."""
    with pytest.raises(CommandParseError):
        parse_command("point")


def test_parser_empty_command():
    """Raise error on empty input."""
    with pytest.raises(CommandParseError):
        parse_command("")


def test_parser_whitespace_only():
    """Raise error on whitespace-only input."""
    with pytest.raises(CommandParseError):
        parse_command("   ")


def test_parser_quoted_arguments():
    """Parse arguments with quotes correctly."""
    parsed = parse_command('volume load "path with spaces.nii" --name "my brain"')
    assert parsed.args == ["path with spaces.nii"]
    assert parsed.kwargs["name"] == "my brain"
