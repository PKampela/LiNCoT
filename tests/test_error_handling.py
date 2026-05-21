"""Tests for error handling and edge cases."""

import pytest

from cli.main import _bootstrap_session, build_command_registry
from cli.parser import CommandParseError, parse_command
from core.frames import CoordinateFrame
from core.transform import Transform
from registry.command_registry import CommandExecutionError


@pytest.fixture
def session():
    """Fresh session for each test."""
    session = _bootstrap_session()
    session.add_frame(CoordinateFrame("head", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mri", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mni", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("scanner", ("R", "A", "S"), "mm"))
    head = session.get_frame("head")
    mri = session.get_frame("mri")
    mni = session.get_frame("mni")
    session.add_transform(
        "head_to_mri",
        Transform(head, mri, [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]),
    )
    session.add_transform(
        "mri_to_mni",
        Transform(mri, mni, [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3], [0, 0, 0, 1]]),
    )
    return session


@pytest.fixture
def registry():
    """Fresh registry for each test."""
    return build_command_registry()


def test_invalid_frame_in_point_add(session, registry):
    """Registry: point.add with unknown frame raises CommandExecutionError."""
    with pytest.raises(CommandExecutionError):
        registry.execute(
            session,
            "point.add",
            ["p1", "0", "0", "0", "unknown_frame"],
            {}
        )


def test_invalid_coordinates_in_transform(session, registry):
    """Registry: non-numeric coordinates raise CommandExecutionError."""
    with pytest.raises(CommandExecutionError):
        registry.execute(
            session,
            "transform",
            ["head", "mni", "abc", "def", "ghi"],
            {}
        )


def test_impossible_transform_chain(session, registry):
    """Registry: unreachable frames in transform raise CommandExecutionError."""
    # Try to transform to a non-existent frame
    with pytest.raises(CommandExecutionError):
        registry.execute(
            session,
            "transform",
            ["head", "nonexistent_frame", "0", "0", "0"],
            {}
        )


def test_unknown_command_raises_keyerror(session, registry):
    """Registry: unknown command raises KeyError."""
    with pytest.raises(KeyError):
        registry.execute(session, "nonexistent.command", [], {})


def test_wrong_argument_count_point_add(session, registry):
    """Registry: point.add with wrong number of args raises CommandExecutionError."""
    # Needs exactly 5 args: name, x, y, z, frame
    with pytest.raises(CommandExecutionError):
        registry.execute(session, "point.add", ["p1", "0", "0"], {})
    
    with pytest.raises(CommandExecutionError):
        registry.execute(session, "point.add", ["p1", "0", "0", "0", "head", "extra"], {})


def test_wrong_argument_count_transform(session, registry):
    """Registry: transform with wrong number of args raises CommandExecutionError."""
    # transform now accepts 2/3 args for points or 5 args for legacy coordinates
    with pytest.raises(CommandExecutionError):
        registry.execute(session, "transform", ["head"], {})


def test_invalid_parser_syntax():
    """Parser: missing required group/action raises CommandParseError."""
    with pytest.raises(CommandParseError):
        parse_command("onlyaction")


def test_transform_object_first_parser():
    """Parser: object-first transform syntax maps to transform."""
    parsed = parse_command("transform p1 T1_voxel")

    assert parsed.command == "transform"
    assert parsed.args == ["p1", "T1_voxel"]
    assert parsed.kwargs == {}


def test_invalid_parser_empty():
    """Parser: empty string raises CommandParseError."""
    with pytest.raises(CommandParseError):
        parse_command("")


def test_frame_registered_after_first_command(session, registry):
    """Test that frames from session are properly used in commands."""
    # Add a point explicitly in head frame
    registry.execute(
        session,
        "point.add",
        ["p1", "10", "20", "30", "head"],
        {}
    )
    
    # Try to add point in mni frame (should work)
    registry.execute(
        session,
        "point.add",
        ["p2", "0", "0", "0", "mni"],
        {}
    )
    
    # Both should exist
    assert session.get_point("p1").frame.name == "head"
    assert session.get_point("p2").frame.name == "mni"


def test_command_with_wrong_flag_value(session, registry):
    """Registry: flags with incompatible values are handled."""
    # Transform with valid boolean flag should work
    result = registry.execute(
        session,
        "transform",
        ["head", "mni", "0", "0", "0"],
        {"json": True, "show_chain": True}
    )
    
    assert result.output_format == "json"
    assert "Transform chain:" in result.message


def test_point_with_extreme_coordinates(session, registry):
    """Registry: large coordinates are handled correctly."""
    result = registry.execute(
        session,
        "point.add",
        ["p1", "1000000", "-999999", "0.001", "head"],
        {}
    )
    
    assert "Added point" in result.message
    point = session.get_point("p1")
    assert point.coords[0] == 1000000
    assert point.coords[1] == -999999


def test_transform_preserves_coordinate_sign(session, registry):
    """Registry: negative coordinates are preserved through transform."""
    result = registry.execute(
        session,
        "transform",
        ["head", "mni", "-10", "-20", "-30"],
        {}
    )
    
    assert result.data["input"]["coords"] == [-10.0, -20.0, -30.0]


def test_multiple_flags_with_string_values(session, registry):
    """Registry: CommandExecutionError wraps file not found errors."""
    # volume.load should raise CommandExecutionError when file doesn't exist
    with pytest.raises(CommandExecutionError):
        registry.execute(
            session,
            "volume.load",
            ["dummy_path.nii"],
            {"name": "mybrain", "frame": "scanner", "transform_name": "voxel_to_world"}
        )
