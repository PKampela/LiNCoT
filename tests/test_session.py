"""Tests for session state persistence and isolation."""

import numpy as np
import pytest

from cli.console import Console
from cli.main import _bootstrap_session, build_command_registry, run_interactive_cli
from core.frames import CoordinateFrame
from core.point import Point
from core.session import Session


class MockConsole(Console):
    """Mock console for testing interactive sessions."""
    
    def __init__(self, commands):
        self.commands = commands
        self.idx = 0
        self.output = []
    
    def read_input(self):
        if self.idx >= len(self.commands):
            raise EOFError()
        cmd = self.commands[self.idx]
        self.idx += 1
        return cmd
    
    def print_lines(self, lines):
        self.output.extend(lines)
    
    def print_error(self, msg):
        self.output.append(f"ERROR: {msg}")


def test_interactive_session_persistence():
    """Multiple commands in sequence share session state."""
    session = _bootstrap_session()
    session.add_frame(CoordinateFrame("head", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mri", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mni", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("scanner", ("R", "A", "S"), "mm"))
    registry = build_command_registry()
    
    mock_console = MockConsole([
        "point add p1 10 20 30 head",
        "point add p2 5 15 25 mni",
        "exit"
    ])
    
    result = run_interactive_cli(session, registry, mock_console)
    assert result == 0
    
    # Points added in commands should persist in session after CLI exits
    assert session.get_point("p1") is not None
    assert session.get_point("p2") is not None
    assert len(session.list_points()) == 2


def test_session_state_survives_multiple_operations():
    """Session maintains state through add, list, and transform."""
    session = _bootstrap_session()
    session.add_frame(CoordinateFrame("head", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mri", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mni", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("scanner", ("R", "A", "S"), "mm"))
    registry = build_command_registry()
    
    mock_console = MockConsole([
        "point add target 100 50 75 head",
        "transform head mni 100 50 75",  # Uses same point coordinates
        "point list",
        "exit"
    ])
    
    result = run_interactive_cli(session, registry, mock_console)
    assert result == 0
    
    # Point should exist after transform command
    assert session.get_point("target") is not None
    assert len(session.list_points()) == 1


def test_separate_sessions_are_isolated():
    """Two sessions don't share state."""
    session1 = _bootstrap_session()
    session2 = _bootstrap_session()
    for session in (session1, session2):
        session.add_frame(CoordinateFrame("head", ("R", "A", "S"), "mm"))
        session.add_frame(CoordinateFrame("mri", ("R", "A", "S"), "mm"))
        session.add_frame(CoordinateFrame("mni", ("R", "A", "S"), "mm"))
        session.add_frame(CoordinateFrame("scanner", ("R", "A", "S"), "mm"))
    
    # Add point to session1
    session1.add_point("p1", Point(np.array([1, 2, 3]), session1.get_frame("head")))
    
    # Verify isolation
    assert len(session1.list_points()) == 1
    assert len(session2.list_points()) == 0
    
    # Session2 should not have p1
    with pytest.raises(KeyError):
        session2.get_point("p1")


def test_session_summary_reflects_state():
    """Session.summary command shows current accumulated state."""
    session = _bootstrap_session()
    session.add_frame(CoordinateFrame("head", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mri", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mni", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("scanner", ("R", "A", "S"), "mm"))
    registry = build_command_registry()
    
    mock_console = MockConsole([
        "point add p1 1 2 3 head",
        "point add p2 4 5 6 mni",
        "session.summary",
        "exit"
    ])
    
    result = run_interactive_cli(session, registry, mock_console)
    assert result == 0
    
    # Verify points were actually added
    points = session.list_points()
    assert len(points) == 2
    assert "p1" in points
    assert "p2" in points


def test_frames_are_available_across_commands():
    """Frames registered in session are available to all commands."""
    session = _bootstrap_session()
    session.add_frame(CoordinateFrame("head", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mri", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mni", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("scanner", ("R", "A", "S"), "mm"))
    registry = build_command_registry()
    
    # Verify default frames are available
    mock_console = MockConsole([
        "frame list",
        "point add p1 0 0 0 head",
        "point add p2 0 0 0 mni",
        "exit"
    ])
    
    result = run_interactive_cli(session, registry, mock_console)
    assert result == 0
    
    # Both points should be in different frames
    p1_frame = session.get_point("p1").frame.name
    p2_frame = session.get_point("p2").frame.name
    assert p1_frame == "head"
    assert p2_frame == "mni"


def test_command_errors_dont_break_session():
    """Session remains valid after command error."""
    session = _bootstrap_session()
    session.add_frame(CoordinateFrame("head", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mri", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mni", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("scanner", ("R", "A", "S"), "mm"))
    registry = build_command_registry()
    
    mock_console = MockConsole([
        "point add p1 1 2 3 head",
        "point add p2 bad bad bad head",  # This will fail
        "point list",  # This should still work
        "exit"
    ])
    
    result = run_interactive_cli(session, registry, mock_console)
    assert result == 0
    
    # First point should exist, session should be recoverable
    assert session.get_point("p1") is not None
    assert len(session.list_points()) == 1  # Only p1 was added successfully


