"""Tests for CommandRegistry command execution."""

import json
from pathlib import Path

import numpy as np
import pytest

from cli.main import _bootstrap_session, build_command_registry
from core.frames import CoordinateFrame
from core.point import Point
from core.session import Session
from registry.command_registry import CommandExecutionError


@pytest.fixture
def session():
    """Fresh session for each test."""
    session = _bootstrap_session()
    session.add_frame(CoordinateFrame("head", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mri", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mni", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("scanner", ("R", "A", "S"), "mm"))
    return session


@pytest.fixture
def registry():
    """Fresh registry for each test."""
    return build_command_registry()


@pytest.fixture
def working_repo_dir(monkeypatch):
    """Change working directory to repo root so transforms.json is accessible."""
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    return repo_root


def test_point_add_command(session, registry):
    """Registry: point.add adds point to session."""
    result = registry.execute(
        session,
        "point.add",
        ["p1", "10", "20", "30", "head"],
        {}
    )
    
    assert result.message == "Added point 'p1' in frame 'head'"
    assert session.get_point("p1") is not None
    assert np.allclose(session.get_point("p1").coords, [10, 20, 30])


def test_point_list_command_empty(session, registry):
    """Registry: point.list with no points."""
    result = registry.execute(session, "point.list", [], {})
    
    assert result.message == "No points in session"
    assert result.data["points"] == []


def test_point_list_command_with_points(session, registry):
    """Registry: point.list returns added points."""
    session.add_point("p1", Point(np.array([10, 20, 30]), session.get_frame("head")))
    session.add_point("p2", Point(np.array([1, 2, 3]), session.get_frame("mni")))
    
    result = registry.execute(session, "point.list", [], {})
    
    assert "p1, p2" in result.message
    assert len(result.data["points"]) == 2
    assert result.data["points"][0]["name"] == "p1"
    assert result.data["points"][1]["name"] == "p2"


def test_frame_list_command(session, registry):
    """Registry: frame.list shows available frames."""
    result = registry.execute(session, "frame.list", [], {})
    
    assert "head" in result.message
    assert "mri" in result.message
    assert "mni" in result.message
    assert "scanner" in result.message
    assert set(result.data["frames"]) == {"head", "mri", "mni", "scanner"}


def test_transform_list_command(session, registry, working_repo_dir):
    """Registry: transform.list shows registered transforms."""
    result = registry.execute(session, "transform.list", [], {})
    
    # Session transforms are explicit; nothing is preloaded by default.
    assert result.data["transforms"] == []
    assert result.message == "Transforms: none"


def test_transform_command_basic(session, registry, working_repo_dir):
    """Registry: transform resolves and applies transform."""
    result = registry.execute(
        session,
        "transform",
        ["head", "mni", "0", "0", "0"],
        {}
    )
    
    assert result.output_format == "text"
    assert "Input point (head)" in result.message
    assert "Output point (mni)" in result.message
    assert result.data["input"]["frame"] == "head"
    assert result.data["output"]["frame"] == "mni"


def test_transform_command_with_json_flag(session, registry, working_repo_dir):
    """Registry: transform returns JSON format when requested."""
    result = registry.execute(
        session,
        "transform",
        ["head", "mni", "10", "20", "30"],
        {"json": True}
    )
    
    assert result.output_format == "json"
    assert result.data is not None
    assert result.data["input"]["coords"] == [10.0, 20.0, 30.0]


def test_transform_command_with_show_chain(session, registry, working_repo_dir):
    """Registry: transform with --show-chain flag."""
    result = registry.execute(
        session,
        "transform",
        ["head", "mni", "0", "0", "0"],
        {"show_chain": True}
    )
    
    assert "Transform chain:" in result.message
    assert "head_to_mri" in result.message
    assert "mri_to_mni" in result.message


def test_transform_command_with_show_matrix(session, registry, working_repo_dir):
    """Registry: transform with --show-matrix flag."""
    result = registry.execute(
        session,
        "transform",
        ["head", "mni", "0", "0", "0"],
        {"show_matrix": True}
    )
    
    assert "Composed affine:" in result.message
    assert "composed_matrix" in result.data


def test_transform_with_numeric_kwargs(session, registry, working_repo_dir):
    """Registry: numeric arguments parsed as strings work correctly."""
    result = registry.execute(
        session,
        "transform",
        ["head", "mni", "100.5", "-50.25", "75"],
        {}
    )
    
    assert result.data["input"]["coords"] == [100.5, -50.25, 75.0]


def test_session_summary_command(session, registry):
    """Registry: session.summary shows current state."""
    # Add a point
    session.add_point("p1", Point(np.array([1, 2, 3]), session.get_frame("head")))
    
    result = registry.execute(session, "session.summary", [], {})
    
    assert "Session summary" in result.message
    assert "1 points" in result.message
    assert result.data["points"] == ["p1"]
    assert 4 == len(result.data["frames"])  # head, mri, mni, scanner


def test_help_command(session, registry):
    """Registry: help lists all commands."""
    result = registry.execute(session, "help", [], {})
    
    assert "point.add" in result.message
    assert "point.list" in result.message
    assert "transform" in result.message
    assert "help" in result.message
    assert "session.summary" in result.message


def test_transform_with_numeric_kwargs(session, registry, working_repo_dir):
    """Registry: numeric arguments parsed as strings work correctly."""
    result = registry.execute(
        session,
        "transform",
        ["head", "mni", "100.5", "-50.25", "75"],
        {}
    )
    
    assert result.data["input"]["coords"] == [100.5, -50.25, 75.0]
