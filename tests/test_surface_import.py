"""Tests for surface import workflow."""

from pathlib import Path

import numpy as np
import pytest

from cli.main import _bootstrap_session, build_command_registry
from core.import_service import ImportError, UnsupportedFormatError
from nibabel.freesurfer.io import write_geometry


@pytest.fixture
def surface_file(tmp_path: Path) -> str:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    path = tmp_path / "lh.white"
    write_geometry(str(path), vertices, faces)
    return str(path)


def test_session_import_surface_registers_surface_and_frame(surface_file: str) -> None:
    session = _bootstrap_session()

    surface, info = session.import_surface(surface_file)

    assert surface is not None
    assert surface.frame.name == "surface_ras"
    assert session.list_surfaces() == ["lh_white"]
    assert session.get_surface("lh_white") is surface
    assert "Imported surface: lh_white" in info
    assert "Vertices: 3" in info
    assert "Faces: 1" in info
    assert "surface_ras" in session.frames.list_frames()


def test_surface_import_command_uses_same_workflow(surface_file: str) -> None:
    session = _bootstrap_session()
    registry = build_command_registry()

    result = registry.execute(session, "surface.import", [surface_file], {})

    assert result.data is not None
    assert result.data["surface"]["name"] == "lh_white"
    assert result.data["surface"]["frame"] == "surface_ras"
    assert session.get_surface("lh_white").frame.name == "surface_ras"


def test_surface_import_missing_file_raises() -> None:
    session = _bootstrap_session()

    with pytest.raises(ImportError):
        session.import_surface("C:/does/not/exist/lh.white")


def test_surface_import_rejects_unsupported_format(tmp_path: Path) -> None:
    session = _bootstrap_session()
    path = tmp_path / "surface.txt"
    path.write_text("not a surface", encoding="utf-8")

    with pytest.raises(UnsupportedFormatError):
        session.import_surface(str(path))
