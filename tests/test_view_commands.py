"""Tests for session-backed viewer commands."""

import numpy as np

from cli.main import _bootstrap_session, build_command_registry
from core.frames import CoordinateFrame
from core.image import Image
from core.surface import Surface


def _seed_frames(session):
    session.add_frame(CoordinateFrame("head", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mri", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mni", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("scanner", ("R", "A", "S"), "mm"))


def test_view_volume_returns_viewer_descriptor():
    session = _bootstrap_session()
    _seed_frames(session)
    registry = build_command_registry()

    image = Image(
        data=np.zeros((8, 9, 10), dtype=np.float32),
        affine=np.eye(4),
        frame=session.get_frame("scanner"),
    )
    session.add_image("brain", image)

    result = registry.execute(session, "view.volume", ["brain"], {})

    assert result.data is not None
    assert result.data["viewer"] == {"type": "volume", "name": "brain"}
    assert result.data["volume"]["shape"] == (8, 9, 10)
    assert "Opened volume viewer" in result.message


def test_view_surface_returns_viewer_descriptor():
    session = _bootstrap_session()
    _seed_frames(session)
    registry = build_command_registry()

    surface = Surface(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        frame=session.get_frame("head"),
    )
    session.add_surface("scalp", surface)

    result = registry.execute(session, "view.surface", ["scalp"], {})

    assert result.data is not None
    assert result.data["viewer"] == {"type": "surface", "name": "scalp"}
    assert result.data["surface"]["vertices"] == 3
    assert result.data["surface"]["faces"] == 1
    assert "Opened surface viewer" in result.message