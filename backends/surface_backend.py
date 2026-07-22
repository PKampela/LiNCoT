"""Surface geometry backend wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.frames import CoordinateFrame
from core.surface import Surface


def load_surface_geometry(path: str, frame: CoordinateFrame) -> Surface:
    """Load a FreeSurfer surface geometry file into a Surface object."""

    try:
        from nibabel.freesurfer.io import read_geometry
    except ImportError as exc:
        raise ImportError(f"Nibabel FreeSurfer surface reader unavailable: {exc}") from exc

    coords, faces = read_geometry(path)
    vertices = np.asarray(coords, dtype=float)
    face_array = np.asarray(faces, dtype=np.int64)
    surface_kind = Path(path).suffix.lower().lstrip(".")

    return Surface(
        vertices=vertices,
        faces=face_array,
        frame=frame,
        metadata={
            "source_path": path,
            "source_file": Path(path).name,
            "surface_kind": surface_kind,
        },
    )
