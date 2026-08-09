"""Surface geometry backend wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

def load_surface_geometry(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a FreeSurfer surface."""

    from nibabel.freesurfer.io import read_geometry

    geometry = read_geometry(filepath=path) 
    vertices, faces = geometry[:2]

    return (
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=np.int64),
    )