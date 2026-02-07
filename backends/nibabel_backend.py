"""NiBabel backend wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

import numpy as np

from ..core.frames import CoordinateFrame
from ..core.transform import Transform
from nibabel.nifti1 import Nifti1Image


@dataclass(frozen=True)
class MRIImageInfo:
    """Minimal MRI image metadata extracted via NiBabel."""

    path: str
    shape: tuple[int, ...]
    affine: np.ndarray


def load_nifti(path: str) -> MRIImageInfo:
    """Load a NIfTI image and return minimal metadata.

    This function hides NiBabel objects from the user.
    """
    from nibabel import loadsave

    img = cast(Nifti1Image, loadsave.load(path))
    affine = np.asarray(img.affine, dtype=float)
    return MRIImageInfo(path=path, shape=img.shape, affine=affine)


def _select_affine(info: MRIImageInfo, sform: Optional[np.ndarray], qform: Optional[np.ndarray]) -> np.ndarray:
    if sform is not None and np.any(sform):
        return np.asarray(sform, dtype=float)
    if qform is not None and np.any(qform):
        return np.asarray(qform, dtype=float)
    return info.affine


def voxel_to_world_transform(
    info: MRIImageInfo,
    voxel_frame: CoordinateFrame,
    world_frame: CoordinateFrame,
    prefer_sform: bool = True,
) -> Transform:
    """Create a voxel -> world transform using qform/sform/affine."""

    from nibabel import loadsave

    img = cast(Nifti1Image, loadsave.load(info.path))
    sform = img.get_sform()
    qform = img.get_qform()
    matrix = _select_affine(info, sform if prefer_sform else qform, qform if prefer_sform else sform)
    return Transform(source=voxel_frame, target=world_frame, matrix=matrix)


def world_to_voxel_transform(
    info: MRIImageInfo,
    voxel_frame: CoordinateFrame,
    world_frame: CoordinateFrame,
    prefer_sform: bool = True,
) -> Transform:
    """Create a world -> voxel transform using qform/sform/affine."""

    return voxel_to_world_transform(info, voxel_frame, world_frame, prefer_sform).invert()
