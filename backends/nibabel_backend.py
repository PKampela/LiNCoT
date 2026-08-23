"""NiBabel backend wrapper."""

from __future__ import annotations

from typing import cast
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel import loadsave
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    inv_ornt_aff,
    io_orientation,
    ornt_transform,
)
from nibabel.nifti1 import Nifti1Image

from core.frames import CoordinateFrame
from core.image import Image
from core.transform import Transform


def _affine_is_valid(matrix: np.ndarray | None) -> bool:
    if matrix is None:
        return False

    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (4, 4):
        return False
    if not np.all(np.isfinite(arr)):
        return False
    if np.linalg.matrix_rank(arr[:3, :3]) < 3:
        return False
    return True


def _select_affine(img: Nifti1Image) -> tuple[np.ndarray, str, int]:
    """Select voxel->world affine for import.

    Policy:
    - Prefer sform when present (industry-standard best-affine precedence).
    - Else use qform.
    - Else fall back to NiBabel's best affine.
    """

    qform, qform_code = img.get_qform(coded=True)
    sform, sform_code = img.get_sform(coded=True)

    if int(sform_code) > 0 and _affine_is_valid(sform):
        return np.asarray(sform, dtype=float), "sform", int(sform_code)

    if int(qform_code) > 0 and _affine_is_valid(qform):
        return np.asarray(qform, dtype=float), "qform", int(qform_code)

    best = np.asarray(img.affine, dtype=float)
    return best, "best_affine", 0


def _load_canonical(path: Path) -> Nifti1Image:
    """
    Load a NIfTI image and convert it to canonical RAS voxel order.

    All functions in this backend operate on canonical images so that
    voxel ordering is consistent regardless of how the file was stored.
    """

    img = cast(Nifti1Image, loadsave.load(path))
    selected_affine, _source, _code = _select_affine(img)
    data = np.asarray(img.dataobj)
    canonical_data, canonical_affine = _canonicalize_to_ras(data, selected_affine)
    return Nifti1Image(canonical_data, canonical_affine)


def _canonicalize_to_ras(
    data: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize array+affine to RAS orientation explicitly.

    This avoids relying on header-mediated affine selection in
    `as_closest_canonical` when qform/sform metadata conflict.
    """

    arr = np.asarray(data)
    aff = np.asarray(affine, dtype=float)

    if arr.ndim < 3:
        return arr, aff

    current_ornt = io_orientation(aff)
    target_ornt = axcodes2ornt(("R", "A", "S"))
    transform = ornt_transform(current_ornt, target_ornt)

    canonical_data = apply_orientation(arr, transform)
    canonical_affine = aff @ inv_ornt_aff(transform, arr.shape[:3])

    return np.asarray(canonical_data), np.asarray(canonical_affine, dtype=float)




def load_nifti_image(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI image in canonical RAS voxel order.
    """

    img = _load_canonical(path)
    data =np.asarray(img.dataobj)
    affine=np.asarray(img.affine, dtype=float)

    return data, affine


def voxel_to_world_transform(
    image: Image,
    voxel_frame: CoordinateFrame,
    world_frame: CoordinateFrame,
) -> Transform:
    """
    Create the voxel -> world transform for an imported image.

    Since Image.affine already represents the canonical voxel-to-world
    transform, this simply wraps it as a Transform object.
    """

    return Transform(
        source=voxel_frame,
        target=world_frame,
        matrix=image.affine.copy(),
    )


def world_to_voxel_transform(
    image: Image,
    voxel_frame: CoordinateFrame,
    world_frame: CoordinateFrame,
) -> Transform:
    """
    Create the world -> voxel transform for an imported image.
    """

    return voxel_to_world_transform(
        image,
        voxel_frame,
        world_frame,
    ).invert()