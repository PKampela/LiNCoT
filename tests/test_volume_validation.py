"""Tests for NIfTI volume affine handling and voxel<->world transforms.

Creates a synthetic NIfTI and verifies the backend-derived transforms
match the known affine and perform correct round-trips.
"""

import numpy as np
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import save
from pathlib import Path

from core.frames import CoordinateFrame
from core.point import Point
from backends.nibabel_backend import load_nifti, voxel_to_world_transform, world_to_voxel_transform


def test_nifti_affine_and_voxel_world_roundtrip(tmp_path: Path):
    # Create a small synthetic NIfTI with a known affine
    shape = (8, 8, 8)
    data = np.zeros(shape, dtype=np.float32)
    data[4, 4, 4] = 1.0
    affine = np.eye(4, dtype=float)
    affine[:3, 3] = [12.0, -6.0, 3.0]

    img = Nifti1Image(data, affine)
    path = tmp_path / "test_brain.nii"
    save(img, path)

    info = load_nifti(str(path))

    # Create frames and ask backend for transforms
    voxel_frame = CoordinateFrame("test_voxel", ("i", "j", "k"), "voxel")
    mri_frame = CoordinateFrame("test_mri", ("R", "A", "S"), "mm")

    v2w = voxel_to_world_transform(info, voxel_frame, mri_frame)
    w2v = world_to_voxel_transform(info, voxel_frame, mri_frame)

    # The voxel->world transform should equal the image affine
    assert np.allclose(v2w.matrix, affine)

    # Check voxel->world application for a known voxel
    voxel = np.array([4.0, 4.0, 4.0])
    world_expected = (affine @ np.append(voxel, 1.0))[:3]
    from core.point import Point as CorePoint
    p_voxel = CorePoint(voxel, voxel_frame)
    p_world = v2w.apply(p_voxel)
    assert np.allclose(p_world.coords, world_expected)

    # Round-trip: world -> voxel should map back to original voxel
    p_voxel_back = w2v.apply(p_world)
    assert np.allclose(p_voxel_back.coords, voxel)
