import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import save
from pathlib import Path

from ..core.frames import CoordinateFrame
from ..core.point import Point
from ..core.transform import Transform
from ..backends.nibabel_backend import (
    load_nifti,
    voxel_to_world_transform,
    world_to_voxel_transform,
)


def test_identity_transform():
    frame = CoordinateFrame("head", ("R", "A", "S"), "mm")
    matrix = np.eye(4)
    transform = Transform(frame, frame, matrix)
    point = Point(np.array([1.0, 2.0, 3.0]), frame)
    result = transform.apply(point)
    assert np.allclose(result.coords, point.coords)


def test_inversion_correctness():
    head = CoordinateFrame("head", ("R", "A", "S"), "mm")
    mri = CoordinateFrame("mri", ("R", "A", "S"), "mm")
    matrix = np.eye(4)
    matrix[:3, 3] = [10.0, -5.0, 2.0]
    transform = Transform(head, mri, matrix)
    inv = transform.invert()
    point = Point(np.array([1.0, 2.0, 3.0]), head)
    forward = transform.apply(point)
    back = inv.apply(forward)
    assert np.allclose(back.coords, point.coords)


def test_frame_mismatch_raises():
    head = CoordinateFrame("head", ("R", "A", "S"), "mm")
    mri = CoordinateFrame("mri", ("R", "A", "S"), "mm")
    matrix = np.eye(4)
    transform = Transform(head, mri, matrix)
    wrong_point = Point(np.array([1.0, 2.0, 3.0]), mri)
    with pytest.raises(ValueError):
        transform.apply(wrong_point)

def test_nifti_backend_voxel_world_roundtrip(tmp_path: Path):
    shape = (32, 32, 32)
    data = np.zeros(shape, dtype=np.float32)
    data[16, 16, 16] = 1.0
    affine = np.eye(4)
    affine[:3, 3] = [10.0, -5.0, 2.0]
    img = Nifti1Image(data, affine)
    path = tmp_path / "test.nii"
    save(img, path)
    voxel = CoordinateFrame("voxel", ("i", "j", "k"), "index")
    world = CoordinateFrame("mri", ("R", "A", "S"), "mm")
    info = load_nifti(str(path))
    v2w = voxel_to_world_transform(info, voxel, world)
    w2v = world_to_voxel_transform(info, voxel, world)
    point_voxel = Point(np.array([16.0, 16.0, 16.0]), voxel)
    point_world = v2w.apply(point_voxel)
    back = w2v.apply(point_world)
    assert np.allclose(back.coords, point_voxel.coords)
    expected = (affine @ np.array([16.0, 16.0, 16.0, 1.0]))[:3]
    assert np.allclose(point_world.coords, expected)