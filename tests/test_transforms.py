import numpy as np
import pytest

from tmscoords.core.frames import CoordinateFrame
from tmscoords.core.point import Point
from tmscoords.core.transform import Transform


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
