import numpy as np
import pytest

from tmscoords.core.frames import CoordinateFrame
from tmscoords.core.point import Point


def test_frame_equality_by_name_and_axes():
    f1 = CoordinateFrame("head", ("R", "A", "S"), "mm", "desc1")
    f2 = CoordinateFrame("head", ("R", "A", "S"), "cm", "desc2")
    assert f1 == f2


def test_frame_invalid_axes():
    with pytest.raises(ValueError):
        CoordinateFrame("bad", ("R", "A"), "mm") # type: ignore


def test_point_requires_frame():
    frame = CoordinateFrame("head", ("R", "A", "S"), "mm")
    point = Point(np.array([1.0, 2.0, 3.0]), frame)
    assert point.frame == frame
