import numpy as np
import pytest

from core.chain import TransformChain
from core.frames import CoordinateFrame
from core.point import Point
from core.transform import Transform


def _frame(name: str) -> CoordinateFrame:
    return CoordinateFrame(name, ("R", "A", "S"), "mm")


def test_chain_composition():
    head = _frame("head")
    mri = _frame("mri")
    mni = _frame("mni")

    t1 = Transform(head, mri, np.array([[1, 0, 0, 1],
                                       [0, 1, 0, 2],
                                       [0, 0, 1, 3],
                                       [0, 0, 0, 1]], dtype=float))
    t2 = Transform(mri, mni, np.array([[1, 0, 0, -1],
                                      [0, 1, 0, -2],
                                      [0, 0, 1, -3],
                                      [0, 0, 0, 1]], dtype=float))

    chain = TransformChain([t1, t2])
    point = Point(np.array([0.0, 0.0, 0.0]), head)
    result = chain.apply(point)
    assert np.allclose(result.coords, [0.0, 0.0, 0.0])


def test_chain_discontinuity_raises():
    head = _frame("head")
    mri = _frame("mri")
    mni = _frame("mni")

    t1 = Transform(head, mri, np.eye(4))
    t2 = Transform(head, mni, np.eye(4))

    with pytest.raises(ValueError):
        TransformChain([t1, t2])
