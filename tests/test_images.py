import numpy as np

from core.frames import CoordinateFrame
from core.image import Image, transform_image
from core.transform import Transform


def _frame(name: str) -> CoordinateFrame:
    return CoordinateFrame(name, ("R", "A", "S"), "mm")


def test_transform_image_translation_3d():
    source = _frame("source")
    target = _frame("target")
    data = np.zeros((5, 5, 5), dtype=np.float32)
    data[2, 2, 2] = 1.0
    image = Image(data=data, affine=np.eye(4), frame=source)

    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 0.0, 0.0]
    transform = Transform(source, target, matrix)

    result = transform_image(image, transform, order=0, output_affine=np.eye(4))

    assert result.frame == target
    assert np.allclose(result.affine, np.eye(4))
    assert result.data[3, 2, 2] == 1.0
    assert np.isclose(result.data.sum(), 1.0)


def test_transform_image_translation_2d():
    source = _frame("source")
    target = _frame("target")
    data = np.zeros((5, 5), dtype=np.float32)
    data[2, 2] = 1.0
    image = Image(data=data, affine=np.eye(3), frame=source)

    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 0.0, 0.0]
    transform = Transform(source, target, matrix)

    result = transform_image(image, transform, order=0, output_affine=np.eye(3))

    assert result.frame == target
    assert result.data[3, 2] == 1.0
    assert np.isclose(result.data.sum(), 1.0)
