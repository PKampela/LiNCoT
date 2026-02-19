"""Example: single-point transform using a simple affine."""

from __future__ import annotations

import numpy as np

from ..core.chain import TransformChain
from ..core.frames import CoordinateFrame
from ..core.point import Point
from ..core.transform import Transform


def main() -> None:
    head = CoordinateFrame("head", ("R", "A", "S"), "mm")
    mri = CoordinateFrame("mri", ("R", "A", "S"), "mm")

    affine = np.eye(4)
    affine[:3, 3] = [10.0, -5.0, 2.0]

    head_to_mri = Transform(source=head, target=mri, matrix=affine)
    chain = TransformChain([head_to_mri])

    point = Point(np.array([1.0, 2.0, 3.0]), head)
    transformed = chain.apply(point)

    print("Input:", point.coords, point.frame.name)
    print("Output:", transformed.coords, transformed.frame.name)


if __name__ == "__main__":
    main()
