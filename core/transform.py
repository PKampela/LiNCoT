"""Affine transform between coordinate frames."""

from __future__ import annotations

from typing import Mapping
from dataclasses import dataclass

import numpy as np

from core.asset import AssetMetadata

from .frames import CoordinateFrame
from .point import Point


@dataclass(frozen=True)
class Transform:
    """Represents an affine transform between two frames."""

    source: CoordinateFrame
    target: CoordinateFrame
    matrix: np.ndarray
    asset: AssetMetadata | None = None

    def __post_init__(self) -> None:
        matrix_array = np.asarray(self.matrix, dtype=float)
        if matrix_array.shape != (4, 4):
            raise ValueError("matrix must be a 4x4 affine matrix")
        object.__setattr__(self, "matrix", matrix_array)

    def apply(self, point: Point) -> Point:
        if point.frame != self.source:
            raise ValueError(
                f"Point frame '{point.frame.name}' does not match transform source '{self.source.name}'"
            )
        coords_h = np.append(point.coords, 1.0)
        transformed = self.matrix @ coords_h
        return Point(transformed[:3], self.target)

    def invert(self) -> "Transform":
        inv = np.linalg.inv(self.matrix)
        return Transform(source=self.target, target=self.source, matrix=inv)

    def to_dict(self) -> dict:
        """Convert the transform to a dictionary representation."""
        return {
            "source": self.source.name,
            "target": self.target.name,
            "matrix": self.matrix.tolist(),
            "asset": self.asset.to_dict() if self.asset else None,
        }

    @classmethod
    def from_dict(cls, data: dict, frames: Mapping[str, CoordinateFrame]) -> "Transform":
        source = frames[data["source"]]
        target = frames[data["target"]]
        asset = AssetMetadata.from_dict(data["asset"]) if data.get("asset") else None
        return cls(
            source=source,
            target=target,
            matrix=np.array(data["matrix"]),
            asset=asset,
        )
