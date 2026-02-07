"""Point representation with frame context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .frames import CoordinateFrame


@dataclass(frozen=True)
class Point:
    """Represents a 3D point with a coordinate frame."""

    coords: np.ndarray
    frame: CoordinateFrame

    def __post_init__(self) -> None:
        coords_array = np.asarray(self.coords, dtype=float)
        if coords_array.shape != (3,):
            raise ValueError("coords must be a 3-element vector")
        object.__setattr__(self, "coords", coords_array)

    @classmethod
    def from_iterable(cls, coords: Iterable[float], frame: CoordinateFrame) -> "Point":
        return cls(np.asarray(coords, dtype=float), frame)
