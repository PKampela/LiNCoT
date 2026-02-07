"""Transform chain composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .point import Point
from .transform import Transform


@dataclass(frozen=True)
class TransformChain:
    """Represents a sequence of transforms."""

    transforms: List[Transform]

    def __post_init__(self) -> None:
        if not self.transforms:
            raise ValueError("TransformChain requires at least one Transform")
        for idx in range(len(self.transforms) - 1):
            if self.transforms[idx].target != self.transforms[idx + 1].source:
                raise ValueError(
                    "TransformChain has discontinuity between "
                    f"'{self.transforms[idx].target.name}' and "
                    f"'{self.transforms[idx + 1].source.name}'"
                )

    @classmethod
    def from_iterable(cls, transforms: Iterable[Transform]) -> "TransformChain":
        return cls(list(transforms))

    @property
    def source(self):
        return self.transforms[0].source

    @property
    def target(self):
        return self.transforms[-1].target

    def apply(self, point: Point) -> Point:
        if point.frame != self.source:
            raise ValueError(
                f"Point frame '{point.frame.name}' does not match chain source '{self.source.name}'"
            )
        current = point
        for transform in self.transforms:
            current = transform.apply(current)
        return current

    def compose(self) -> Transform:
        matrix = np.eye(4)
        for transform in self.transforms:
            matrix = transform.matrix @ matrix
        return Transform(source=self.source, target=self.target, matrix=matrix)

    def provenance(self) -> list[str]:
        return [f"{t.source.name} -> {t.target.name}" for t in self.transforms]
