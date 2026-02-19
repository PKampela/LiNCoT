from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np

from .frames import CoordinateFrame
from .point import Point
from .transform import Transform
from .chain import TransformChain
from ..registry.frames import FrameRegistry
from ..registry.transforms import TransformRegistry
from .image import Image

@dataclass
class Session:
    """
    Represents a single TMS session.

    Contains:
    - MRI images
    - points of interest
    - coordinate frames
    - transformations
    """

    images: Dict[str, Image] = field(default_factory=dict)
    points: Dict[str, Point] = field(default_factory=dict)
    frames: FrameRegistry = field(default_factory=FrameRegistry)
    transforms: TransformRegistry = field(default_factory=TransformRegistry)

    subject_id: Optional[str] = None
    description: Optional[str] = None

    def add_image(self, name: str, image: Image):
        self.images[name] = image

    def get_image(self, name: str) -> Image:
        return self.images[name]

    def add_point(self, name: str, point: Point):
        self.points[name] = point

    def get_point(self, name: str) -> Point:
        return self.points[name]

    def add_transform(self, name: str, transform: Transform):
        self.transforms.register_transform(name, transform)

    def get_transform_chain(self, names: List[str]) -> TransformChain:
        transforms = [self.transforms.get_transform(n) for n in names]
        return TransformChain(transforms)

    def apply_transform_to_point(self, point_name: str, chain_names: List[str]) -> Point:
        point = self.get_point(point_name)
        chain = self.get_transform_chain(chain_names)
        return chain.apply(point)

    def add_frame(self, frame: CoordinateFrame):
        self.frames.register_frame(frame)

    def get_frame(self, name: str) -> CoordinateFrame:
        return self.frames.get_frame(name)
