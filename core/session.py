from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from .frames import CoordinateFrame
from .point import Point
from .transform import Transform
from .chain import TransformChain
from registry.frame_registry import FrameRegistry
from registry.transform_registry import TransformRegistry
from registry.surface_registry import SurfaceRegistry
from .image import Image
from .surface import Surface

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
    surfaces: SurfaceRegistry = field(default_factory=SurfaceRegistry)

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

    def list_points(self) -> List[str]:
        return sorted(self.points.keys())

    def list_images(self) -> List[str]:
        return sorted(self.images.keys())

    def add_surface(self, name: str, surface: Surface):
        self.surfaces.register_surface(name, surface)

    def get_surface(self, name: str) -> Surface:
        return self.surfaces.get_surface(name)

    def list_surfaces(self) -> List[str]:
        return self.surfaces.list_surfaces()

    def import_transform(
        self,
        path: str,
        source_frame_name: str | None = None,
        target_frame_name: str | None = None,
    ) -> tuple[Transform, str]:
        """Import a transform from file into the session.

        Routes to appropriate backend based on file extension.
        Automatically maps frames and registers the transform.

        Args:
            path: Path to transform file (.fif, etc)

        Returns:
            Tuple of (Transform object, info message)

        Raises:
            ImportError: If import or validation fails
        """
        from .import_service import import_transform as service_import

        return service_import(
            self,
            path,
            source_frame_name=source_frame_name,
            target_frame_name=target_frame_name,
        )

    def import_image(self, path: str) -> tuple[Image, str]:
        """Import an MRI image from file into the session.

        Routes to the import service, which creates subject-specific MRI and
        voxel frames based on the filename and registers affine transforms.
        """
        from .import_service import import_image as service_import

        return service_import(self, path)
