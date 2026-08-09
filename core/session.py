from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List
from uuid import uuid4

from .project import ProjectMetadata
from .frames import CoordinateFrame
from .point import Point
from .transform import Transform
from .chain import TransformChain
from .image import Image
from .surface import HeadModel, Surface
    
from registry.frame_registry import FrameRegistry
from registry.transform_registry import TransformRegistry
from registry.surface_registry import SurfaceRegistry
from registry.image_registry import ImageRegistry
from registry.point_registry import PointRegistry

@dataclass
class SubjectMetadata:

    subject_id: str | None = None

    description: str | None = None

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SubjectMetadata:
        return cls(
            subject_id=data.get("subject_id"),
            description=data.get("description"),
        )


@dataclass
class Session:

    project: ProjectMetadata = field(
        default_factory=ProjectMetadata
    )

    subject: SubjectMetadata = field(
        default_factory=SubjectMetadata
    )

    images: ImageRegistry = field(
        default_factory=ImageRegistry
    )

    points: PointRegistry = field(
        default_factory=PointRegistry
    )

    frames: FrameRegistry = field(
        default_factory=FrameRegistry
    )

    transforms: TransformRegistry = field(
        default_factory=TransformRegistry
    )

    surfaces: SurfaceRegistry = field(
        default_factory=SurfaceRegistry
    )

    load_warnings: list[str] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    def add_image(self, name: str, image: Image) -> None:
        self.images.add_image(name, image)

    def get_image(self, name: str) -> Image:
        return self.images.get_image(name)

    def list_images(self) -> List[str]:
        return self.images.names()

    def remove_image(self, name: str) -> None:
        self.images.remove_image(name)

    # ------------------------------------------------------------------
    # Point management
    # ------------------------------------------------------------------

    def add_point(self, name: str, point: Point) -> None:
        self.points.add_point(name, point)

    def get_point(self, name: str) -> Point:
        return self.points.get_point(name)

    def list_points(self) -> List[str]:
        return self.points.names()

    def remove_point(self, name: str) -> None:
        self.points.remove_point(name)

    def rename_point(self, old: str, new: str) -> None:
        self.points.rename_point(old, new)

    def has_point(self, name: str) -> bool:
        return name in self.points

    def create_transformed_point(
    self,
    name: str,
    point: Point,
    ) -> Point:
        """Register a new transformed point.

        The input point should already contain the desired coordinates
        and target frame.
        """
        self.points.add_point(name, point)
        return point

    # ------------------------------------------------------------------
    # Coordinate frame management
    # ------------------------------------------------------------------

    def add_frame(self, frame: CoordinateFrame) -> None:
        self.frames.add_frame(frame)

    def get_frame(self, name: str) -> CoordinateFrame:
        return self.frames.get_frame(name)

    def list_frames(self) -> List[str]:
        return self.frames.names()
    
    def remove_frame(self, name: str) -> None:
        self.frames.remove_frame(name)

    # ------------------------------------------------------------------
    # Transform management
    # ------------------------------------------------------------------

    def add_transform(self, name: str, transform: Transform) -> None:
        self.transforms.register_transform(name, transform)

    def get_transform(self, name: str) -> Transform:
        return self.transforms.get_transform(name)

    def list_transforms(self) -> List[str]:
        return self.transforms.names()

    def get_transform_chain(self, names: List[str]) -> TransformChain:
        transforms = [self.get_transform(name) for name in names]
        return TransformChain(transforms)

    def apply_transform_to_point(
        self,
        point_name: str,
        chain_names: List[str],
    ) -> Point:
        point = self.get_point(point_name)
        chain = self.get_transform_chain(chain_names)
        return chain.apply(point)
    
    def remove_transform(self, name: str) -> None:
        self.transforms.remove_transform(name)

    # ------------------------------------------------------------------
    # Surface management
    # ------------------------------------------------------------------

    def add_surface(self, name: str, surface: Surface) -> None:
        self.surfaces.register_surface(name, surface)

    def get_surface(self, name: str) -> Surface:
        return self.surfaces.get_surface(name)

    def list_surfaces(self) -> List[str]:
        return self.surfaces.names_surfaces()
    
    def remove_surface(self, name: str) -> None:
        self.surfaces.remove_surface(name)

    def add_headmodel(self, name: str, model: HeadModel) -> None:
        self.surfaces.register_headmodel(name, model)

    def get_headmodel(self, name: str) -> HeadModel:
        return self.surfaces.get_headmodel(name)
    
    def list_headmodels(self) -> List[str]:
        return self.surfaces.names_headmodels()
    
    def remove_headmodel(self, name: str) -> None:
        self.surfaces.remove_headmodel(name)
        

    # ------------------------------------------------------------------
    # Import interface
    # ------------------------------------------------------------------

    def import_transform(
        self,
        path: Path,
        source_frame_name: str | None = None,
        target_frame_name: str | None = None,
    ) -> tuple[Transform, str]:
        """Import a transform from file into the session."""
        from .import_service import import_transform

        return import_transform(
            self,
            path,
            source_frame_name=source_frame_name,
            target_frame_name=target_frame_name,
        )

    def import_image(self, path: Path) -> tuple[Image, str]:
        """Import an MRI image into the session."""
        from .import_service import import_image

        return import_image(self, path)

    def import_surface(
        self,
        path: Path,
        frame_name: str | None = None,
        surface_name: str | None = None,
    ) -> tuple[Surface, str, str]:
        """Import a cortical or anatomical surface into the session.

        Returns:
            Tuple containing:
            - imported Surface object
            - registered surface name
            - information message
        """
        from .import_service import import_surface

        return import_surface(
            self,
            path,
            frame_name=frame_name,
            surface_name=surface_name,
        )
    
    def import_images(
        self,
        paths: list[Path],
    ) -> list[tuple[Image, str]]:
        """Import multiple MRI images."""

        from .import_service import import_multiple_images

        return import_multiple_images(
            self,
            paths,
        )


    def import_surfaces(
        self,
        paths: list[Path],
        frame_names: list[str] | None = None,
        surface_names: list[str] | None = None,
    ) -> list[tuple[Surface, str, str]]:
        """Import multiple surfaces."""

        from .import_service import import_multiple_surfaces

        return import_multiple_surfaces(
            self,
            paths,
            frame_names=frame_names,
            surface_names=surface_names,
        )


    def import_transforms(
        self,
        paths: list[Path],
        source_frame_name: str | None = None,
        target_frame_name: str | None = None,
    ) -> list[tuple[Transform, str]]:
        """Import multiple transforms."""

        from .import_service import import_multiple_transforms

        return import_multiple_transforms(
            self,
            paths,
            source_frame_name=source_frame_name,
            target_frame_name=target_frame_name,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self):

        return {
            "project": self.project.to_dict(),

            "subject": self.subject.to_dict(),

            "frames": self.frames.to_dict(),

            "transforms": self.transforms.to_dict(),

            "images": self.images.to_dict(),

            "surfaces": self.surfaces.to_dict(),

            "points": self.points.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":

        project = ProjectMetadata.from_dict(data.get("project", {}))

        subject = SubjectMetadata.from_dict(data.get("subject", {}))

        frames = FrameRegistry.from_dict(data.get("frames", {}))

        transforms = TransformRegistry.from_dict(data.get("transforms", {}), frames.get_all())

        images = ImageRegistry.from_dict(data.get("images", {}),frames.get_all(),)

        surfaces = SurfaceRegistry.from_dict(data.get("surfaces", {}))

        points = PointRegistry.from_dict(data.get("points", {}))

        return cls(
            project=project,
            subject=subject,
            frames=frames,
            transforms=transforms,
            images=images,
            surfaces=surfaces,
            points=points,
        )

    @staticmethod
    def create_empty_session() -> Session:
        """Create an empty session with default project and subject metadata."""
        return Session(
            project=ProjectMetadata(
                name=f"Untitled-{uuid4().hex[:8]}",
                project_path=None,
            ),
            subject=SubjectMetadata(
                subject_id="unknown",
            ),
            frames=FrameRegistry(),
            transforms=TransformRegistry(),
            images=ImageRegistry(),
            surfaces=SurfaceRegistry(),
            points=PointRegistry(),
        )

    