"""Image representation and resampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np
from scipy.ndimage import affine_transform

from core.asset import AssetMetadata

from .frames import CoordinateFrame
from .transform import Transform


@dataclass(frozen=True)
class Image:
    """Represents a 2D/3D image with an affine and coordinate frame."""

    data: np.ndarray | None
    affine: np.ndarray
    voxel_frame: CoordinateFrame
    world_frame: CoordinateFrame
    asset: AssetMetadata | None = None
    shape_metadata: Tuple[int, ...] | None = None  # Optional shape metadata for images without data

    def __post_init__(self) -> None:
        if self.data is not None:
            data_array = np.asarray(self.data)

            if data_array.ndim not in (2, 3):
                raise ValueError("data must be a 2D or 3D array")

            object.__setattr__(self, "data", data_array)

        affine_array = _normalize_affine(
            self.affine,
            self.data.ndim if self.data is not None else 3,
        )

        object.__setattr__(self, "affine", affine_array)

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.data is not None:
            return self.data.shape
        if self.shape_metadata is not None:
            return self.shape_metadata
        raise ValueError("Image has no shape information")

    @property
    def ndim(self) -> int:
        return self.data.ndim if self.data is not None else 3

    def to_dict(self) -> dict:
        """Convert image metadata to a dictionary representation."""
        return {
            "shape_metadata": list(self.shape),
            "affine": self.affine.tolist(),
            "voxel_frame": self.voxel_frame.name,
            "world_frame": self.world_frame.name,
            "asset": self.asset.to_dict() if self.asset else None,
        }
    
    @classmethod
    def from_dict(
        cls,
        data: dict,
        frames: Mapping[str, CoordinateFrame],
    ) -> "Image":
        """Reconstruct image metadata without loading voxel data."""

        voxel_frame_name = data["voxel_frame"]
        world_frame_name = data["world_frame"]

        try:
            voxel_frame = frames[voxel_frame_name]
        except KeyError as exc:
            raise ValueError(
                f"Image references unknown voxel frame "
                f"'{voxel_frame_name}'"
            ) from exc

        try:
            world_frame = frames[world_frame_name]
        except KeyError as exc:
            raise ValueError(
                f"Image references unknown world frame "
                f"'{world_frame_name}'"
            ) from exc

        asset = (
            AssetMetadata.from_dict(data["asset"])
            if data.get("asset")
            else None
        )

        return cls(
            data=None,
            affine=np.asarray(data["affine"], dtype=float),
            voxel_frame=voxel_frame,
            world_frame=world_frame,
            asset=asset,
            shape_metadata=tuple(data["shape_metadata"]),
        )


def _normalize_affine(affine: np.ndarray, ndim: int) -> np.ndarray:
    affine_array = np.asarray(affine, dtype=float)
    if affine_array.shape == (4, 4):
        return affine_array
    if ndim == 2 and affine_array.shape == (3, 3):
        normalized = np.eye(4, dtype=float)
        normalized[:2, :2] = affine_array[:2, :2]
        normalized[:2, 3] = affine_array[:2, 2]
        return normalized
    raise ValueError("affine must be 4x4 (or 3x3 for 2D images)")


def _matrix_and_offset(affine: np.ndarray, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    return affine[:ndim, :ndim], affine[:ndim, 3]


def transform_image(
    image: Image,
    transform: Transform,
    order: int = 1,
    output_shape: Optional[Tuple[int, ...]] = None,
    output_affine: Optional[np.ndarray] = None,
    output_frame: Optional[CoordinateFrame] = None,
) -> Image:
    """Resample an image using an affine transform.

    Parameters
    ----------
    image
        Source image to transform.
    transform
        Transform from the image frame to a target frame.
    order
        Interpolation order (0=nearest, 1=linear).
    output_shape
        Output shape; defaults to input shape.
    output_affine
        Output affine (voxel -> target frame). Defaults to the transformed affine.
    output_frame
        Target frame for the output image. Defaults to the transform target.
    """

    if image.voxel_frame != transform.source:
        raise ValueError(
            f"Image voxel frame '{image.voxel_frame.name}' does not match transform source '{transform.source.name}'"
        )

    if output_shape is None:
        output_shape = image.shape
    if len(output_shape) != image.ndim:
        raise ValueError("output_shape must match image dimensionality")

    input_affine = _normalize_affine(image.affine, image.ndim)
    target_affine = _normalize_affine(
        output_affine if output_affine is not None else (transform.matrix @ input_affine),
        image.ndim,
    )

    voxel_to_voxel = np.linalg.inv(input_affine) @ np.linalg.inv(transform.matrix) @ target_affine
    matrix, offset = _matrix_and_offset(voxel_to_voxel, image.ndim)

    data = affine_transform(
        image.data,
        matrix=matrix,
        offset=offset.tolist(),
        output_shape=output_shape,
        order=order,
        mode="constant",
        cval=0.0,
    )

    return Image(
        data=data,
        affine=target_affine,
        voxel_frame=output_frame if output_frame is not None else transform.target,
        world_frame=transform.target,
    )
