"""NiBabel backend wrapper."""

from __future__ import annotations

from typing import cast

import numpy as np
from nibabel import as_closest_canonical
from nibabel.nifti1 import Nifti1Image

from core.frames import CoordinateFrame
from core.image import Image
from core.transform import Transform


def _load_canonical(path: str) -> Nifti1Image:
    """
    Load a NIfTI image and convert it to canonical RAS voxel order.

    All functions in this backend operate on canonical images so that
    voxel ordering is consistent regardless of how the file was stored.
    """
    from nibabel import loadsave

    img = cast(Nifti1Image, loadsave.load(path))
    return as_closest_canonical(img)


def load_nifti_image(
    path: str,
    frame: CoordinateFrame,
) -> Image:
    """
    Load a NIfTI image in canonical RAS voxel order.
    """

    img = _load_canonical(path)

    return Image(
        data=np.asarray(img.dataobj),
        affine=np.asarray(img.affine, dtype=float),
        frame=frame,
    )


def voxel_to_world_transform(
    image: Image,
    voxel_frame: CoordinateFrame,
    world_frame: CoordinateFrame,
) -> Transform:
    """
    Create the voxel -> world transform for an imported image.

    Since Image.affine already represents the canonical voxel-to-world
    transform, this simply wraps it as a Transform object.
    """

    return Transform(
        source=voxel_frame,
        target=world_frame,
        matrix=image.affine.copy(),
    )


def world_to_voxel_transform(
    image: Image,
    voxel_frame: CoordinateFrame,
    world_frame: CoordinateFrame,
) -> Transform:
    """
    Create the world -> voxel transform for an imported image.
    """

    return voxel_to_world_transform(
        image,
        voxel_frame,
        world_frame,
    ).invert()