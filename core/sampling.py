"""Utilities for sampling image intensities in continuous world-space."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from .image import Image
from .transform import Transform


def world_to_voxel(image: Image, points_world: np.ndarray) -> np.ndarray:
    """
    Convert world-space coordinates into voxel coordinates.

    Parameters
    ----------
    image
        Source image.
    points_world
        (..., 3) array of coordinates in image.frame.

    Returns
    -------
    (..., 3) ndarray
        Coordinates expressed in voxel space.
    """
    points = np.asarray(points_world, dtype=float)

    original_shape = points.shape
    points = points.reshape(-1, 3)

    homogeneous = np.column_stack((points, np.ones(len(points))))

    transform = np.linalg.inv(image.affine)
    voxel = (transform @ homogeneous.T).T[:, :3]

    return voxel.reshape(original_shape)


def voxel_to_world(image: Image, points_voxel: np.ndarray) -> np.ndarray:
    """
    Convert voxel coordinates into world coordinates.
    """
    points = np.asarray(points_voxel, dtype=float)

    original_shape = points.shape
    points = points.reshape(-1, 3)

    homogeneous = np.column_stack((points, np.ones(len(points))))

    world = (image.affine @ homogeneous.T).T[:, :3]

    return world.reshape(original_shape)


def sample_image(
    image: Image,
    points_world: np.ndarray,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    return_mask: bool = False,
):
    """
    Sample image values at arbitrary world-space coordinates.

    Parameters
    ----------
    image
        Source image.

    points_world
        (..., 3) coordinates expressed in image.frame.

    order
        Interpolation order.
        0 = nearest neighbour
        1 = trilinear
        3 = cubic spline

    mode
        Boundary handling passed directly to scipy.ndimage.map_coordinates.

        Common choices:
            "constant"
            "nearest"
            "mirror"
            "reflect"
            "wrap"

    cval
        Constant value used when mode="constant".

    return_mask
        If True, also return a boolean mask indicating which sample
        locations lie inside the image volume.

    Returns
    -------
    values
        Sampled image intensities.

    mask (optional)
        Boolean validity mask.
    """

    points = np.asarray(points_world, dtype=float)

    original_shape = points.shape[:-1]
    points = points.reshape(-1, 3)

    voxel = world_to_voxel(image, points)

    coords = [voxel[:, d] for d in range(image.ndim)]

    values = map_coordinates(
        image.data,
        coords,
        order=order,
        mode=mode,
        cval=cval,
    )

    values = values.reshape(original_shape)

    if not return_mask:
        return values

    mask = np.ones(len(voxel), dtype=bool)

    for dim in range(image.ndim):
        mask &= voxel[:, dim] >= 0
        mask &= voxel[:, dim] <= image.shape[dim] - 1

    mask = mask.reshape(original_shape)

    return values, mask


def sample_registration_points(
    moving_image: Image,
    reference_image: Image,
    moving_to_reference: Transform,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
    step: int = 1,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a moving image onto the voxel grid of a reference image.

    Parameters
    ----------
    moving_image
        Image to be sampled.

    reference_image
        Defines the output voxel grid.

    transform
        4×4 affine mapping moving_image world coordinates into
        reference_image world coordinates.

    order
        Interpolation order.

    mode
        Boundary handling passed to scipy.ndimage.map_coordinates.

    cval
        Constant value for out-of-bounds sampling.

    Returns
    -------
    ndarray
        Resampled image with the same shape as reference_image.
    """

    # ------------------------------------------------------------------
    # Generate every voxel coordinate of the reference image
    # ------------------------------------------------------------------

    x = np.arange(0, reference_image.shape[0], step)
    y = np.arange(0, reference_image.shape[1], step)
    z = np.arange(0, reference_image.shape[2], step)

    grid = np.meshgrid(x, y, z, indexing="ij")

    voxel_ref = np.stack(
        [g.ravel() for g in grid],
        axis=1,
    )

    reference_values = reference_image.data[
        voxel_ref[:, 0].astype(int),
        voxel_ref[:, 1].astype(int),
        voxel_ref[:, 2].astype(int),
    ]

    # ------------------------------------------------------------------
    # Convert reference voxels -> reference world coordinates
    # ------------------------------------------------------------------

    world_ref = voxel_to_world(reference_image, voxel_ref)

    # ------------------------------------------------------------------
    # Map reference world -> moving world
    #
    # transform is assumed to map:
    #
    #     moving --> reference
    #
    # therefore invert it.
    # ------------------------------------------------------------------

    inverse = np.linalg.inv(moving_to_reference.matrix)

    homogeneous = np.column_stack(
        (world_ref, np.ones(len(world_ref)))
    )

    moving_world = (inverse @ homogeneous.T).T[:, :3]

    moving_voxel = world_to_voxel(moving_image, moving_world)

    inside = (
        (moving_voxel[:, 0] >= 0) &
        (moving_voxel[:, 0] < moving_image.shape[0]) &
        (moving_voxel[:, 1] >= 0) &
        (moving_voxel[:, 1] < moving_image.shape[1]) &
        (moving_voxel[:, 2] >= 0) &
        (moving_voxel[:, 2] < moving_image.shape[2])
    )


    if debug:

        print("\nSampling diagnostics")

        print(
            f"Inside moving image: "
            f"{inside.sum()} / {len(inside)} "
            f"({100*inside.mean():.1f}%)"
        )

        print(
            "Moving voxel bounds:"
        )

        print(
            moving_voxel.min(axis=0),
            moving_voxel.max(axis=0),
        )

        print(
            "Reference world bounds:"
        )

        print(
            world_ref.min(axis=0),
            world_ref.max(axis=0),
        )

        print(
            "Moving world bounds:"
        )

        print(
            moving_world.min(axis=0),
            moving_world.max(axis=0),
        )

    # ------------------------------------------------------------------
    # Sample moving image
    # ------------------------------------------------------------------

    sampled = sample_image(
        moving_image,
        moving_world,
        order=order,
        mode=mode,
        cval=cval,
    )

    return reference_values, sampled.reshape(len(x), len(y), len(z))