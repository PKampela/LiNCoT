import numpy as np
from scipy.ndimage import map_coordinates
from .image import Image


def sample_volume(
    image: Image,
    points_world: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """
    Sample image values at arbitrary world-space coordinates.

    Parameters
    ----------
    image : Image
        Source image.
    points_world : (N, 3) array
        Coordinates in image.frame space.
    order : int
        Interpolation order (0=nearest, 1=linear).

    Returns
    -------
    values : (N,) array
        Interpolated values.
    """

    points_world = np.asarray(points_world, dtype=float)

    if points_world.ndim == 1:
        points_world = points_world[None, :]

    # Convert to homogeneous
    ones = np.ones((points_world.shape[0], 1))
    points_h = np.hstack([points_world, ones])

    # World → voxel
    world_to_voxel = np.linalg.inv(image.affine)
    voxel_coords = (world_to_voxel @ points_h.T).T[:, :3]

    # map_coordinates expects coordinates as separate arrays
    coords = [voxel_coords[:, d] for d in range(image.ndim)]

    values = map_coordinates(
        image.data,
        coords,
        order=order,
        mode="constant",
        cval=0.0,
    )

    return values