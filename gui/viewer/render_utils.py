"""Conversions from domain objects to PyVista renderable data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyvista as pv
from nibabel.orientations import aff2axcodes

from core.image import Image
from core.surface import Surface


@dataclass(frozen=True)
class AnatomicalAxisInfo:
    """Display mapping from an anatomical axis to the underlying voxel axis."""

    group: str
    voxel_axis: int
    code: str
    sign: int


def surface_to_polydata(surface: Surface) -> pv.PolyData:
    """Convert a triangular surface mesh to PyVista polydata."""

    vertices = np.asarray(surface.vertices, dtype=float)
    faces = np.asarray(surface.faces, dtype=np.int64)
    face_array = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]
    ).ravel()
    mesh = pv.PolyData(vertices, face_array)
    mesh.cell_data["surface_face_id"] = np.arange(faces.shape[0], dtype=np.int32)
    mesh.compute_normals(
        point_normals=True,
        cell_normals=True,
        auto_orient_normals=True,
        consistent_normals=True,
        inplace=True,
    )
    try:
        mesh.point_data["mean_curvature"] = mesh.curvature(curv_type="mean")
    except Exception:
        pass
    return mesh


def affine_origin_spacing(image: Image) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Extract a best-effort origin and spacing from an image affine."""

    affine = np.asarray(image.affine, dtype=float)
    origin = (
        float(affine[0, 3]),
        float(affine[1, 3]),
        float(affine[2, 3]),
    )
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    spacing = np.where(spacing == 0.0, 1.0, spacing)
    return origin, (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def voxel_to_world(image: Image, voxel_coords: tuple[float, float, float]) -> tuple[float, float, float]:
    """Map voxel coordinates to world coordinates using the image affine."""

    point = np.asarray([voxel_coords[0], voxel_coords[1], voxel_coords[2], 1.0], dtype=float)
    world = np.asarray(image.affine, dtype=float) @ point
    return (float(world[0]), float(world[1]), float(world[2]))


def world_to_voxel(image: Image, world_coords: tuple[float, float, float]) -> tuple[float, float, float]:
    """Map world coordinates to voxel coordinates using the inverse affine."""

    affine_inv = np.linalg.inv(np.asarray(image.affine, dtype=float))
    point = np.asarray([world_coords[0], world_coords[1], world_coords[2], 1.0], dtype=float)
    voxel = affine_inv @ point
    return (float(voxel[0]), float(voxel[1]), float(voxel[2]))


def image_orientation(image: Image) -> str:
    """Return the three-letter orientation code for an image affine."""

    return "".join(aff2axcodes(np.asarray(image.affine, dtype=float)))


def anatomical_axis_info(image: Image) -> dict[str, AnatomicalAxisInfo]:
    """Map anatomical LR/AP/SI groups to voxel axes from the image affine."""

    codes = aff2axcodes(np.asarray(image.affine, dtype=float))
    mapping: dict[str, AnatomicalAxisInfo] = {}

    for voxel_axis, code in enumerate(codes):
        if code in {"L", "R"}:
            group = "lr"
            sign = 1 if code == "R" else -1
        elif code in {"P", "A"}:
            group = "ap"
            sign = 1 if code == "A" else -1
        elif code in {"I", "S"}:
            group = "si"
            sign = 1 if code == "S" else -1
        else:
            continue

        mapping[group] = AnatomicalAxisInfo(
            group=group,
            voxel_axis=voxel_axis,
            code=str(code),
            sign=sign,
        )

    # Fallback for uncommon/degenerate affines: keep predictable identity mapping.
    if "lr" not in mapping:
        mapping["lr"] = AnatomicalAxisInfo(group="lr", voxel_axis=0, code="R", sign=1)
    if "ap" not in mapping:
        fallback_axis = 1 if mapping["lr"].voxel_axis != 1 else 2
        mapping["ap"] = AnatomicalAxisInfo(group="ap", voxel_axis=fallback_axis, code="A", sign=1)
    if "si" not in mapping:
        used = {mapping["lr"].voxel_axis, mapping["ap"].voxel_axis}
        fallback_axis = 0
        for candidate in (0, 1, 2):
            if candidate not in used:
                fallback_axis = candidate
                break
        mapping["si"] = AnatomicalAxisInfo(group="si", voxel_axis=fallback_axis, code="S", sign=1)

    return mapping


def voxel_size(image: Image) -> tuple[float, float, float]:
    """Return voxel spacing inferred from the affine basis vectors."""

    affine = np.asarray(image.affine, dtype=float)
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    spacing = np.where(spacing == 0.0, 1.0, spacing)
    return (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def image_world_bounds(image: Image) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Compute axis-aligned world bounds from the image corners."""

    shape = image.shape
    corners = np.asarray(
        [
            voxel_to_world(image, (x, y, z))
            for x in (0, shape[0] - 1)
            for y in (0, shape[1] - 1)
            for z in (0, shape[2] - 1)
        ],
        dtype=float,
    )
    return (
        (float(corners[:, 0].min()), float(corners[:, 0].max())),
        (float(corners[:, 1].min()), float(corners[:, 1].max())),
        (float(corners[:, 2].min()), float(corners[:, 2].max())),
    )


def image_to_uniform_grid(image: Image, scalar_name: str = "values") -> pv.ImageData:
    """Convert an MRI volume to a PyVista image grid."""

    if image.ndim != 3:
        raise ValueError("Only 3D images can be rendered in the volume viewer")

    origin, spacing = affine_origin_spacing(image)
    grid = pv.ImageData(dimensions=image.shape)
    grid.origin = origin
    grid.spacing = spacing
    grid.point_data[scalar_name] = np.asarray(image.data, dtype=float).ravel(order="F")
    grid.set_active_scalars(scalar_name)
    return grid


def orthogonal_slice_data(image: Image, axis: int, index: int) -> np.ndarray:
    """Return a 2D orthogonal slice from a 3D image."""

    if image.ndim != 3:
        raise ValueError("Only 3D images can be sliced orthogonally")
    if axis == 0:
        return np.asarray(image.data[index, :, :])
    if axis == 1:
        return np.asarray(image.data[:, index, :])
    if axis == 2:
        return np.asarray(image.data[:, :, index])
    raise ValueError("axis must be 0, 1, or 2")


def slider_value_to_world(image: Image, axis: int, index: int) -> float:
    """Map a voxel index to the corresponding world coordinate along one axis."""

    voxel_coords = [0.0, 0.0, 0.0]
    voxel_coords[axis] = float(index)
    coords = (float(voxel_coords[0]), float(voxel_coords[1]), float(voxel_coords[2]))
    return voxel_to_world(image, coords)[axis]