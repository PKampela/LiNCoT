from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Tuple
from enum import Enum
from scipy.spatial.ckdtree import cKDTree
import numpy as np

from .frames import CoordinateFrame
from .transform import Transform

@dataclass(frozen=True)
class Surface:
    
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray     # (M, 3)
    frame: CoordinateFrame
    metadata: dict | None = None
    _kdtree: cKDTree | None = None

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be an (N, 3) array")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be an (M, 3) array")

    def _get_kdtree(self) -> cKDTree:
        tree = self._kdtree
        if tree is None:
            if cKDTree is None:
                raise RuntimeError("scipy.spatial.cKDTree is unavailable")
            tree = cKDTree(self.vertices)
            object.__setattr__(self, "_kdtree", tree)
        return tree
        
    def transform_surface(self, transform: Transform) -> Surface:
        """Apply a spatial transform to the surface."""
        transformed_vertices = np.array([transform.apply(v) for v in self.vertices])
        return Surface(
            vertices=transformed_vertices,
            faces=self.faces.copy(),
            frame=self.frame,
            metadata=self.metadata.copy() if self.metadata else None
        )
    
    def face_normals(self) -> np.ndarray:
        """Compute the normal vector for each face."""
        v1 = self.vertices[self.faces[:, 0]]
        v2 = self.vertices[self.faces[:, 1]]
        v3 = self.vertices[self.faces[:, 2]]
        normals = np.cross(v2 - v1, v3 - v1)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.where(norms == 0, 1, norms)  # Avoid division by zero

    def closest_vertex(self, point: np.ndarray) -> Tuple[int, float]:
        """Find the index of the closest vertex to a given point."""
        distance, closest_index = self._get_kdtree().query(point)
        return int(closest_index), float(distance)
    
    def distance_to_surface(self, point: np.ndarray) -> float:
        """Compute the distance from a point to the surface."""
        _, distance = self.closest_vertex(point)
        return distance
    
    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the axis-aligned bounding box of the surface."""
        min_coords = np.min(self.vertices, axis=0)
        max_coords = np.max(self.vertices, axis=0)
        return min_coords, max_coords
    
class SurfaceRole(str, Enum):
    SCALP = "scalp"
    OUTER_SKULL = "outer_skull"
    INNER_SKULL = "inner_skull"
    LH_PIAL = "lh_pial"
    RH_PIAL = "rh_pial"


@dataclass(frozen=True)
class HeadModel:
    """
    Geometric head model consisting of layered anatomical surfaces.
    """
    surfaces: Dict[SurfaceRole, Surface]

    _MIN_REQUIRED = frozenset({
        SurfaceRole.SCALP,
        SurfaceRole.LH_PIAL,
        SurfaceRole.RH_PIAL,
    })

    def __post_init__(self) -> None:
        # Defensive copy
        object.__setattr__(self, "surfaces", dict(self.surfaces))

        # Validation
        missing = self._MIN_REQUIRED - self.surfaces.keys()
        if missing:
            raise ValueError(f"Missing required surfaces: {missing}")
        for role, surface in self.surfaces.items():
            if not isinstance(surface, Surface):
                raise TypeError(f"Surface for role '{role}' must be a Surface instance")
        
    @property
    def all_surfaces(self) -> Mapping[SurfaceRole, Surface]:
        return MappingProxyType(self.surfaces)
    
    def get_surface(self, role: SurfaceRole) -> Surface:
        try:
            return self.surfaces[role]
        except KeyError as exc:
            raise KeyError(f"Surface role '{role}' not found in head model") from exc

    @staticmethod
    def _closest_vertex_on_surface(surface: Surface, point: np.ndarray) -> Tuple[int, float]:
        # Prefer direct KD-tree query when available, otherwise fall back to the public API.
        if hasattr(surface, "_get_kdtree"):
            distance, closest_index = surface._get_kdtree().query(point)
            return int(closest_index), float(distance)
        return surface.closest_vertex(point)
        
    def project_to_surface(self, point: np.ndarray, role: SurfaceRole) -> Tuple[np.ndarray, float]:
        """Project a point onto a specified surface and return the projected point and distance."""
        surface = self.get_surface(role)
        closest_index, distance = self._closest_vertex_on_surface(surface, point)
        projected_point = surface.vertices[closest_index]
        return projected_point, distance
    
    def projection_normal(self, point: np.ndarray, role: SurfaceRole) -> np.ndarray:
        """Compute the normal vector at the projected point on the surface."""
        surface = self.get_surface(role)
        closest_index, _ = self._closest_vertex_on_surface(surface, point)
        face_indices = np.where(surface.faces == closest_index)[0]
        if len(face_indices) == 0:
            raise ValueError("No faces found for the closest vertex")
        normals = surface.face_normals()[face_indices]
        return np.mean(normals, axis=0)  # Average normal if multiple faces share the vertex
    