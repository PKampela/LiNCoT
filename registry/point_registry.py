"""Registry for coordinate points."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import name
from typing import Dict, List

from core.point import Point


@dataclass
class PointRegistry:
    """Registry for named coordinate points."""

    _points: Dict[str, Point] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self._points
    
    def __len__(self) -> int:
        return len(self._points)
    
    def __iter__(self):
        return iter(self._points.items())

    def items(self):
        """Return registered point items."""
        return self._points.items()
    
    def unique_name(self, base_name: str) -> str:
        if base_name not in self._points:
            return base_name

        counter = 1
        while f"{base_name}_{counter}" in self._points:
            counter += 1

        return f"{base_name}_{counter}"

    def add_point(self, name: str, point: Point) -> None:
        """Register a point under a unique name."""
        name = self.unique_name(name)
        self._points[name] = point

    def get_point(self, name: str) -> Point:
        """Retrieve a registered point."""
        try:
            return self._points[name]
        except KeyError as exc:
            raise KeyError(f"Point '{name}' not found") from exc

    def names(self) -> List[str]:
        """Return registered point names."""
        return sorted(self._points.keys())

    def values(self):
        """Return registered point values."""
        return self._points.values()

    def remove_point(self, name: str) -> None:
        """Remove a registered point."""
        try:
            del self._points[name]
        except KeyError as exc:
            raise KeyError(f"Point '{name}' not found") from exc

    def rename_point(self, old_name: str, new_name: str) -> None:
        """Rename a registered point."""
        if old_name not in self._points:
            raise KeyError(f"Point '{old_name}' not found")
        if new_name in self._points:
            raise ValueError(f"Point '{new_name}' already registered")

        self._points[new_name] = self._points.pop(old_name)

    def to_dict(self) -> Dict[str, dict]:
        """Return a dictionary representation of the registry."""
        return {name: point.to_dict() for name, point in self._points.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, dict]) -> "PointRegistry":
        """Create a registry from a dictionary representation."""
        registry = cls()
        for name, payload in data.items():
            point = Point.from_dict(payload)
            registry.register_point(name, point)
        return registry


__all__ = ["PointRegistry"]