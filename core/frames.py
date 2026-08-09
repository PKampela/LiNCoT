"""Coordinate frame definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple




@dataclass(frozen=True, eq=False)
class CoordinateFrame:
    """Represents a 3D coordinate frame.

    Parameters
    ----------
    name
        Human-readable frame name.
    axes
        Tuple of axis labels (e.g., ("R", "A", "S")).
    units
        Units for coordinates (e.g., "mm").
    coordinate_system
        Optional coordinate system name. Such as World or Voxel.
    description
        Optional longer description.
    """

    name: str
    axes: Tuple[str, str, str]
    units: str
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.axes) != 3:
            raise ValueError("axes must contain exactly three axis labels")
        if any(not axis or not isinstance(axis, str) for axis in self.axes):
            raise ValueError("axes must be non-empty strings")
        if not self.name or not isinstance(self.name, str):
            raise ValueError("name must be a non-empty string")
        if not self.units or not isinstance(self.units, str):
            raise ValueError("units must be a non-empty string")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoordinateFrame):
            return False
        return (self.name, self.axes, self.units) == (other.name, other.axes, other.units)

    def __hash__(self) -> int:
        return hash((self.name, self.axes, self.units))

    
    def to_dict(self) -> dict:
        """Convert the frame to a dictionary representation."""
        return {
            "name": self.name,
            "axes": self.axes,
            "units": self.units,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CoordinateFrame:
        """Create a CoordinateFrame from a dictionary representation."""
        return cls(
            name=data["name"],
            axes=tuple(data["axes"]),
            units=data["units"],
            description=data.get("description"),
        )
