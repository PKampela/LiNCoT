"""Surface registry for complex multisurface objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from core.surface import HeadModel, Surface


@dataclass
class SurfaceRegistry:
	"""Registry for surfaces and head models."""

	_surfaces: Dict[str, Surface] = field(default_factory=dict)
	_headmodels: Dict[str, HeadModel] = field(default_factory=dict)

	def __contains__(self, name: str) -> bool:
		return name in self._surfaces or name in self._headmodels
	
	def __len__(self) -> int:
		return len(self._surfaces) + len(self._headmodels)
	
	def __iter__(self):
		for name, surface in self._surfaces.items():
			yield name, surface
		for name, model in self._headmodels.items():
			yield name, model

	def items(self):
		"""Return registered surface and head model items."""
		for name, surface in self._surfaces.items():
			yield name, surface
		for name, model in self._headmodels.items():
			yield name, model

	def register_surface(self, name: str, surface: Surface) -> None:
		if name in self._surfaces:
			raise ValueError(f"Surface '{name}' already registered")
		self._surfaces[name] = surface

	def get_surface(self, name: str) -> Surface:
		try:
			return self._surfaces[name]
		except KeyError as exc:
			raise KeyError(f"Surface '{name}' not found") from exc
		
	def names_all(self) -> List[str]:
		return sorted(list(self._surfaces.keys()) + list(self._headmodels.keys()))

	def names_surfaces(self) -> List[str]:
		return sorted(self._surfaces.keys())

	def register_headmodel(self, name: str, model: HeadModel) -> None:
		if name in self._headmodels:
			raise ValueError(f"HeadModel '{name}' already registered")
		self._headmodels[name] = model

	def get_headmodel(self, name: str) -> HeadModel:
		try:
			return self._headmodels[name]
		except KeyError as exc:
			raise KeyError(f"HeadModel '{name}' not found") from exc

	def names_headmodels(self) -> List[str]:
		return sorted(self._headmodels.keys())

	def register_headmodel_surfaces(self, prefix: str, model: HeadModel) -> None:
		for role, surface in model.surfaces.items():
			name = f"{prefix}_{role.value}"
			self.register_surface(name, surface)

	def remove_surface(self, name: str) -> None:
		try:
			del self._surfaces[name]
		except KeyError as exc:
			raise KeyError(f"Surface '{name}' not found") from exc
		
	def remove_headmodel(self, name: str) -> None:
		try:
			del self._headmodels[name]
		except KeyError as exc:
			raise KeyError(f"HeadModel '{name}' not found") from exc

	def values(self):
		return list(self._surfaces.values()) + list(self._headmodels.values())

	def to_dict(self) -> Dict[str, dict]:
		return {
			"surfaces": {name: surface.to_dict() for name, surface in self._surfaces.items()},
		}

	@classmethod
	def from_dict(cls, data: Dict[str, dict]) -> "SurfaceRegistry":
		registry = cls()
		for name, payload in data.get("surfaces", {}).items():
			surface = Surface.from_dict(payload)
			registry.register_surface(name, surface)
		return registry


__all__ = ["SurfaceRegistry"]
