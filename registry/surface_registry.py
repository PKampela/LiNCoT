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

	def register_surface(self, name: str, surface: Surface) -> None:
		if name in self._surfaces:
			raise ValueError(f"Surface '{name}' already registered")
		self._surfaces[name] = surface

	def get_surface(self, name: str) -> Surface:
		try:
			return self._surfaces[name]
		except KeyError as exc:
			raise KeyError(f"Surface '{name}' not found") from exc

	def list_surfaces(self) -> List[str]:
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

	def list_headmodels(self) -> List[str]:
		return sorted(self._headmodels.keys())

	def register_headmodel_surfaces(self, prefix: str, model: HeadModel) -> None:
		for role, surface in model.surfaces.items():
			name = f"{prefix}_{role.value}"
			self.register_surface(name, surface)


__all__ = ["SurfaceRegistry"]
