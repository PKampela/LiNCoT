"""Transform registry for named transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import name
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import json
import numpy as np

from core.frames import CoordinateFrame
from core.transform import Transform


@dataclass
class TransformRegistry:
	"""Registry for named transforms."""

	_transforms: Dict[str, Transform] = field(default_factory=dict)

	def __contains__(self, name: str) -> bool:
		return name in self._transforms
	
	def __len__(self) -> int:
		return len(self._transforms)
	
	def __iter__(self):
		return iter(self._transforms.items())

	def items(self):
		"""Return registered transform items."""
		return self._transforms.items()

	def register_transform(self, name: str, transform: Transform) -> None:
		if name in self._transforms:
			raise ValueError(f"Transform '{name}' already registered")
		self._transforms[name] = transform

	def get_transform(self, name: str) -> Transform:
		try:
			return self._transforms[name]
		except KeyError as exc:
			raise KeyError(f"Transform '{name}' not found") from exc

	def names(self) -> List[str]:
		return sorted(self._transforms.keys())

	def values(self):
		"""Return registered transform values."""
		return self._transforms.values()

	def register_many(self, items: Iterable[tuple[str, Transform]]) -> None:
		for name, transform in items:
			self.register_transform(name, transform)

	def remove_transform(self, name: str) -> None:
		try:
			del self._transforms[name]
		except KeyError as exc:
			raise KeyError(f"Transform '{name}' not found") from exc

	

	def to_dict(self) -> dict:
		"""Convert the registry to a dictionary representation."""
		return {name: transform.to_dict() for name, transform in self._transforms.items()}

	@classmethod
	def from_dict(cls, data: dict, frames: Mapping[str, CoordinateFrame]) -> "TransformRegistry":
		registry = cls()
		for name, transform_data in data.items():
			transform = Transform.from_dict(transform_data, frames)
			registry.register_transform(name, transform)
		return registry


__all__ = ["TransformRegistry"]
