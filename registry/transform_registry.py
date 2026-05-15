"""Transform registry for named transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping

import json
import numpy as np

from core.frames import CoordinateFrame
from core.transform import Transform


@dataclass
class TransformRegistry:
	"""Registry for named transforms."""

	_transforms: Dict[str, Transform] = field(default_factory=dict)

	def register_transform(self, name: str, transform: Transform) -> None:
		if name in self._transforms:
			raise ValueError(f"Transform '{name}' already registered")
		self._transforms[name] = transform

	def get_transform(self, name: str) -> Transform:
		try:
			return self._transforms[name]
		except KeyError as exc:
			raise KeyError(f"Transform '{name}' not found") from exc

	def list_transforms(self) -> List[str]:
		return sorted(self._transforms.keys())

	def register_many(self, items: Iterable[tuple[str, Transform]]) -> None:
		for name, transform in items:
			self.register_transform(name, transform)

	def save(self, path: str) -> None:
		data = {
			name: {
				"source": t.source.name,
				"target": t.target.name,
				"matrix": t.matrix.tolist(),
			}
			for name, t in self._transforms.items()
		}
		with open(path, "w", encoding="utf-8") as f:
			json.dump(data, f, indent=2)

	@classmethod
	def load(
		cls,
		path: str,
		frames: Mapping[str, CoordinateFrame],
	) -> "TransformRegistry":
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
		registry = cls()
		for name, payload in data.items():
			source = frames[payload["source"]]
			target = frames[payload["target"]]
			matrix = np.asarray(payload["matrix"], dtype=float)
			registry.register_transform(name, Transform(source, target, matrix))
		return registry


__all__ = ["TransformRegistry"]
