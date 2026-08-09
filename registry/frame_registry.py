"""Frame registry for known coordinate frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from core.frames import CoordinateFrame


@dataclass
class FrameRegistry:
	"""Registry for coordinate frames.

	This registry is intentionally explicit and contains no global state.
	"""

	_frames: Dict[str, CoordinateFrame] = field(default_factory=dict)

	def __contains__(self, name: str) -> bool:
		return name in self._frames
	
	def __len__(self) -> int:
		return len(self._frames)
	
	def __iter__(self):
		return iter(self._frames.items())

	def items(self):
		"""Return registered frame items."""
		return self._frames.items()

	def add_frame(self, frame: CoordinateFrame) -> None:
		if frame.name in self._frames:
			raise ValueError(f"Frame '{frame.name}' already registered")
		self._frames[frame.name] = frame

	def get_frame(self, name: str) -> CoordinateFrame:
		try:
			return self._frames[name]
		except KeyError as exc:
			raise KeyError(f"Frame '{name}' not found") from exc

	def names(self) -> List[str]:
		return sorted(self._frames.keys())

	def add_many(self, frames: Iterable[CoordinateFrame]) -> None:
		for frame in frames:
			self.add_frame(frame)

	def remove_frame(self, name: str) -> None:
		try:
			del self._frames[name]
		except KeyError as exc:
			raise KeyError(f"Frame '{name}' not found") from exc
		
	def get_all(self) -> dict[str, CoordinateFrame]:
		return self._frames.copy()	

	def values(self):
		return self._frames.values()

	def to_dict(self) -> Dict[str, dict]:
		return {name: frame.to_dict() for name, frame in self._frames.items()}

	@classmethod
	def from_dict(cls, data: Dict[str, dict]) -> "FrameRegistry":
		registry = cls()
		for name, payload in data.items():
			frame = CoordinateFrame.from_dict(payload)
			registry.add_frame(frame)
		return registry



__all__ = ["FrameRegistry"]
