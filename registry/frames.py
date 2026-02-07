"""Frame registry for known coordinate frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from ..core.frames import CoordinateFrame


@dataclass
class FrameRegistry:
    """Registry for coordinate frames.

    This registry is intentionally explicit and contains no global state.
    """

    _frames: Dict[str, CoordinateFrame] = field(default_factory=dict)

    def register_frame(self, frame: CoordinateFrame) -> None:
        if frame.name in self._frames:
            raise ValueError(f"Frame '{frame.name}' already registered")
        self._frames[frame.name] = frame

    def get_frame(self, name: str) -> CoordinateFrame:
        try:
            return self._frames[name]
        except KeyError as exc:
            raise KeyError(f"Frame '{name}' not found") from exc

    def list_frames(self) -> List[str]:
        return sorted(self._frames.keys())

    def register_many(self, frames: Iterable[CoordinateFrame]) -> None:
        for frame in frames:
            self.register_frame(frame)


def default_frames() -> list[CoordinateFrame]:
    """Provide common frames without registering them globally."""

    return [
        CoordinateFrame(name="head", axes=("R", "A", "S"), units="mm", description="Head coordinates"),
        CoordinateFrame(name="mri", axes=("R", "A", "S"), units="mm", description="MRI coordinates"),
        CoordinateFrame(name="mni", axes=("R", "A", "S"), units="mm", description="MNI coordinates"),
        CoordinateFrame(name="scanner", axes=("R", "A", "S"), units="mm", description="Scanner coordinates"),
    ]
