"""Registry for MRI images."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping
from unicodedata import name

from core.frames import CoordinateFrame
from core.image import Image


@dataclass
class ImageRegistry:
    """Registry for imported MRI images."""

    _images: Dict[str, Image] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self._images

    def __len__(self) -> int:
        return len(self._images)
    
    def __iter__(self):
        return iter(self._images.items())

    def items(self):
        """Return registered image items."""
        return self._images.items()

    def add_image(self, name: str, image: Image) -> None:
        """Register an image under a unique name."""
        if name in self._images:
            raise ValueError(f"Image '{name}' already registered")
        self._images[name] = image

    def get_image(self, name: str) -> Image:
        """Retrieve a registered image."""
        try:
            return self._images[name]
        except KeyError as exc:
            raise KeyError(f"Image '{name}' not found") from exc

    def names(self) -> List[str]:
        """Return registered image names."""
        return sorted(self._images.keys())

    def values(self):
        """Return registered image values."""
        return self._images.values()

    def remove_image(self, name: str) -> None:
        """Remove a registered image."""
        try:
            del self._images[name]
        except KeyError as exc:
            raise KeyError(f"Image '{name}' not found") from exc

    def replace_image(self, name: str, image: Image) -> None:
        """Replace an existing registered image."""
        if name not in self._images:
            raise KeyError(f"Image '{name}' not found")

        self._images[name] = image

    def to_dict(self) -> Dict[str, dict]:
        """Return a dictionary representation of the registry."""
        return {name: image.to_dict() for name, image in self._images.items()}

    @classmethod
    def from_dict(
        cls,
        data: dict,
        frames: Mapping[str, CoordinateFrame],
    ) -> "ImageRegistry":
        registry = cls()

        for name, image_data in data.items():
            image = Image.from_dict(
                image_data,
                frames,
            )
            registry.add_image(name, image)

        return registry


__all__ = ["ImageRegistry"]