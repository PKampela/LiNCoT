"""GUI viewer components."""

from __future__ import annotations

from importlib import import_module

__all__ = ["BaseViewer", "SurfaceViewer", "ViewerManager", "ViewerTab", "VolumeViewer"]

_MODULES = {
	"BaseViewer": "base_viewer",
	"SurfaceViewer": "surface_viewer",
	"ViewerManager": "viewer_manager",
	"ViewerTab": "viewer_tab",
	"VolumeViewer": "volume_viewer",
}


def __getattr__(name: str):
	module_name = _MODULES.get(name)
	if module_name is None:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

	module = import_module(f"{__name__}.{module_name}")
	value = getattr(module, name)
	globals()[name] = value
	return value
