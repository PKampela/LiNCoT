"""Backward-compatible import for viewer components."""

from .viewer.base_viewer import BaseViewer

ViewerPanel = BaseViewer

__all__ = ["BaseViewer", "ViewerPanel"]