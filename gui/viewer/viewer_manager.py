"""Viewer tab lifecycle manager."""

from __future__ import annotations

from typing import Dict, Optional, Type

from PySide6.QtWidgets import QTabWidget, QWidget

from core.session import Session

from .base_viewer import BaseViewer
from .surface_viewer import SurfaceViewer
from .volume_viewer import VolumeViewer


class ViewerManager:
    """Manage tabbed viewers and simple viewer type mapping."""

    def __init__(self, tabs: QTabWidget, session: Session) -> None:
        self._tabs = tabs
        self._session = session
        self._viewer_factories: Dict[str, Type[BaseViewer]] = {
            "base": BaseViewer,
            "surface": SurfaceViewer,
            "volume": VolumeViewer,
        }

    def create_viewer_tab(self, viewer_type: str, title: str | None = None) -> QWidget:
        key = viewer_type.lower()
        factory = self._viewer_factories.get(key)
        if factory is None:
            raise ValueError(f"Unknown viewer type '{viewer_type}'")

        viewer = factory() if factory is not BaseViewer else BaseViewer(title=title or "Viewer")
        tab_title = title or viewer_type.title()
        self._tabs.addTab(viewer, tab_title)
        self._tabs.setCurrentWidget(viewer)
        return viewer

    @property
    def active_viewer(self) -> Optional[QWidget]:
        return self._tabs.currentWidget()

    def open_surface(self, surface_name: str) -> QWidget:
        surface = self._session.get_surface(surface_name)
        viewer = self.create_viewer_tab("surface", f"Surface: {surface_name}")
        assert isinstance(viewer, SurfaceViewer)
        viewer.load_surface(surface_name, surface, session=self._session)
        return viewer

    def open_volume(self, image_name: str) -> QWidget:
        image = self._session.get_image(image_name)
        viewer = self.create_viewer_tab("volume", f"Volume: {image_name}")
        assert isinstance(viewer, VolumeViewer)
        viewer.load_image(image_name, image, session=self._session)
        return viewer

    def open_from_descriptor(self, descriptor: dict | None) -> Optional[QWidget]:
        if not descriptor:
            return None

        viewer_type = descriptor.get("type")
        object_name = descriptor.get("name")
        if viewer_type == "surface" and object_name:
            return self.open_surface(object_name)
        if viewer_type == "volume" and object_name:
            return self.open_volume(object_name)
        return None

    def close_tab(self, index: int) -> None:
        widget = self._tabs.widget(index)
        self._tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()
