"""Viewer tab lifecycle manager."""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from core.session import Session


class _UnavailableViewer(QWidget):
    def __init__(self, title: str, message: str) -> None:
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        heading = QLabel(title)
        heading.setStyleSheet("font-size: 16px; font-weight: 600;")
        body = QLabel(message)
        body.setWordWrap(True)

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addStretch(1)

    def reset_camera(self) -> None:
        return


class ViewerManager:
    """Manage tabbed viewers and simple viewer type mapping."""

    def __init__(
        self,
        tabs: QTabWidget,
        session: Session,
        registration_callback: Callable[[str | None], None] | None = None,
    ) -> None:
        self._tabs = tabs
        self._session = session
        self._registration_callback = registration_callback
        self._viewer_backend_error: str | None = None

    def set_session(self, session: Session) -> None:
        """Replace the active session used by viewers."""
        self._session = session

    def create_viewer_tab(self, viewer_type: str, title: str | None = None) -> QWidget:
        key = viewer_type.lower()
        if key not in {"base", "surface", "volume"}:
            raise ValueError(f"Unknown viewer type '{viewer_type}'")

        viewer = self._build_viewer(key, title or viewer_type.title())
        tab_title = title or viewer_type.title()
        self._tabs.addTab(viewer, tab_title)
        self._tabs.setCurrentWidget(viewer)
        return viewer

    def _build_viewer(self, viewer_type: str, title: str) -> QWidget:
        try:
            if viewer_type == "base":
                from .base_viewer import BaseViewer

                return BaseViewer(title=title)
            if viewer_type == "surface":
                from .surface_viewer import SurfaceViewer

                return SurfaceViewer()
            if viewer_type == "volume":
                from .volume_viewer import VolumeViewer

                return VolumeViewer()
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            self._viewer_backend_error = str(exc)
            return _UnavailableViewer(
                title,
                "The interactive 3D viewer backend could not be loaded on this Windows installation. "
                f"Details: {exc}",
            )

        raise ValueError(f"Unknown viewer type '{viewer_type}'")

    @property
    def active_viewer(self) -> Optional[QWidget]:
        return self._tabs.currentWidget()

    def open_surface(self, surface_name: str) -> QWidget:
        surface = self._session.get_surface(surface_name)
        viewer = self.create_viewer_tab("surface", f"Surface: {surface_name}")
        viewer_any = cast(Any, viewer)
        if hasattr(viewer_any, "load_surface"):
            viewer_any.load_surface(surface_name, surface, session=self._session)
        return viewer
    
    def open_volume(self, image_name: str) -> QWidget:
        image = self._session.get_image(image_name)
        viewer = self.create_viewer_tab("volume", f"Volume: {image_name}")
        viewer_any = cast(Any, viewer)
        if hasattr(viewer_any, "load_image"):
            viewer_any.load_image(image_name, image, session=self._session)
        if hasattr(viewer_any, "set_registration_request_handler"):
            viewer_any.set_registration_request_handler(self._registration_callback)
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

    def refresh_viewers(self) -> None:
        for index in range(self._tabs.count()):
            viewer = self._tabs.widget(index)
            if viewer is None:
                continue
            viewer_any = cast(Any, viewer)
            if hasattr(viewer_any, "refresh_from_session"):
                viewer_any.refresh_from_session(self._session)
            elif hasattr(viewer_any, "update_scene"):
                viewer_any.update_scene()

    def close_tab(self, index: int) -> None:
        widget = self._tabs.widget(index)

        if widget is None:
            return

        self._tabs.removeTab(index)

        viewer_any = cast(Any, widget)
        if hasattr(viewer_any, "cleanup"):
            viewer_any.cleanup()

        widget.deleteLater()

    def close_all_tabs(self) -> None:
        for index in reversed(range(self._tabs.count())):
            self.close_tab(index)

    def focus_point(self, point_name: str) -> bool:
        try:
            point = self._session.get_point(point_name)
        except KeyError:
            return False

        compatible_viewers: list[QWidget] = []

        for index in range(self._tabs.count()):
            viewer = self._tabs.widget(index)

            if viewer is None:
                continue

            viewer_any = cast(Any, viewer)

            image = getattr(viewer_any, "image", None)

            if image is None:
                continue

            if image.voxel_frame == point.frame:
                compatible_viewers.append(viewer)

        if not compatible_viewers:
            return False

        active_viewer = self.active_viewer

        # Prefer the currently active compatible viewer.
        if active_viewer in compatible_viewers:
            viewer = active_viewer
        else:
            # Otherwise use the first compatible open viewer.
            viewer = compatible_viewers[0]

        viewer_any = cast(Any, viewer)

        if not viewer_any.focus_point(point_name):
            return False

        self._tabs.setCurrentWidget(viewer)
        return True

    def _volume_viewers(self) -> list[QWidget]:
        viewers: list[QWidget] = []

        for index in range(self._tabs.count()):
            viewer = self._tabs.widget(index)

            if viewer is None:
                continue

            viewer_any = cast(Any, viewer)

            if getattr(viewer_any, "image", None) is not None:
                viewers.append(viewer)

        return viewers

    def _viewer_can_focus_point(
        self,
        viewer: QWidget,
        point_name: str,
    ) -> bool:
        viewer_any = cast(Any, viewer)

        image = getattr(viewer_any, "image", None)

        if image is None:
            return False

        try:
            point = self._session.get_point(point_name)
        except KeyError:
            return False

        return image.voxel_frame == point.frame
