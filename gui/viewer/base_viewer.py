"""Compatibility wrapper for the shared viewer tab base class."""

from __future__ import annotations

import pyvista as pv

from .viewer_tab import ViewerTab


class BaseViewer(ViewerTab):
    """Generic viewer tab used for overview and placeholder views."""

    def __init__(self, title: str = "Viewer") -> None:
        super().__init__(title=title)
        self.hide_header()
        self._load_placeholder_scene()

    def load_object(self, _obj: object) -> None:
        return

    def _load_placeholder_scene(self) -> None:
        self.clear_scene()
        self.plotter.add_mesh(
            pv.Cube(center=(0.0, 0.0, 0.0), x_length=1.6, y_length=1.2, z_length=1.0),
            name="placeholder_cube",
            color="#60a5fa",
            opacity=0.32,
            smooth_shading=True,
            show_edges=True,
            edge_color="#2563eb",
        )
        self.plotter.add_mesh(
            pv.Sphere(radius=0.65, theta_resolution=32, phi_resolution=24),
            name="placeholder_sphere",
            color="#f59e0b",
            opacity=0.95,
            smooth_shading=True,
        )
        self.set_metadata(
            [
                ("Scene", "Placeholder geometry"),
                ("Workflow role", "Confirm the scene orientation before loading targets or anatomy"),
                ("What to inspect", "Loaded volumes, surfaces, points, and transforms"),
                ("Action", "Rotate, zoom, or open data from the File menu"),
                ("Status", "Viewer is active"),
            ]
        )
        self.fit_camera()

    def update_scene(self) -> None:
        self.plotter.render()

    def reset_camera(self) -> None:
        super().reset_camera()
