"""Interactive surface mesh viewer."""

from __future__ import annotations

from typing import Optional

from core.session import Session
from core.surface import Surface

from .overlays import add_points, points_in_frame
from .render_utils import surface_to_polydata
from .viewer_tab import ViewerTab


class SurfaceViewer(ViewerTab):
    def __init__(self) -> None:
        super().__init__(title="Surface Viewer")
        self._surface: Surface | None = None
        self._surface_name: Optional[str] = None
        self._session: Session | None = None

    def load_surface(self, surface_name: str, surface: Surface, session: Session | None = None) -> None:
        self.clear_scene()
        self._surface = surface
        self._surface_name = surface_name
        self._session = session

        mesh = surface_to_polydata(surface)
        self.plotter.add_mesh(
            mesh,
            name="surface_mesh",
            color="#7dd3fc",
            opacity=0.92,
            smooth_shading=True,
            show_edges=False,
        )
        self.plotter.add_text(surface_name, font_size=12, name="surface_name")

        if session is not None:
            point_items = [(name, session.get_point(name)) for name in session.list_points()]
            add_points(self.plotter, points_in_frame(point_items, surface.frame.name))

        self.set_title(f"Surface: {surface_name}")
        self.set_metadata([
            ("Object type", "Surface mesh"),
            ("Workflow role", "Scalp or cortical geometry used to judge target position and anatomy alignment"),
            ("Frame", surface.frame.name),
            ("Vertices", str(surface.vertices.shape[0])),
            ("Faces", str(surface.faces.shape[0])),
        ])
        self.fit_camera()
        self.set_status(f"Loaded surface '{surface_name}'")

    def refresh_from_session(self, session: Session | None = None) -> None:
        active_session = session or self._session
        if active_session is None or self._surface_name is None:
            return

        try:
            surface = active_session.get_surface(self._surface_name)
        except KeyError:
            self.clear_scene()
            return

        self.load_surface(self._surface_name, surface, session=active_session)
