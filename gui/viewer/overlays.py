"""Overlay helpers for viewer tabs."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pyvista as pv

from core.point import Point


def add_axes(plotter: pv.Plotter) -> None:
    """Add a standard orientation widget and scene axes."""

    plotter.show_axes()
    plotter.add_axes(line_width=2)


def add_points(plotter: pv.Plotter, named_points: Sequence[tuple[str, Point]], color: str = "#f97316") -> None:
    """Render named points as spheres with labels."""

    if not named_points:
        return

    coords = np.asarray([point.coords for _, point in named_points], dtype=float)
    cloud = pv.PolyData(coords)
    cloud["labels"] = [name for name, _ in named_points]
    plotter.add_points(
        cloud,
        color=color,
        point_size=12,
        render_points_as_spheres=True,
        name="session_points",
    )
    plotter.add_point_labels(
        cloud,
        "labels",
        point_size=0,
        font_size=10,
        fill_shape=False,
        margin=0,
        text_color=color,
        name="session_point_labels",
    )


def points_in_frame(point_items: Iterable[tuple[str, Point]], frame_name: str) -> list[tuple[str, Point]]:
    """Filter session points to those already expressed in the rendered frame."""

    return [(name, point) for name, point in point_items if point.frame.name == frame_name]