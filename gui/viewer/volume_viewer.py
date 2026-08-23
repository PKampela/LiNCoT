"""Interactive MRI viewer with axial, coronal, and sagittal slice panes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from nibabel import aff2axcodes
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QSizePolicy, QSlider, QVBoxLayout, QWidget

from core.image import Image
from core.point import Point
from core.session import Session

from .render_utils import anatomical_axis_info, image_orientation, voxel_size, voxel_to_world, world_to_voxel, point_to_voxel, voxel_to_point
from .viewer_tab import ViewerTab

class SliceImageLabel(QLabel):
    """Label that scales a stored pixmap to fit the available pane."""

    clicked = Signal(float, float)

    def __init__(self) -> None:
        super().__init__()
        self._source_pixmap: Optional[QPixmap] = None

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(180, 180)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setFrameShape(QFrame.Shape.NoFrame)

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_pixmap()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self._source_pixmap is None
            or self._source_pixmap.isNull()
        ):
            super().mousePressEvent(event)
            return

        x, y = event.position().x(), event.position().y()

        source_x, source_y = self._widget_to_source_position(x, y)

        if source_x is None or source_y is None:
            return

        self.clicked.emit(source_x, source_y)

    def _widget_to_source_position(
        self,
        widget_x: float,
        widget_y: float,
    ) -> tuple[float | None, float | None]:

        if self._source_pixmap is None or self._source_pixmap.isNull():
            return None, None

        source_width = self._source_pixmap.width()
        source_height = self._source_pixmap.height()

        scale = min(
            self.width() / source_width,
            self.height() / source_height,
        )

        displayed_width = source_width * scale
        displayed_height = source_height * scale

        offset_x = (self.width() - displayed_width) / 2.0
        offset_y = (self.height() - displayed_height) / 2.0

        source_x = (widget_x - offset_x) / scale
        source_y = (widget_y - offset_y) / scale

        if not (
            0 <= source_x < source_width
            and 0 <= source_y < source_height
        ):
            return None, None

        return source_x, source_y

    def _refresh_pixmap(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            self.setPixmap(QPixmap())
            return

        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

@dataclass(frozen=True)
class _PlaneSpec:
    title: str
    fixed_group: str
    row_group: str
    col_group: str



PLANE_LAYOUT: tuple[_PlaneSpec, ...] = (
    _PlaneSpec(title="Axial", fixed_group="si", row_group="ap", col_group="lr"),
    _PlaneSpec(title="Coronal", fixed_group="ap", row_group="si", col_group="lr"),
    _PlaneSpec(title="Sagittal", fixed_group="lr", row_group="si", col_group="ap"),
)


class _AxisInfo(Protocol):
    voxel_axis: int
    code: str
    sign: int


class VolumeViewer(ViewerTab):
    def __init__(self) -> None:
        super().__init__(title="MRI Viewer")
        self.hide_header()
        self._reset_button.setToolTip("Center slices")
        self._image: Optional[Image] = None
        self._image_name: Optional[str] = None
        self._session: Session | None = None
        self._slice_sliders: list[QSlider] = []
        self._slice_labels: list[QLabel] = []
        self._slice_titles: list[QLabel] = []
        self._slice_views: list[SliceImageLabel] = []
        self._slice_titles_by_plane: dict[str, QLabel] = {}
        self._slice_views_by_plane: dict[str, SliceImageLabel] = {}
        self._slice_names = tuple(spec.title for spec in PLANE_LAYOUT)
        self._updating_controls = False
        self._registration_request_handler: Callable[[str | None], None] | None = None
        self._focused_point_name: str | None = None
        self._axis_mapping: dict[str, _AxisInfo] = {}
        self._slider_group_order = ("lr", "ap", "si")
        self._slider_prefix = {
            "lr": "L-R",
            "ap": "P-A",
            "si": "I-S",
        }
        # Keep pane order fixed for all images; image orientation only maps voxel axes.
        self._plane_specs = PLANE_LAYOUT

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda position: self._show_context_menu(self, position))

        plotter_item = self._content_layout.takeAt(0)
        if plotter_item is not None and plotter_item.widget() is not None:
            plotter_item.widget().setParent(None)

        self._slice_container = QWidget(self)
        self._slice_container_layout = QHBoxLayout(self._slice_container)
        self._slice_container_layout.setContentsMargins(0, 0, 0, 0)
        self._slice_container_layout.setSpacing(8)
        self._content_layout.insertWidget(0, self._slice_container, 1)

        for axis_name in ("X", "Y", "Z"):
            label = QLabel(f"{axis_name}: 0")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.valueChanged.connect(self._update_slices)
            self._sidebar_layout.insertWidget(self._sidebar_layout.count() - 1, label)
            self._sidebar_layout.insertWidget(self._sidebar_layout.count() - 1, slider)
            self._slice_labels.append(label)
            self._slice_sliders.append(slider)

        self._cursor_title = QLabel("Current Cursor Position")
        self._cursor_title.setObjectName("viewerSidebarTitle")
        self._sidebar_layout.insertWidget(self._sidebar_layout.count() - 1, self._cursor_title)

        self._cursor_voxel_label = QLabel("Voxel: -")
        self._cursor_voxel_label.setWordWrap(True)
        self._sidebar_layout.insertWidget(self._sidebar_layout.count() - 1, self._cursor_voxel_label)

        self._cursor_mri_label = QLabel("MRI: -")
        self._cursor_mri_label.setWordWrap(True)
        self._sidebar_layout.insertWidget(self._sidebar_layout.count() - 1, self._cursor_mri_label)

        self._cursor_intensity_label = QLabel("Intensity: -")
        self._cursor_intensity_label.setWordWrap(True)
        self._sidebar_layout.insertWidget(self._sidebar_layout.count() - 1, self._cursor_intensity_label)

        for spec in PLANE_LAYOUT:
            title = spec.title
            pane = QWidget(self._slice_container)
            pane.setObjectName("slicePane")
            pane_layout = QVBoxLayout(pane)
            pane_layout.setContentsMargins(8, 8, 8, 8)
            pane_layout.setSpacing(6)
            pane.setStyleSheet("QWidget#slicePane { background: #111827; border: 1px solid #374151; border-radius: 8px; }")

            pane_title = QLabel(title)
            pane_title.setObjectName("viewerSidebarTitle")
            pane_layout.addWidget(pane_title)

            slice_view = SliceImageLabel()
            pane_layout.addWidget(slice_view, 1)
            self._slice_container_layout.addWidget(pane, 1)

            slice_view.clicked.connect(
                lambda x, y, plane=title: self._handle_slice_click(
                    plane,
                    x,
                    y,
                )
            )

            slice_view.setContextMenuPolicy(
                Qt.ContextMenuPolicy.CustomContextMenu
            )
            slice_view.customContextMenuRequested.connect(
                lambda position, widget=slice_view: self._show_context_menu(widget, position)
            )

            self._slice_titles.append(pane_title)
            self._slice_views.append(slice_view)
            self._slice_titles_by_plane[title] = pane_title
            self._slice_views_by_plane[title] = slice_view

    @property
    def image(self) -> Image | None:
        return self._image

    def _handle_slice_click(
        self,
        plane_name: str,
        image_x: float,
        image_y: float,
    ) -> None:
        """Handle a click inside one of the MRI slice views."""

        point_name = self._find_point_at_position(
            plane_name,
            image_x,
            image_y,
        )

        if point_name is not None:
            self.focus_point(point_name)
            return

        voxel = self._plane_pixel_to_voxel(
            plane_name,
            image_x,
            image_y,
        )

        if voxel is None:
            return

        self._focused_point_name = None
        self._set_cursor_from_voxel(voxel)

    def _find_point_at_position(
        self,
        plane_name: str,
        image_x: float,
        image_y: float,
    ) -> str | None:
        if self._image is None or self._session is None:
            return None

        plane_spec = self._get_plane_spec(plane_name)
        if plane_spec is None:
            return None

        fixed_info = self._axis_mapping.get(
            plane_spec.fixed_group
        )

        if fixed_info is None:
            return None

        fixed_slider = self._slider_for_group(
            plane_spec.fixed_group
        )

        if fixed_slider is None:
            return None

        current_fixed_index = fixed_slider.value()

        row_info = self._axis_mapping.get(
            plane_spec.row_group
        )
        col_info = self._axis_mapping.get(
            plane_spec.col_group
        )

        if row_info is None or col_info is None:
            return None

        best_name: str | None = None
        best_distance_squared = float("inf")

        hit_radius = 8.0
        hit_radius_squared = hit_radius * hit_radius

        for name, point in self._session.points.items():
            try:
                voxel_coords = point_to_voxel(
                    self._image,
                    point,
                )
            except Exception:
                continue

            indices = tuple(
                int(round(value))
                for value in voxel_coords
            )

            fixed_index = indices[
                fixed_info.voxel_axis
            ]

            if abs(
                fixed_index - current_fixed_index
            ) > 1:
                continue

            point_x, point_y = self._voxel_to_plane_pixel(
                indices,
                row_axis=row_info.voxel_axis,
                col_axis=col_info.voxel_axis,
                row_sign=row_info.sign,
                col_sign=col_info.sign,
            )

            dx = float(point_x) - image_x
            dy = float(point_y) - image_y

            distance_squared = (
                dx * dx + dy * dy
            )

            if distance_squared <= hit_radius_squared:
                if distance_squared < best_distance_squared:
                    best_distance_squared = distance_squared
                    best_name = name

        return best_name

    def set_registration_request_handler(self, handler: Callable[[str | None], None] | None) -> None:
        self._registration_request_handler = handler

    def _show_context_menu(self, widget: QWidget, position) -> None:
        if self._image_name is None or self._registration_request_handler is None:
            return

        handler = self._registration_request_handler
        image_name = self._image_name

        menu = QMenu(self)
        menu.addAction(
            "Register to...",
            lambda: handler(image_name),
        )
        global_position = widget.mapToGlobal(position)
        menu.exec(global_position)

    def load_image(self, image_name: str, image: Image, session: Session | None = None, focused_point_name: str | None = None) -> None:
        self.clear_scene()
        self._image = image
        self._image_name = image_name
        self._session = session
        self._axis_mapping = anatomical_axis_info(image)

        self._focused_point_name = focused_point_name

        center = tuple((float(size) - 1.0) / 2.0 for size in image.shape)
        for slider_index, slider in enumerate(self._slice_sliders):
            group = self._slider_group_order[slider_index]
            voxel_axis = self._axis_mapping[group].voxel_axis
            self._set_slider_range(slider, 0, image.shape[voxel_axis] - 1)

        self.set_title("")

        voxel_spacing = voxel_size(image)
        self.set_metadata([
            ("Image", image_name),
            ("Type", "MRI Slice Viewer"),
            ("Dimensions", f"{image.shape[0]} × {image.shape[1]} × {image.shape[2]}"),
            ("Voxel size", f"{voxel_spacing[0]:0.3f} × {voxel_spacing[1]:0.3f} × {voxel_spacing[2]:0.3f} mm"),
            ("Orientation", image_orientation(image)),
            ("Voxel Frame", image.voxel_frame.name),
            ("World Frame", image.world_frame.name),
            ("Display", "Axial / Coronal / Sagittal slices"),
        ])

        if self._focused_point_name is not None:
            if not self.focus_point(self._focused_point_name):
                self._focused_point_name = None
                self._set_cursor_from_voxel(center)
        else:
            self._set_cursor_from_voxel(center)

        self.set_status(f"Loaded MRI '{image_name}'")

    def refresh_from_session(
        self,
        session: Session | None = None,
    ) -> None:
        active_session = session or self._session

        if active_session is None or self._image_name is None:
            return

        try:
            image = active_session.get_image(self._image_name)
        except KeyError:
            self.clear_scene()
            return

        focused_point_name = self._focused_point_name

        self.load_image(
            self._image_name,
            image,
            session=active_session,
            focused_point_name=focused_point_name,
        )

    def _current_cursor_indices(self) -> tuple[int, int, int]:
        return tuple(
            slider.value()
            for slider in self._slice_sliders
        )  # type: ignore[return-value]

    def _update_slices(self) -> None:
        if self._image is None or self._updating_controls:
            return

        indices = [0, 0, 0]
        for slider_index, slider in enumerate(self._slice_sliders):
            group = self._slider_group_order[slider_index]
            voxel_axis = self._axis_mapping[group].voxel_axis
            indices[voxel_axis] = slider.value()

        self._apply_cursor((indices[0], indices[1], indices[2]))

    def _set_slider_range(self, slider: QSlider, minimum: int, maximum: int) -> None:
        slider.blockSignals(True)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.blockSignals(False)

    def _slider_for_group(self, group: str) -> QSlider | None:
        try:
            slider_index = self._slider_group_order.index(group)
        except ValueError:
            return None

        return self._slice_sliders[slider_index]

    def _set_cursor_from_voxel(
        self,
        voxel_coords: tuple[float, float, float],
    ) -> None:
        if self._image is None:
            return

        indices = tuple(
            int(round(value))
            for value in voxel_coords
        )

        indices = self._clamp_indices(indices)
        self._apply_cursor(indices)


    def _set_cursor_from_world(
        self,
        world_coords: tuple[float, float, float],
    ) -> None:
        if self._image is None:
            return

        voxel_coords = world_to_voxel(
            self._image,
            world_coords,
        )

        self._set_cursor_from_voxel(
            (
                float(voxel_coords[0]),
                float(voxel_coords[1]),
                float(voxel_coords[2]),
            )
        )

    def _clamp_indices(self, indices: tuple[int, int, int]) -> tuple[int, int, int]:
        assert self._image is not None
        return tuple(
            int(np.clip(index, 0, self._image.shape[axis] - 1))
            for axis, index in enumerate(indices)
        )

    def _apply_cursor(self, indices: tuple[int, int, int]) -> None:
        if self._image is None:
            return

        self._updating_controls = True
        for slider_index, slider in enumerate(self._slice_sliders):
            group = self._slider_group_order[slider_index]
            voxel_axis = self._axis_mapping[group].voxel_axis
            slider_value = indices[voxel_axis]
            slider.setValue(slider_value)
            self._slice_labels[slider_index].setText(
                f"{self._slider_prefix[group]} slice: {slider_value}"
            )
        self._updating_controls = False

        world = voxel_to_world(self._image, tuple(float(value) for value in indices))
        self._update_slice_panels(indices)
        self._update_cursor_details(indices, world)

    def focus_point(self, point_name: str) -> bool:
        if self._image is None or self._session is None:
            return False

        try:
            point = self._session.get_point(point_name)
        except KeyError:
            return False

        try:
            voxel_coords = point_to_voxel(self._image, point)
        except Exception:
            return False

        self._focused_point_name = point_name

        self._set_cursor_from_voxel(
            (
                float(voxel_coords[0]),
                float(voxel_coords[1]),
                float(voxel_coords[2]),
            )
        )

        return True

    def _update_slice_panels(self, indices: tuple[int, int, int]) -> None:
        if self._image is None:
            return

        point_items = self._matching_points_in_image_frame()
        for spec in PLANE_LAYOUT:
            fixed = self._axis_mapping[spec.fixed_group]
            row = self._axis_mapping[spec.row_group]
            col = self._axis_mapping[spec.col_group]

            slice_index = indices[fixed.voxel_axis]
            title = f"{spec.title} ({self._slider_prefix[spec.fixed_group]} = {slice_index})"
            pane_title = self._slice_titles_by_plane.get(spec.title)
            if pane_title is None:
                continue
            pane_title.setText(title)

            slice_data = self._extract_oriented_slice(
                fixed_axis=fixed.voxel_axis,
                slice_index=slice_index,
                row_axis=row.voxel_axis,
                col_axis=col.voxel_axis,
                row_sign=row.sign,
                col_sign=col.sign,
            )
            crosshair = self._voxel_to_plane_pixel(
                indices,
                row_axis=row.voxel_axis,
                col_axis=col.voxel_axis,
                row_sign=row.sign,
                col_sign=col.sign,
            )

            pixmap = self._slice_to_pixmap(slice_data, crosshair)
            if point_items:
                self._draw_points_on_slice(
                    pixmap,
                    fixed_axis=fixed.voxel_axis,
                    slice_index=slice_index,
                    row_axis=row.voxel_axis,
                    col_axis=col.voxel_axis,
                    row_sign=row.sign,
                    col_sign=col.sign,
                    point_items=point_items,
                )
            pane_view = self._slice_views_by_plane.get(spec.title)
            if pane_view is not None:
                pane_view.set_source_pixmap(pixmap)

    def _extract_oriented_slice(
        self,
        fixed_axis: int,
        slice_index: int,
        row_axis: int,
        col_axis: int,
        row_sign: int,
        col_sign: int,
    ) -> np.ndarray:
        """Extract a slice in a deterministic anatomical row/column order."""

        assert self._image is not None

        data = np.asarray(self._image.data)

        # The slice is indexed by the fixed voxel axis.
        slice_data = np.take(
            data,
            slice_index,
            axis=fixed_axis,
        )

        remaining_axes = [
            axis for axis in range(3)
            if axis != fixed_axis
        ]

        # Determine where the requested row/column axes ended up
        # after removing fixed_axis.
        row_position = remaining_axes.index(row_axis)
        col_position = remaining_axes.index(col_axis)

        # Explicitly put:
        #
        #   output axis 0 = anatomical row
        #   output axis 1 = anatomical column
        #
        oriented = np.transpose(
            slice_data,
            axes=(row_position, col_position),
        )

        # Convert voxel-axis direction into the desired anatomical
        # display direction.
        if row_sign > 0:
            oriented = np.flip(oriented, axis=0)

        if col_sign > 0:
            oriented = np.flip(oriented, axis=1)


        return np.asarray(oriented)

    def _voxel_to_plane_pixel(
        self,
        indices: tuple[int, int, int],
        row_axis: int,
        col_axis: int,
        row_sign: int,
        col_sign: int,
    ) -> tuple[int, int]:
        """
        Convert voxel coordinates to display pixel coordinates.

        The display convention is:

            horizontal: negative anatomical direction -> positive
            vertical:   negative anatomical direction -> positive

        This must mirror _extract_oriented_slice().
        """

        assert self._image is not None

        row_index = int(indices[row_axis])
        col_index = int(indices[col_axis])

        if row_sign > 0:
            row_index = (
                self._image.shape[row_axis] - 1
            ) - row_index

        if col_sign > 0:
            col_index = (
                self._image.shape[col_axis] - 1
            ) - col_index

        # QImage/QPixmap coordinates are (x, y), i.e. (column, row).
        return col_index, row_index

    def _plane_pixel_to_voxel(
        self,
        plane_name: str,
        image_x: float,
        image_y: float,
    ) -> tuple[float, float, float] | None:
        """
        Convert a displayed slice pixel back to voxel coordinates.

        This is the exact inverse of _voxel_to_plane_pixel().
        """

        if self._image is None:
            return None

        plane_spec = self._get_plane_spec(plane_name)
        if plane_spec is None:
            return None

        row_info = self._axis_mapping.get(
            plane_spec.row_group
        )
        col_info = self._axis_mapping.get(
            plane_spec.col_group
        )
        fixed_info = self._axis_mapping.get(
            plane_spec.fixed_group
        )

        if (
            row_info is None
            or col_info is None
            or fixed_info is None
        ):
            return None

        fixed_slider = self._slider_for_group(
            plane_spec.fixed_group
        )

        if fixed_slider is None:
            return None

        row_index = int(round(image_y))
        col_index = int(round(image_x))

        # Undo the display flip performed by
        # _extract_oriented_slice().
        if row_info.sign > 0:
            row_index = (
                self._image.shape[row_info.voxel_axis] - 1
            ) - row_index

        if col_info.sign > 0:
            col_index = (
                self._image.shape[col_info.voxel_axis] - 1
            ) - col_index

        voxel = [0.0, 0.0, 0.0]

        voxel[row_info.voxel_axis] = float(row_index)
        voxel[col_info.voxel_axis] = float(col_index)
        voxel[fixed_info.voxel_axis] = float(
            fixed_slider.value()
        )

        return (
            voxel[0],
            voxel[1],
            voxel[2],
        )
    
    def _get_plane_spec(self, plane_name: str) -> _PlaneSpec | None:
        for spec in self._plane_specs:
            if spec.title == plane_name:
                return spec

        return None

    def _matching_points_in_image_frame(self) -> list[tuple[str, Point]]:
        if self._image is None or self._session is None:
            return []

        points = []
        for name in self._session.list_points():
            point = self._session.get_point(name)
            if point.frame.name == self._image.voxel_frame.name:
                points.append((name, point))
        return points

    def _draw_points_on_slice(
        self,
        pixmap: QPixmap,
        fixed_axis: int,
        slice_index: int,
        row_axis: int,
        col_axis: int,
        row_sign: int,
        col_sign: int,
        point_items: list[tuple[str, Point]],
    ) -> None:
        if self._image is None:
            return

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        for name, point in point_items:
            try:
                voxel_coords = point_to_voxel(self._image, point)
            except Exception:
                continue

            if abs(float(voxel_coords[fixed_axis]) - float(slice_index)) > 0.5:
                continue

            voxel_index = (
                int(round(float(voxel_coords[0]))),
                int(round(float(voxel_coords[1]))),
                int(round(float(voxel_coords[2]))),
            )
            x_pos, y_pos = self._voxel_to_plane_pixel(
                voxel_index,
                row_axis=row_axis,
                col_axis=col_axis,
                row_sign=row_sign,
                col_sign=col_sign,
            )

            x_int = int(round(x_pos))
            y_int = int(round(y_pos))
            if x_int < 0 or y_int < 0:
                continue

            pen = QPen(QColor("#f97316"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QColor("#f97316"))
            painter.drawEllipse(x_int - 4, y_int - 4, 8, 8)
            painter.drawText(x_int + 6, y_int - 6, name)

        painter.end()

    def _slice_to_pixmap(self, slice_data: np.ndarray, crosshair: tuple[int, int]) -> QPixmap:
        data = np.asarray(slice_data, dtype=float)
        finite = np.isfinite(data)
        if not finite.any():
            scaled = np.zeros_like(data, dtype=np.uint8)
        else:
            valid = data[finite]
            minimum = float(valid.min())
            maximum = float(valid.max())
            if maximum <= minimum:
                scaled = np.zeros_like(data, dtype=np.uint8)
            else:
                normalized = (np.clip(data, minimum, maximum) - minimum) / (maximum - minimum)
                scaled = np.asarray(np.round(normalized * 255.0), dtype=np.uint8)

        scaled = np.ascontiguousarray(scaled)
        height, width = scaled.shape
        qimage = QImage(scaled.data, width, height, scaled.strides[0], QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage.copy())

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor("#f97316"))
        pen.setWidth(1)
        painter.setPen(pen)
        cross_x, cross_y = crosshair
        painter.drawLine(cross_x, 0, cross_x, height - 1)
        painter.drawLine(0, cross_y, width - 1, cross_y)
        painter.end()
        return pixmap

    def _update_cursor_details(self, indices: tuple[int, int, int], world: tuple[float, float, float]) -> None:
        if self._image is None:
            return

        voxel_text = f"Voxel: ({indices[0]}, {indices[1]}, {indices[2]})"
        mri_text = f"MRI: ({world[0]:0.1f}, {world[1]:0.1f}, {world[2]:0.1f}) mm"

        sample = self._sample_intensity(indices)
        self._cursor_voxel_label.setText(voxel_text)
        self._cursor_mri_label.setText(mri_text)
        self._cursor_intensity_label.setText(f"Intensity: {sample:0.3f}" if sample is not None else "Intensity: n/a")
        self.set_status(f"{voxel_text} | {mri_text}")

    def _sample_intensity(self, indices: tuple[int, int, int]) -> float | None:
        if self._image is None:
            return None

        try:
            return float(self._image.data[indices[0], indices[1], indices[2]])
        except Exception:
            return None

    def clear_scene(self) -> None:
        self._image = None
        self._image_name = None
        self._axis_mapping = {}
        for axis, label in enumerate(self._slice_labels):
            label.setText(f"{self._slider_prefix[self._slider_group_order[axis]]}: -")
        for axis, title in enumerate(self._slice_titles):
            title.setText(self._slice_names[axis])
        for view in self._slice_views:
            view.set_source_pixmap(QPixmap())
        self._cursor_voxel_label.setText("Voxel: -")
        self._cursor_mri_label.setText("MRI: -")
        self._cursor_intensity_label.setText("Intensity: -")
        self.set_status("")

    def fit_camera(self) -> None:
        if self._image is None:
            return

        center = tuple((float(size) - 1.0) / 2.0 for size in self._image.shape)
        self._set_cursor_from_voxel(center)

    def reset_camera(self) -> None:
        self.fit_camera()

    def _update_hover_status(self, *_args) -> None:
        return