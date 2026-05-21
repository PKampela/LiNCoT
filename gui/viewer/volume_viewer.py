"""Interactive MRI viewer with axial, coronal, and sagittal slice panes."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QSlider, QVBoxLayout, QWidget

from core.image import Image
from core.session import Session

from .render_utils import image_orientation, orthogonal_slice_data, voxel_size, voxel_to_world, world_to_voxel
from .viewer_tab import ViewerTab


class SliceImageLabel(QLabel):
    """Label that scales a stored pixmap to fit the available pane."""

    def __init__(self) -> None:
        super().__init__()
        self._source_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(180, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setLineWidth(1)

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_pixmap()

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


class VolumeViewer(ViewerTab):
    def __init__(self) -> None:
        super().__init__(title="MRI Viewer")
        self._reset_button.setToolTip("Center slices")
        self._image: Optional[Image] = None
        self._image_name: Optional[str] = None
        self._session: Session | None = None
        self._slice_sliders: list[QSlider] = []
        self._slice_labels: list[QLabel] = []
        self._slice_titles: list[QLabel] = []
        self._slice_views: list[SliceImageLabel] = []
        self._slice_names = ("Axial", "Coronal", "Sagittal")
        self._updating_controls = False

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

        for title in self._slice_names:
            pane = QWidget(self._slice_container)
            pane_layout = QVBoxLayout(pane)
            pane_layout.setContentsMargins(8, 8, 8, 8)
            pane_layout.setSpacing(6)
            pane.setStyleSheet("QWidget { background: #111827; border: 1px solid #374151; border-radius: 8px; }")

            pane_title = QLabel(title)
            pane_title.setObjectName("viewerSidebarTitle")
            pane_layout.addWidget(pane_title)

            slice_view = SliceImageLabel()
            pane_layout.addWidget(slice_view, 1)
            self._slice_container_layout.addWidget(pane, 1)

            self._slice_titles.append(pane_title)
            self._slice_views.append(slice_view)

    def load_image(self, image_name: str, image: Image, session: Session | None = None) -> None:
        self.clear_scene()
        self._image = image
        self._image_name = image_name
        self._session = session

        center = tuple((float(size) - 1.0) / 2.0 for size in image.shape)
        for axis, slider in enumerate(self._slice_sliders):
            self._set_slider_range(slider, 0, image.shape[axis] - 1)

        self.set_title(f"MRI: {image_name}")

        voxel_spacing = voxel_size(image)
        intensity_min = float(np.min(image.data))
        intensity_max = float(np.max(image.data))
        self.set_metadata([
            ("Image", image_name),
            ("Type", "MRI Slice Viewer"),
            ("Dimensions", f"{image.shape[0]} × {image.shape[1]} × {image.shape[2]}"),
            ("Voxel size", f"{voxel_spacing[0]:0.3f} × {voxel_spacing[1]:0.3f} × {voxel_spacing[2]:0.3f} mm"),
            ("Orientation", image_orientation(image)),
            ("Data type", str(image.data.dtype)),
            ("Intensity range", f"{intensity_min:0.3f} → {intensity_max:0.3f}"),
            ("Frame", image.frame.name),
            ("Display", "Axial / Coronal / Sagittal slices"),
        ])

        self._set_cursor_from_voxel(center)
        self._focus_point_in_image_frame()
        self.set_status(f"Loaded MRI '{image_name}'")

    def refresh_from_session(self, session: Session | None = None) -> None:
        active_session = session or self._session
        if active_session is None or self._image_name is None:
            return

        try:
            image = active_session.get_image(self._image_name)
        except KeyError:
            self.clear_scene()
            return

        self.load_image(self._image_name, image, session=active_session)

    def _update_slices(self) -> None:
        if self._image is None or self._updating_controls:
            return

        x_index, y_index, z_index = [slider.value() for slider in self._slice_sliders]
        self._apply_cursor((x_index, y_index, z_index))

    def _set_slider_range(self, slider: QSlider, minimum: int, maximum: int) -> None:
        slider.blockSignals(True)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.blockSignals(False)

    def _set_cursor_from_voxel(self, voxel_coords: tuple[float, float, float]) -> None:
        if self._image is None:
            return

        indices = tuple(int(round(value)) for value in voxel_coords)
        indices = self._clamp_indices(indices)
        self._apply_cursor(indices)

    def _set_cursor_from_world(self, world_coords: tuple[float, float, float]) -> None:
        if self._image is None:
            return

        voxel_coords = world_to_voxel(self._image, world_coords)
        indices = tuple(int(round(value)) for value in voxel_coords)
        indices = self._clamp_indices(indices)
        self._apply_cursor(indices)

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
        for axis, slider in enumerate(self._slice_sliders):
            slider.setValue(indices[axis])
            self._slice_labels[axis].setText(f"{('XYZ'[axis])} slice: {indices[axis]}")
        self._updating_controls = False

        world = voxel_to_world(self._image, tuple(float(value) for value in indices))
        self._update_slice_panels(indices)
        self._update_cursor_details(indices, world)

    def _focus_point_in_image_frame(self) -> None:
        if self._image is None or self._session is None:
            return

        matching_points = [
            (name, self._session.get_point(name))
            for name in self._session.list_points()
            if self._session.get_point(name).frame.name == self._image.frame.name
        ]
        if not matching_points:
            return

        _, point = matching_points[0]
        self._set_cursor_from_world(tuple(float(value) for value in point.coords.tolist()))

    def _update_slice_panels(self, indices: tuple[int, int, int]) -> None:
        if self._image is None:
            return

        slice_specs = (
            (2, indices[2], (indices[0], indices[1]), f"Axial (z = {indices[2]})"),
            (1, indices[1], (indices[0], indices[2]), f"Coronal (y = {indices[1]})"),
            (0, indices[0], (indices[1], indices[2]), f"Sagittal (x = {indices[0]})"),
        )
        point_items = self._matching_points_in_image_frame()
        for pane_index, (axis, slice_index, crosshair, title) in enumerate(slice_specs):
            self._slice_titles[pane_index].setText(title)
            slice_data = orthogonal_slice_data(self._image, axis, slice_index)
            pixmap = self._slice_to_pixmap(slice_data, crosshair)
            if point_items:
                self._draw_points_on_slice(pixmap, axis, slice_index, point_items)
            self._slice_views[pane_index].set_source_pixmap(pixmap)

    def _matching_points_in_image_frame(self) -> list[tuple[str, object]]:
        if self._image is None or self._session is None:
            return []

        points = []
        for name in self._session.list_points():
            point = self._session.get_point(name)
            if point.frame.name == self._image.frame.name:
                points.append((name, point))
        return points

    def _draw_points_on_slice(
        self,
        pixmap: QPixmap,
        axis: int,
        slice_index: int,
        point_items: list[tuple[str, object]],
    ) -> None:
        if self._image is None:
            return

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        for name, point in point_items:
            try:
                voxel_coords = world_to_voxel(self._image, tuple(float(value) for value in point.coords.tolist()))
            except Exception:
                continue

            if abs(float(voxel_coords[axis]) - float(slice_index)) > 0.5:
                continue

            if axis == 2:
                x_pos, y_pos = float(voxel_coords[0]), float(voxel_coords[1])
            elif axis == 1:
                x_pos, y_pos = float(voxel_coords[0]), float(voxel_coords[2])
            else:
                x_pos, y_pos = float(voxel_coords[1]), float(voxel_coords[2])

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
        data = np.asarray(slice_data, dtype=float).T
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
        for axis, label in enumerate(self._slice_labels):
            label.setText(f"{('XYZ'[axis])}: -")
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