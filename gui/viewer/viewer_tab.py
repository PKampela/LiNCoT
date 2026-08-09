"""Shared Qt/PyVista viewer tab infrastructure."""

from __future__ import annotations

import copy

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QPushButton, QStyle, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkWorldPointPicker

from gui.viewer.overlays import add_axes


class ViewerTab(QWidget):
    """Base widget for interactive viewer tabs."""

    def __init__(self, title: str = "Viewer") -> None:
        super().__init__()
        self._title = title
        self._picker = vtkWorldPointPicker()
        self._default_camera_position = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._header_widget = QWidget(self)
        header = QHBoxLayout(self._header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self._title_label = QLabel(title)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._reset_button = QPushButton("")
        self._reset_button.setToolTip("Reset camera")
        self._reset_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self._reset_button.setFlat(True)
        self._reset_button.clicked.connect(self.reset_camera)
        header.addWidget(self._title_label)
        header.addStretch(1)
        header.addWidget(self._status_label)
        header.addWidget(self._reset_button)
        root.addWidget(self._header_widget)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        self._content_layout = content

        sidebar = QWidget()
        sidebar.setMinimumWidth(120)
        sidebar.setMaximumWidth(300)
        sidebar_layout = QVBoxLayout(sidebar)
        self._sidebar_layout = sidebar_layout
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(8)

        self._meta_layout = QFormLayout()
        self._meta_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._meta_layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        self._meta_layout.setHorizontalSpacing(12)
        self._meta_layout.setVerticalSpacing(6)
        sidebar_layout.addLayout(self._meta_layout)
        sidebar_layout.addStretch(1)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#101418")
        add_axes(self.plotter)
        self._observer_id = self.plotter.interactor.AddObserver(
            "MouseMoveEvent",
            self._update_hover_status
        )
        content.addWidget(self.plotter, 1)
        content.addWidget(sidebar)
        root.addLayout(content)

    def set_title(self, title: str) -> None:
        self._title = title
        self._title_label.setText(title)

    def hide_header(self) -> None:
        self._header_widget.hide()

    def set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def set_metadata(self, items: list[tuple[str, str]]) -> None:
        while self._meta_layout.rowCount():
            row = self._meta_layout.takeRow(0)
            if row.labelItem is not None and row.labelItem.widget() is not None:
                row.labelItem.widget().deleteLater()
            if row.fieldItem is not None and row.fieldItem.widget() is not None:
                row.fieldItem.widget().deleteLater()

        if not items:
            return

        for key, value in items:
            label = QLabel(value)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

            self._meta_layout.addRow(key, label)

    def reset_camera(self) -> None:
        if self._default_camera_position is None:
            self.fit_camera()
            return

        self.plotter.camera_position = copy.deepcopy(self._default_camera_position)
        self.plotter.render()

    def fit_camera(self) -> None:
        self.plotter.reset_camera()
        self._default_camera_position = copy.deepcopy(self.plotter.camera_position)
        self.plotter.render()

    def clear_scene(self) -> None:
        if self.plotter is None:
            return

        self.plotter.clear()
        add_axes(self.plotter)

    def _update_hover_status(self, *_args) -> None:
        interactor = self.plotter.interactor
        if interactor is None:
            return
        x_pos, y_pos = interactor.GetEventPosition()
        if self._picker.Pick(x_pos, y_pos, 0, self.plotter.renderer):
            x_world, y_world, z_world = self._picker.GetPickPosition()
            self.set_status(f"Cursor: ({x_world:0.2f}, {y_world:0.2f}, {z_world:0.2f})")

    def cleanup(self):
        try:
            if self._observer_id:
                self.plotter.interactor.RemoveObserver(self._observer_id)
        except Exception:
            pass

        try:
            self.plotter.close()
        except RuntimeError:
            pass