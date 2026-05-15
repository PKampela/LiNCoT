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

        header = QHBoxLayout()
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
        root.addLayout(header)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        self._content_layout = content

        sidebar = QWidget()
        sidebar.setMinimumWidth(240)
        sidebar.setMaximumWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        self._sidebar_layout = sidebar_layout
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(8)

        sidebar_title = QLabel("Workflow Details")
        sidebar_title.setObjectName("viewerSidebarTitle")
        sidebar_layout.addWidget(sidebar_title)

        self._meta_summary = QLabel(
            "This panel shows the active object’s reference frame, size, and TMS-relevant notes."
        )
        self._meta_summary.setWordWrap(True)
        sidebar_layout.addWidget(self._meta_summary)

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
        self.plotter.interactor.AddObserver("MouseMoveEvent", self._update_hover_status)
        content.addWidget(self.plotter, 1)
        content.addWidget(sidebar)
        root.addLayout(content)

    def set_title(self, title: str) -> None:
        self._title = title
        self._title_label.setText(title)

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
            self._meta_summary.setText(
                "No object is loaded yet. Open a volume or surface to see frame, size, and planning details."
            )
            return

        self._meta_summary.setText("Current object metadata and planning notes")
        for key, value in items:
            self._meta_layout.addRow(key, QLabel(value))

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
        self.plotter.clear()
        add_axes(self.plotter)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.plotter.close()
        super().closeEvent(event)

    def _update_hover_status(self, *_args) -> None:
        interactor = self.plotter.interactor
        if interactor is None:
            return
        x_pos, y_pos = interactor.GetEventPosition()
        if self._picker.Pick(x_pos, y_pos, 0, self.plotter.renderer):
            x_world, y_world, z_world = self._picker.GetPickPosition()
            self.set_status(f"Cursor: ({x_world:0.2f}, {y_world:0.2f}, {z_world:0.2f})")