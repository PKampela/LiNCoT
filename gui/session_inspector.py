"""Bottom-panel inspector for the current TMSLabs session."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.session import Session


class SessionInspectorWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setObjectName("sessionInspector")

        # Header row with title on the left and status on the right
        from PySide6.QtWidgets import QHBoxLayout

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Session Inspector")
        header.setObjectName("sessionInspectorTitle")
        header_row.addWidget(header)

        # Status label aligned to the right of the header
        self._status_label = QLabel("")
        self._status_label.setObjectName("sessionInspectorStatus")
        self._status_label.setToolTip("")
        self._status_label.setText("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # Clicking the status label shows details
        self._status_label.mousePressEvent = lambda event: self.show_status_details()
        header_row.addWidget(self._status_label, 0)

        layout.addLayout(header_row)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.overview = QPlainTextEdit()
        self.overview.setReadOnly(True)
        self.overview.setPlaceholderText("Session summary will appear here.")

        self.points_table = self._make_table(["Name", "Frame", "X", "Y", "Z"])
        self.images_table = self._make_table(["Name", "Voxel Frame", "World Frame", "Shape"])
        self.transforms_table = self._make_table(["Name", "Source", "Target", "Matrix"])
        self.surfaces_table = self._make_table(["Name", "Frame", "Vertices", "Faces"])
        self.frames_table = self._make_table(["Name", "Axes", "Units", "Description"])

        self.images_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.tabs.addTab(self.overview, "Overview")
        self.tabs.addTab(self.points_table, "Points")
        self.tabs.addTab(self.images_table, "Images")
        self.tabs.addTab(self.transforms_table, "Transforms")
        self.tabs.addTab(self.surfaces_table, "Surfaces")
        self.tabs.addTab(self.frames_table, "Frames")

    def _make_table(self, headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    def refresh(self, session: Session) -> None:
        self.overview.setPlainText(self._build_overview_text(session))

        self._populate_table(
            self.points_table,
            [
                [
                    name,
                    session.get_point(name).frame.name,
                    *[f"{value:0.3f}" for value in session.get_point(name).coords.tolist()],
                ]
                for name in session.list_points()
            ],
        )
        rows = []

        for name in session.list_images():
            image = session.get_image(name)

            rows.append([
                name,
                image.voxel_frame.name,
                image.world_frame.name,
                str(tuple(int(v) for v in image.shape)),
            ])

        self._populate_table(self.images_table, rows)
        self._populate_table(
            self.transforms_table,
            [
                [
                    name,
                    session.transforms.get_transform(name).source.name,
                    session.transforms.get_transform(name).target.name,
                    self._matrix_summary(session.transforms.get_transform(name).matrix.tolist()),
                ]
                for name in session.transforms.names()
            ],
        )
        self._populate_table(
            self.surfaces_table,
            [
                [
                    name,
                    session.get_surface(name).frame.name,
                    str(int(session.get_surface(name).vertices.shape[0])),
                    str(int(session.get_surface(name).faces.shape[0])),
                ]
                for name in session.list_surfaces()
            ],
        )
        self._populate_table(
            self.frames_table,
            [
                [
                    name,
                    ", ".join(session.frames.get_frame(name).axes),
                    session.frames.get_frame(name).units,
                    session.frames.get_frame(name).description or "",
                ]
                for name in session.frames.names()
            ],
        )

    def show_overview(self) -> None:
        self.tabs.setCurrentWidget(self.overview)

    def show_points(self) -> None:
        self.tabs.setCurrentWidget(self.points_table)

    def show_images(self) -> None:
        self.tabs.setCurrentWidget(self.images_table)

    def show_transforms(self) -> None:
        self.tabs.setCurrentWidget(self.transforms_table)

    def show_surfaces(self) -> None:
        self.tabs.setCurrentWidget(self.surfaces_table)

    def show_frames(self) -> None:
        self.tabs.setCurrentWidget(self.frames_table)

    def _populate_table(self, table: QTableWidget, rows: list[list[str]]) -> None:
        table.setRowCount(0)
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row_index, column_index, item)
        table.resizeRowsToContents()

    def _build_overview_text(self, session: Session) -> str:
        lines = [
            f"Subject: {session.subject.subject_id or 'unnamed'}",
            f"Description: {session.project.description or 'No description set'}",
            "",
            f"Frames: {len(session.frames.names())}",
            f"Points: {len(session.list_points())}",
            f"Images: {len(session.list_images())}",
            f"Transforms: {len(session.transforms.names())}",
            f"Surfaces: {len(session.list_surfaces())}",
            "",
            "In a real TMS workflow, this panel is used to verify the session context before targeting or navigation:",
            "- subject and session metadata",
            "- which coordinate frames are active",
            "- whether points, surfaces, and images agree on the same anatomy reference",
            "- transform provenance before applying a target or moving a coil",
            "- live geometry counts and object availability",
            "",
            "Use the File menu to load data, and the View menu to open loaded objects in dedicated viewers.",
        ]
        return "\n".join(lines)

    def _matrix_summary(self, matrix: list[list[float]]) -> str:
        rows = [
            "[" + ", ".join(f"{value:0.2f}" for value in row[:4]) + "]"
            for row in matrix[:2]
        ]
        if len(matrix) > 2:
            rows.append("...")
        return " ".join(rows)

    def set_status(self, short_msg: str, detailed: str | None = None, level: str = "info") -> None:
        """Set a concise status message in the inspector header.

        The detailed message is stored and shown when the status label is clicked.
        """
        self._last_status_info = detailed or short_msg
        self._last_status_level = level
        try:
            self._status_label.setText(short_msg)
            self._status_label.setToolTip(detailed or "")
        except Exception:
            pass

    def show_status_details(self) -> None:
        """Show stored detailed status information in an information dialog."""
        if not getattr(self, "_last_status_info", None):
            return
        info = self._last_status_info or ""
        dlg = QDialog(self)
        dlg.setWindowTitle("Status Details")
        layout = QVBoxLayout(dlg)
        message = QLabel(info)
        message.setWordWrap(True)
        layout.addWidget(message)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.setModal(True)
        dlg.exec()