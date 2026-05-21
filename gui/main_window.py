"""Main Qt window for TMSCoords GUI."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.session import Session
from core.import_service import preview_xfm_transform, get_file_extension
from gui.console_widget import ConsoleWidget
from gui.session_inspector import SessionInspectorWidget
from gui.viewer.viewer_manager import ViewerManager
from registry.command_registry import CommandRegistry
from registry.transform_registry import TransformRegistry
from core.import_service import format_xfm_preview, preview_xfm_transform, get_file_extension


class XfmImportDialog(QDialog):
    def __init__(self, path: str, metadata_lines: list[str], matrix, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import XFM Transform")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._source_edit = QLineEdit()
        self._target_edit = QLineEdit()
        form.addRow("Source frame", self._source_edit)
        form.addRow("Target frame", self._target_edit)
        layout.addLayout(form)

        preview_label = QPlainTextEdit()
        preview_label.setReadOnly(True)
        preview_label.setPlainText(
            format_xfm_preview(metadata_lines, matrix) if metadata_lines or matrix is not None else "No metadata lines found."
        )
        preview_label.setMinimumHeight(180)
        layout.addWidget(preview_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._path = path

    @property
    def source_frame_name(self) -> str:
        return self._source_edit.text().strip()

    @property
    def target_frame_name(self) -> str:
        return self._target_edit.text().strip()


class MainWindow(QMainWindow):
    def __init__(self, session: Session, command_registry: CommandRegistry) -> None:
        super().__init__()
        self._session = session
        self._command_registry = command_registry

        self.setWindowTitle("TMSCoords")
        self.resize(1400, 900)

        self._build_ui()
        self._build_menus()
        self._refresh_session()
        self._rebuild_object_menus()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        root_splitter = QSplitter(Qt.Orientation.Vertical)
        root_splitter.setChildrenCollapsible(False)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setChildrenCollapsible(False)

        self.viewer_tabs = QTabWidget()
        self.viewer_tabs.setDocumentMode(True)
        self.viewer_tabs.setTabsClosable(True)
        self.viewer_tabs.setMovable(True)
        self.viewer_tabs.tabCloseRequested.connect(self._close_viewer_tab)

        self.viewer_manager = ViewerManager(self.viewer_tabs, self._session)
        self.viewer_manager.create_viewer_tab("base", "Overview")

        self.console_widget = ConsoleWidget(
            self._session,
            self._command_registry,
            viewer_manager=self.viewer_manager,
            on_session_changed=self._on_session_changed,
        )

        top_splitter.addWidget(self.viewer_tabs)
        top_splitter.addWidget(self.console_widget)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 1)

        self.inspector = SessionInspectorWidget()

        root_splitter.addWidget(top_splitter)
        root_splitter.addWidget(self.inspector)
        root_splitter.setStretchFactor(0, 4)
        root_splitter.setStretchFactor(1, 1)

        layout.addWidget(root_splitter)

        # Clear transient status bar message; detailed status shown in inspector header
        self.statusBar().clearMessage()
        # Hide bottom status bar to remove extra bottom padding; messages still shown
        # transiently via statusBar().showMessage but the permanent status is
        # displayed in the inspector header.
        try:
            self.statusBar().setVisible(False)
        except Exception:
            pass

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()

        self._file_menu = menu_bar.addMenu("&File")
        self._session_menu = menu_bar.addMenu("&Session")
        self._view_menu = menu_bar.addMenu("&View")
        self._help_menu = menu_bar.addMenu("&Help")

        self._file_menu.addAction(self._action("Import MRI Image...", self._import_mri_dialog))
        self._file_menu.addAction(self._action("Import Transform...", self._import_transform_dialog))
        self._file_menu.addAction(self._action("Load Transform Registry...", self._load_transform_registry_dialog))
        self._file_menu.addAction(self._action("Save Transform Registry...", self._save_transform_registry_dialog))
        self._file_menu.addAction(self._action("Save Session Report...", self._save_session_report_dialog))
        self._file_menu.addSeparator()
        self._file_menu.addAction(self._action("Exit", self.close))

        self._session_menu.addAction(self._action("Refresh Session", self._refresh_session))
        self._session_menu.addSeparator()
        self._session_menu.addAction(self._action("Show Overview", self.inspector.show_overview))
        self._session_menu.addAction(self._action("Show Points", self.inspector.show_points))
        self._session_menu.addAction(self._action("Show Images", self.inspector.show_images))
        self._session_menu.addAction(self._action("Show Transforms", self.inspector.show_transforms))
        self._session_menu.addAction(self._action("Show Surfaces", self.inspector.show_surfaces))
        self._session_menu.addAction(self._action("Show Frames", self.inspector.show_frames))

        self._view_menu.addAction(self._action("Reset Active View", self._reset_active_view))
        self._view_menu.addAction(self._action("Focus Console", self._focus_console))
        self._view_menu.addSeparator()
        self._open_volume_menu = self._view_menu.addMenu("Open Loaded Volume")
        self._open_surface_menu = self._view_menu.addMenu("Open Loaded Surface")
        self._view_menu.aboutToShow.connect(self._rebuild_object_menus)

        self._help_menu.addAction(self._action("Command Reference", self._show_command_reference))
        self._help_menu.addSeparator()
        self._help_menu.addAction(self._action("About TMSCoords", self._show_about))

    def _action(self, text: str, slot) -> QAction:
        action = QAction(text, self)
        action.triggered.connect(lambda _checked=False: slot())
        return action

    def _rebuild_object_menus(self) -> None:
        self._populate_object_menu(
            self._open_volume_menu,
            self._session.list_images(),
            self._open_volume_viewer,
            "No loaded volumes",
        )
        self._populate_object_menu(
            self._open_surface_menu,
            self._session.list_surfaces(),
            self._open_surface_viewer,
            "No loaded surfaces",
        )

    def _populate_object_menu(self, menu: QMenu, names: list[str], callback, empty_label: str) -> None:
        menu.clear()
        if not names:
            action = menu.addAction(empty_label)
            action.setEnabled(False)
            return

        for name in names:
            menu.addAction(self._action(name, lambda _checked=False, object_name=name: callback(object_name)))

    def _refresh_inspector(self) -> None:
        self.inspector.refresh(self._session)

    def _refresh_session(self) -> None:
        self._refresh_inspector()
        self.viewer_manager.refresh_viewers()
        self._rebuild_object_menus()

    def _on_session_changed(self) -> None:
        self._refresh_session()

    def _import_mri_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import MRI Image",
            str(Path.cwd()),
            "NIfTI volumes (*.nii *.nii.gz);;All files (*.*)",
        )
        if not path:
            return

        self._import_mri_from_path(path)

    def _import_mri_from_path(self, path: str) -> None:
        try:
            _image, info_msg = self._session.import_image(path)
        except Exception as exc:
            self._set_status("MRI import failed", str(exc), timeout=8000, level="error")
            return

        first_line = info_msg.splitlines()[0] if info_msg else "Imported MRI image: " + Path(path).stem
        image_name = first_line.split(":", 1)[-1].strip() if ":" in first_line else Path(path).stem
        try:
            self.viewer_manager.open_volume(image_name)
        except Exception as exc:
            QMessageBox.warning(self, "Import MRI Image", f"Image imported but viewer could not open: {exc}")

        self._refresh_session()
        self._set_status("MRI imported", info_msg, timeout=8000, level="info")

    def _import_transform_dialog(self) -> None:
        """Open file dialog to import a transform."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Transform",
            str(Path.cwd()),
            "Transform files (*.fif *.xfm);;MNE Transform (*.fif);;XFM Transform (*.xfm);;All files (*.*)",
        )
        if not path:
            return

        self._import_transform_from_path(path)

    def _import_transform_from_path(self, path: str) -> None:
        """Import a transform from file into the session.

        Args:
            path: Path to transform file
        """
        try:
            ext = get_file_extension(path)
            if ext == "xfm":
                metadata_lines, matrix = preview_xfm_transform(path)
                dialog = XfmImportDialog(path, metadata_lines, matrix, self)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return

                source_frame_name = dialog.source_frame_name
                target_frame_name = dialog.target_frame_name
                if not source_frame_name or not target_frame_name:
                    QMessageBox.information(
                        self,
                        "Import XFM Transform",
                        "Please provide both source and target frames.",
                    )
                    return

                transform, info_msg = self._session.import_transform(
                    path,
                    source_frame_name=source_frame_name,
                    target_frame_name=target_frame_name,
                )
            else:
                transform, info_msg = self._session.import_transform(path)
        except Exception as exc:
            error_msg = str(exc)
            # Show error in status bar; details available on click
            self._set_status("Import failed", error_msg, timeout=8000, level="error")
            return

        # Update GUI
        self._refresh_session()
        # Show concise success message in status bar; details available on click
        self._set_status("Transform imported", info_msg, timeout=8000, level="info")

    def _load_transform_registry_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Transform Registry",
            str(Path.cwd()),
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return

        try:
            self._session.transforms = TransformRegistry.load(path, self._frame_mapping())
        except Exception as exc:
            QMessageBox.critical(self, "Load Transform Registry", str(exc))
            return

        self._refresh_session()
        self.statusBar().showMessage(f"Loaded transform registry from {path}", 8000)

    def _save_transform_registry_dialog(self) -> None:
        if not self._session.transforms.list_transforms():
            QMessageBox.information(self, "Save Transform Registry", "There are no transforms in the current session.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transform Registry",
            str(Path.cwd() / "transforms.json"),
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return

        try:
            self._session.transforms.save(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Transform Registry", str(exc))
            return

        self.statusBar().showMessage(f"Saved transform registry to {path}", 8000)

    def _save_session_report_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session Report",
            str(Path.cwd() / "session_report.json"),
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return

        report = self._build_session_report()
        try:
            Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Save Session Report", str(exc))
            return

        self.statusBar().showMessage(f"Saved session report to {path}", 8000)

    def _build_session_report(self) -> dict:
        return {
            "subject_id": self._session.subject_id,
            "description": self._session.description,
            "frames": [
                {
                    "name": name,
                    "axes": list(self._session.frames.get_frame(name).axes),
                    "units": self._session.frames.get_frame(name).units,
                    "description": self._session.frames.get_frame(name).description,
                }
                for name in self._session.frames.list_frames()
            ],
            "points": [
                {
                    "name": name,
                    "frame": self._session.get_point(name).frame.name,
                    "coords": self._session.get_point(name).coords.tolist(),
                }
                for name in self._session.list_points()
            ],
            "images": [
                {
                    "name": name,
                    "frame": self._session.get_image(name).frame.name,
                    "shape": [int(value) for value in self._session.get_image(name).shape],
                    "dtype": str(self._session.get_image(name).data.dtype),
                }
                for name in self._session.list_images()
            ],
            "transforms": [
                {
                    "name": name,
                    "source": self._session.transforms.get_transform(name).source.name,
                    "target": self._session.transforms.get_transform(name).target.name,
                    "matrix": self._session.transforms.get_transform(name).matrix.tolist(),
                }
                for name in self._session.transforms.list_transforms()
            ],
            "surfaces": [
                {
                    "name": name,
                    "frame": self._session.get_surface(name).frame.name,
                    "vertices": int(self._session.get_surface(name).vertices.shape[0]),
                    "faces": int(self._session.get_surface(name).faces.shape[0]),
                }
                for name in self._session.list_surfaces()
            ],
        }

    def _frame_mapping(self):
        frames = {name: self._session.frames.get_frame(name) for name in self._session.frames.list_frames()}
        return frames

    def _open_volume_viewer(self, volume_name: str) -> None:
        try:
            self.viewer_manager.open_volume(volume_name)
        except Exception as exc:
            QMessageBox.warning(self, "Open Volume Viewer", str(exc))

    def _open_surface_viewer(self, surface_name: str) -> None:
        try:
            self.viewer_manager.open_surface(surface_name)
        except Exception as exc:
            QMessageBox.warning(self, "Open Surface Viewer", str(exc))

    def _reset_active_view(self) -> None:
        active_viewer = self.viewer_manager.active_viewer
        if active_viewer is None or not hasattr(active_viewer, "reset_camera"):
            return
        active_viewer.reset_camera()

    def _focus_console(self) -> None:
        self.console_widget.input.setFocus()

    def _show_command_reference(self) -> None:
        lines = ["Available commands:", ""]
        for spec in self._command_registry.list_commands():
            lines.append(f"- {spec.help_text}")
        QMessageBox.information(self, "Command Reference", "\n".join(lines))

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About TMSCoords",
            "TMSCoords provides a GUI for viewing neuroimaging volumes, surfaces, transforms, and session state.",
        )

    def _set_status(self, short_msg: str, detailed: str | None = None, timeout: int = 8000, level: str = "info") -> None:
        """Set a concise status message and store detailed info for click-to-expand.

        Args:
            short_msg: Short message displayed in status bar
            detailed: Optional detailed text shown when status is clicked
            timeout: Timeout in milliseconds for the transient status message
            level: "info" or "error" (affects dialog type when clicked)
        """
        # Show short message in the bottom status bar
        try:
            self.statusBar().showMessage(short_msg, timeout)
        except Exception:
            pass

        # Delegate detailed status to the inspector header
        try:
            self.inspector.set_status(short_msg, detailed, level)
        except Exception:
            pass

    def _show_status_details(self) -> None:
        """Delegate showing detailed status information to the inspector."""
        try:
            self.inspector.show_status_details()
        except Exception:
            pass

    def _close_viewer_tab(self, index: int) -> None:
        self.viewer_manager.close_tab(index)
