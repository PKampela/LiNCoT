"""Main Qt window for TMSLabs GUI."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.project_manager import ProjectManager
from core.session import Session
from core.import_service import format_xfm_preview, get_file_extension, preview_xfm_transform
from gui.console_widget import ConsoleWidget
from gui.registration_dialog import RegistrationDialog
from gui.session_inspector import SessionInspectorWidget
from gui.viewer.viewer_manager import ViewerManager
from registry.command_registry import CommandRegistry
from registry.transform_registry import TransformRegistry


class XfmImportDialog(QDialog):
    def __init__(self, path: Path, metadata_lines: list[str], matrix, parent: QWidget | None = None) -> None:
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


class SurfaceImportDialog(QDialog):
    def __init__(self, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Surface")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._name_edit = QLineEdit()
        self._frame_edit = QLineEdit()
        form.addRow("Surface name", self._name_edit)
        form.addRow("Frame name", self._frame_edit)
        layout.addLayout(form)

        preview = QPlainTextEdit()
        preview.setReadOnly(True)
        preview.setPlainText(
            "Import a FreeSurfer-style surface such as lh.pial, lh.white, lh.sphere, or outer_skin.surf.\n"
            f"File: {path.name}\n\n"
            "If no name is provided, the loader will infer one from the filename.\n"
            "If no frame is provided, cortical surfaces default to surface_ras and scalp/skull .surf files default to head."
        )
        preview.setMinimumHeight(180)
        layout.addWidget(preview)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._path = path

    @property
    def surface_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def frame_name(self) -> str:
        return self._frame_edit.text().strip()


class MainWindow(QMainWindow):
    def __init__(self, session: Session, command_registry: CommandRegistry) -> None:
        super().__init__()
        self._session = session
        self._command_registry = command_registry
        self._project_manager = ProjectManager()

        self.setWindowTitle("TMSLabs")
        self.resize(1400, 900)

        self._build_ui()
        self._build_menus()
        self._connect_context_menus()

        self._check_recovery()
        self._refresh_session()
        self._rebuild_object_menus()

        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start(300_000)

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

        self.viewer_manager = ViewerManager(
            self.viewer_tabs,
            self._session,
            registration_callback=self._open_registration_dialog,
        )
        self.viewer_manager.create_viewer_tab("base", "Overview")

        self.console_widget = ConsoleWidget(
            self._session,
            self._command_registry,
            viewer_manager=self.viewer_manager,
            on_session_changed=self._on_session_changed,
            on_status=self._set_status,
        )

        top_splitter.addWidget(self.viewer_tabs)
        top_splitter.addWidget(self.console_widget)
        top_splitter.setStretchFactor(0, 4)
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
        self._registration_menu = menu_bar.addMenu("&Registration")
        self._view_menu = menu_bar.addMenu("&View")
        self._help_menu = menu_bar.addMenu("&Help")

        self._file_menu.addAction(self._action("Import MRI Image...", self._import_mri_dialog))
        self._file_menu.addAction(self._action("Import Surface...", self._import_surface_dialog))
        self._file_menu.addAction(self._action("Import Transform...", self._import_transform_dialog))
        self._file_menu.addSeparator()
        self._file_menu.addAction(self._action("New Workspace", self._new_workspace))
        self._file_menu.addAction(self._action("Open Project...", self._open_project_dialog))
        self._file_menu.addAction(self._action("Recent Projects", self._recent_projects))
        self._file_menu.addAction(self._action("Save Project...", self._save_project_dialog))
        self._file_menu.addAction(self._action("Save Project As...", self._save_project_as_dialog))
        self._file_menu.addSeparator()
        self._file_menu.addAction(self._action("Exit", self._exit_application))

        self._session_menu.addAction(self._action("Refresh Session", self._refresh_session))
        self._session_menu.addSeparator()
        self._session_menu.addAction(self._action("Show Overview", self.inspector.show_overview))
        self._session_menu.addAction(self._action("Show Points", self.inspector.show_points))
        self._session_menu.addAction(self._action("Show Images", self.inspector.show_images))
        self._session_menu.addAction(self._action("Show Transforms", self.inspector.show_transforms))
        self._session_menu.addAction(self._action("Show Surfaces", self.inspector.show_surfaces))
        self._session_menu.addAction(self._action("Show Frames", self.inspector.show_frames))

        self._registration_menu.addAction(self._action("Register Images...", self._open_registration_dialog))

        self._view_menu.addAction(self._action("Reset Active View", self._reset_active_view))
        self._view_menu.addAction(self._action("Focus Console", self._focus_console))
        self._view_menu.addSeparator()
        self._open_volume_menu = self._view_menu.addMenu("Open Loaded Volume")
        self._open_surface_menu = self._view_menu.addMenu("Open Loaded Surface")
        self._view_menu.aboutToShow.connect(self._rebuild_object_menus)

        self._help_menu.addAction(self._action("Command Reference", self._show_command_reference))
        self._help_menu.addSeparator()
        self._help_menu.addAction(self._action("About TMSLabs", self._show_about))

    def _connect_context_menus(self) -> None:
        self.inspector.images_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.inspector.images_table.customContextMenuRequested.connect(self._show_image_context_menu)

    def _show_image_context_menu(self, position) -> None:
        index = self.inspector.images_table.indexAt(position)
        if not index.isValid():
            return

        row = index.row()
        item = self.inspector.images_table.item(row, 0)
        if item is None:
            return

        image_name = item.text().strip()
        if not image_name:
            return

        menu = QMenu(self)
        menu.addAction(self._action("Register to...", lambda: self._open_registration_dialog(moving_image_name=image_name)))
        menu.exec(self.inspector.images_table.viewport().mapToGlobal(position))

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

    def _autosave(self) -> None:
        if not self._session.project.is_dirty:
            return

        try:
            self._project_manager.autosave(self._session)
        except Exception as exc:
            self._set_status(
                "Autosave failed",
                str(exc),
                timeout=4000,
                level="warning",
            )

    def _exit_application(self) -> None:
        """Close the application after checking for unsaved work."""

        self.close()


    def closeEvent(self, event: QCloseEvent) -> None:
        print("CLOSE EVENT: entered", flush=True)

        print("CLOSE EVENT: checking unsaved state", flush=True)
        if not self._confirm_discard_unsaved():
            print("CLOSE EVENT: user cancelled", flush=True)
            event.ignore()
            return

        print("CLOSE EVENT: confirmation complete", flush=True)

        print("CLOSE EVENT: starting autosave", flush=True)
        self._project_manager.autosave(self._session)
        print("CLOSE EVENT: autosave complete", flush=True)

        print("CLOSE EVENT: starting viewer shutdown", flush=True)
        self.viewer_manager.close_all_tabs()
        print("CLOSE EVENT: viewer shutdown complete", flush=True)

        print("CLOSE EVENT: accepting event", flush=True)
        event.accept()

        print("CLOSE EVENT: finished", flush=True)

    def _import_mri_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import MRI Image",
            str(Path.cwd()),
            "NIfTI volumes (*.nii *.nii.gz);;All files (*.*)",
        )
        if not paths:
            return

        for path in paths:
            self._import_mri_from_path(Path(path))

    def _new_workspace(self) -> None:

        if not self._confirm_discard_unsaved():
            return

        self._session = Session.create_empty_session()

        self.viewer_manager.close_all_tabs()

        self._refresh_session()
        self._rebuild_object_menus()

        self._update_window_title()

        self._set_status(
            "Workspace created",
            "Started a new workspace.",
            timeout=4000,
        )

    def _confirm_discard_unsaved(self) -> bool:

        if not self._session.project.is_dirty:
            return True

        box = QMessageBox(self)

        box.setWindowTitle("Unsaved changes")

        box.setText(
            "This workspace contains unsaved changes."
        )

        box.setInformativeText(
            "Do you want to save before continuing?"
        )

        box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )

        box.setDefaultButton(
            QMessageBox.StandardButton.Save
        )

        result = box.exec()

        if result == QMessageBox.StandardButton.Save:
            self._save_project_dialog()
            return True
        elif result == QMessageBox.StandardButton.Discard:
            return True
        else:
            return False


    def _recent_projects(self) -> None:
        pass

    def _open_project_dialog(self) -> None:
        """Prompt the user for a project directory."""

        directory = QFileDialog.getExistingDirectory(
            self,
            "Open Project",
            str(Path.cwd()),
        )

        if not directory:
            return

        self._open_project(Path(directory))

    def _open_project(self, project_path: Path) -> None:
        """Load a project from disk."""

        if not self._confirm_discard_unsaved():
            return

        try:
            session = self._project_manager.load(project_path)
        except Exception as exc:
            self._set_status(
                "Project load failed",
                str(exc),
                timeout=8000,
                level="error",
            )
            return

        self._session = session

        try:
            self.viewer_manager.close_all_tabs()
        except Exception:
            pass

        self._refresh_session()
        self._update_window_title()

        if session.load_warnings:
            QMessageBox.warning(
                self,
                "Project Loaded",
                "The project loaded with warnings:\n\n"
                + "\n".join(session.load_warnings),
            )

        self._set_status(
            "Project loaded",
            f"Opened '{session.project.name or project_path.name}'",
            timeout=5000,
            level="info",
        )

    def _save_project_dialog(self) -> None:
        """Save the current project."""

        project_path = self._session.project.project_path

        if project_path is None:
            self._save_project_as_dialog()
            return

        self._save_project(project_path)

    def _save_project_as_dialog(self) -> None:
        """Prompt the user for a project location."""

        directory = QFileDialog.getExistingDirectory(
            self,
            "Save Project",
            str(Path.cwd()),
        )

        if not directory:
            return

        self._save_project(Path(directory))

    def _save_project(self, project_path: Path) -> None:
        """Save the current session as a project."""

        try:
            self._project_manager.save(
                self._session,
                project_path,
                bundle_assets=True,
            )
        except Exception as exc:
            self._set_status(
                "Project save failed",
                str(exc),
                timeout=8000,
                level="error",
            )
            return

        self._update_window_title()
        self._project_manager.clear_recovery()

        self._session.project.is_dirty = False

        self._set_status(
            "Project saved",
            f"Saved '{self._session.project.name}'",
            timeout=5000,
            level="info",
        )

    def _update_window_title(self) -> None:
        """Update the application title."""

        project_name = self._session.project.name

        if not project_name:
            project_name = "Untitled Workspace"

        self.setWindowTitle(f"TMSLabs - {project_name}")

    def _import_mri_from_path(self, path: Path) -> None:
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

    def _import_surface_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Surface",
            str(Path.cwd()),
            "Surface files (*.pial *.white *.sphere *.surf *.inflated *.orig *.smoothwm);;All files (*.*)",
        )
        if not path:
            return

        dialog = SurfaceImportDialog(Path(path), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self._import_surface_from_path(Path(path), surface_name=dialog.surface_name or None, frame_name=dialog.frame_name or None)

    def _import_surface_from_path(self, path: Path, surface_name: str | None = None, frame_name: str | None = None) -> None:
        try:
            surface, name, info_msg = self._session.import_surface(path, frame_name=frame_name, surface_name=surface_name)
        except Exception as exc:
            self._set_status("Surface import failed", str(exc), timeout=8000, level="error")
            return

        first_line = info_msg.splitlines()[0] if info_msg else "Imported surface: " + path.stem
        imported_name = first_line.split(":", 1)[-1].strip() if ":" in first_line else path.stem
        try:
            self.viewer_manager.open_surface(imported_name)
        except Exception as exc:
            QMessageBox.warning(self, "Import Surface", f"Surface imported but viewer could not open: {exc}")

        self._refresh_session()
        self._set_status("Surface imported", info_msg, timeout=8000, level="info")

    def _import_transform_xfm(
        self,
        path: Path,
    ) -> None:

        metadata_lines, matrix = preview_xfm_transform(path)

        dialog = XfmImportDialog(
            path,
            metadata_lines,
            matrix,
            self,
        )

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

        self._set_status(
            "Transform imported",
            info_msg,
            timeout=8000,
            level="info",
        )

    def _import_transform_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Transform",
            str(Path.cwd()),
            "Transform files (*.fif *.xfm);;MNE Transform (*.fif);;XFM Transform (*.xfm);;All files (*.*)",
        )

        if not paths:
            return

        self._import_transforms_from_paths(
            [Path(p) for p in paths]
        )

    def _import_transforms_from_paths(
        self,
        paths: list[Path],
    ) -> None:

        for path in paths:

            try:
                ext = get_file_extension(path)

                if ext == "xfm":
                    self._import_transform_xfm(path)

                else:
                    transform, info_msg = self._session.import_transform(path)

                    self._set_status(
                        "Transform imported",
                        info_msg,
                        timeout=8000,
                        level="info",
                    )

            except Exception as exc:
                self._set_status(
                    "Transform import failed",
                    str(exc),
                    timeout=8000,
                    level="error",
                )

        self._refresh_session()

    def _open_registration_dialog(self, moving_image_name: str | None = None) -> None:
        image_names = self._session.list_images()
        if len(image_names) < 2:
            QMessageBox.information(
                self,
                "Register Images",
                "At least two images are required to run a registration.",
            )
            return

        try:
            dialog = RegistrationDialog(
                image_names,
                moving_image_name=moving_image_name,
                parent=self,
            )
        except ValueError as exc:
            QMessageBox.information(self, "Register Images", str(exc))
            return

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        progress = QProgressDialog("Registering images...", None, 0, 0, self)
        progress.setWindowTitle("Register Images")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        kwargs: dict[str, object] = {
            "quality": dialog.quality,
            "report": True,
        }

        if dialog.transform_name:
            kwargs["name"] = dialog.transform_name

        try:
            result = self._command_registry.execute(
                self._session,
                "register",
                [
                    dialog.moving_image_name,
                    dialog.reference_image_name,
                ],
                kwargs,
            )
        except Exception as exc:
            progress.close()
            self._set_status("Registration failed", str(exc), timeout=8000, level="error")
            return

        progress.close()
        self._refresh_session()

        report = result.data.get("report") if result.data else result.message
        self._set_status("Registration complete", str(report), timeout=8000, level="info")

    def _frame_mapping(self):
        frames = {name: self._session.frames.get_frame(name) for name in self._session.frames.names()}
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
        active_viewer.update()



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
            "About TMSLabs",
            "TMSLabs provides a GUI for viewing neuroimaging volumes, surfaces, transforms, and session state.",
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

    def _check_recovery(self) -> None:
        """Check for autosaved recovery data on startup."""

        if not self._project_manager.has_recovery():
            return

        recovery_path = self._project_manager.default_recovery_path()

        result = QMessageBox.question(
            self,
            "Recover Workspace",
            (
                "A previous workspace was recovered after an unexpected shutdown.\n\n"
                "Would you like to restore it?"
            ),
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
        )

        if result == QMessageBox.StandardButton.Yes:
            try:
                self._session = self._project_manager.load(
                    recovery_path
                )

                self._set_status(
                    "Workspace recovered",
                    "Recovered autosaved workspace",
                    timeout=5000,
                    level="info",
                )

            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Recovery Failed",
                    f"Could not recover workspace:\n\n{exc}",
                )

        else:
            self._project_manager.clear_recovery()
