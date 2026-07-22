"""Graphical console widget that shares command execution with CLI."""

from __future__ import annotations

import json
from typing import Callable

from PySide6.QtWidgets import QLineEdit, QTextEdit, QVBoxLayout, QWidget

from cli.parser import CommandParseError, parse_command
from core.session import Session
from gui.viewer.viewer_manager import ViewerManager
from registry.command_registry import CommandRegistry, CommandExecutionError


class ConsoleWidget(QWidget):
    def __init__(
        self,
        session: Session,
        command_registry: CommandRegistry,
        viewer_manager: ViewerManager | None = None,
        on_session_changed: Callable[[], None] | None = None,
        on_status: Callable[[str, str | None, str], None] | None = None,
    ) -> None:
        super().__init__()
        self._session = session
        self._command_registry = command_registry
        self._viewer_manager = viewer_manager
        self._on_session_changed = on_session_changed
        self._on_status = on_status

        layout = QVBoxLayout(self)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type command, e.g. point add p1 0 0 0 head")

        layout.addWidget(self.output)
        layout.addWidget(self.input)

        self.input.returnPressed.connect(self._on_return_pressed)

    def _append(self, text: str) -> None:
        self.output.append(text)

    def _on_return_pressed(self) -> None:
        raw = self.input.text().strip()
        if not raw:
            return

        self._append(f"> {raw}")

        try:
            parsed = parse_command(raw)
            result = self._command_registry.execute(
                self._session,
                parsed.command,
                parsed.args,
                parsed.kwargs,
            )
            if self._viewer_manager is not None and result.data:
                try:
                    self._viewer_manager.open_from_descriptor(result.data.get("viewer"))
                except Exception as exc:  # pragma: no cover - depends on local GUI stack
                    self._append(f"Viewer error: {exc}")

            if result.output_format == "json" and result.data is not None:
                self._append(json.dumps(result.data, indent=2))
            else:
                for line in result.message.splitlines():
                    self._append(line)

            if self._on_status is not None and result.data and result.data.get("report"):
                self._on_status(
                    result.message.splitlines()[0] if result.message else "Command complete",
                    str(result.data["report"]),
                    "info",
                )

            if self._on_session_changed is not None:
                self._on_session_changed()
        except (CommandParseError, CommandExecutionError, KeyError, ValueError) as exc:
            self._append(f"Error: {exc}")

        self.input.clear()
