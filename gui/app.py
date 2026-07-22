"""GUI application entrypoint helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from core.session import Session
from gui.main_window import MainWindow
from registry.command_registry import CommandRegistry


def _apply_theme(app: QApplication) -> None:
    tab_close_icon = (Path(__file__).resolve().parent / "assets" / "tab-close.svg").as_posix()

    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#eef2f7"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#e2e8f0"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#0f172a"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#dc2626"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    app.setStyleSheet(
        """
        QMainWindow {
            background: #eef2f7;
        }
        QMenuBar {
            background: #f8fafc;
            border-bottom: 1px solid #cbd5e1;
            spacing: 6px;
            padding: 4px 8px;
        }
        QMenuBar::item {
            padding: 6px 10px;
            border-radius: 6px;
            background: transparent;
        }
        QMenuBar::item:selected {
            background: #dbeafe;
            color: #0f172a;
        }
        QMenu {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            padding: 4px;
        }
        QMenu::item {
            padding: 6px 20px 6px 18px;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background: #e0f2fe;
            color: #0f172a;
        }
        QTabBar {
            qproperty-drawBase: 0;
        }
        QTabWidget::pane {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-top: none;
            border-bottom-left-radius: 10px;
            border-bottom-right-radius: 10px;
        }
        QTabWidget > QWidget {
            background: transparent;
        }
        QTabBar::tab {
            background: #e2e8f0;
            color: #0f172a;
            border: 1px solid #cbd5e1;
            border-bottom: 0;
            padding: 7px 14px;
            padding-right: 26px;
            min-height: 26px;
            margin-right: 4px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            margin-bottom: 1px;
        }
        QTabBar::close-button {
            image: url("%s");
            width: 12px;
            height: 12px;
            margin-right: 8px;
            subcontrol-position: right;
        }
        QTabBar::close-button:hover {
            image: url("%s");
        }
        QSplitter::handle {
            background: #cbd5e1;
        }
        QTextEdit, QPlainTextEdit, QLineEdit, QTableWidget {
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
        }
        QHeaderView::section {
            background: #f1f5f9;
            padding: 6px;
            border: none;
            border-bottom: 1px solid #cbd5e1;
        }
        QLabel#viewerSidebarTitle,
        QLabel#sessionInspectorTitle {
            font-size: 14px;
            font-weight: 600;
            color: #0f172a;
        }
        #sessionInspector QPlainTextEdit,
        #sessionInspector QTableWidget {
            background: transparent;
            border: none;
        }

        #sessionInspector QHeaderView::section {
            background: #f1f5f9;
        }
        QStatusBar {
            background: #f8fafc;
            border-top: 1px solid #cbd5e1;
        }
        """ % (tab_close_icon, tab_close_icon)
    )


def run_app(session: Session, command_registry: CommandRegistry) -> int:
    app = QApplication(sys.argv)
    _apply_theme(app)

    window = MainWindow(session=session, command_registry=command_registry)
    window.show()

    return app.exec()
