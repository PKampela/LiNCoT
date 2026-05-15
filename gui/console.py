"""Backward-compatible import for the renamed GUI console widget."""

from .console_widget import ConsoleWidget

ConsolePanel = ConsoleWidget

__all__ = ["ConsoleWidget", "ConsolePanel"]