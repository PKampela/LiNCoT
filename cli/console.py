"""Terminal-specific console behavior for interactive CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Console:
    prompt: str = "tmscoords> "
    history: List[str] = field(default_factory=list)

    def read_input(self) -> str:
        line = input(self.prompt)
        if line.strip():
            self.history.append(line)
        return line

    def print_lines(self, lines: List[str]) -> None:
        for line in lines:
            print(line)

    def print_error(self, message: str) -> None:
        print(f"Error: {message}")
