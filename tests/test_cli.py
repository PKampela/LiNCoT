import json
from pathlib import Path

import numpy as np

from cli import main as cli_main
from cli.console import Console
from core.frames import CoordinateFrame
from core.transform import Transform
from core.session import Session
from registry.command_registry import CommandRegistry, register_default_commands
from core.image import Image
from core.registration import RegistrationReport


def test_cli_auto_chain_json(monkeypatch, capsys):
    """Test transform command via interactive CLI with piped input."""
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)

    # Create session and registry
    session = Session(subject_id="test", description="Test session")
    session.add_frame(CoordinateFrame("head", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mri", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mni", ("R", "A", "S"), "mm"))
    head = session.get_frame("head")
    mri = session.get_frame("mri")
    mni = session.get_frame("mni")
    session.add_transform(
        "head_to_mri",
        Transform(head, mri, [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]),
    )
    session.add_transform(
        "mri_to_mni",
        Transform(mri, mni, [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3], [0, 0, 0, 1]]),
    )
    registry = CommandRegistry()
    register_default_commands(registry)
    
    # Create mock console with piped input
    class MockConsole(Console):
        def __init__(self):
            self.commands = [
                "transform head mni 0 0 0 --json",
                "exit"
            ]
            self.command_index = 0
        
        def read_input(self):
            if self.command_index >= len(self.commands):
                raise EOFError()
            cmd = self.commands[self.command_index]
            self.command_index += 1
            return cmd
        
        def print_lines(self, lines):
            for line in lines:
                print(line)
        
        def print_error(self, msg):
            print(f"Error: {msg}")
    
    # Run the CLI
    mock_console = MockConsole()
    result = cli_main.run_interactive_cli(session, registry, mock_console)
    assert result == 0
    
    # Check output
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.strip()]
    start_index = next(index for index, line in enumerate(lines) if line.strip() == "{")
    end_index = len(lines) - 1 - next(index for index, line in enumerate(reversed(lines)) if line.strip() == "}")
    data = json.loads("\n".join(lines[start_index : end_index + 1]))

    assert data["input"]["frame"] == "head"
    assert data["output"]["frame"] == "mni"
    assert data["output"]["coords"] == [0.0, 0.0, 0.0]
    assert [step["name"] for step in data["chain"]] == [
        "head_to_mri",
        "mri_to_mni",
    ]


def test_cli_register_command_reports_and_stores_transform(monkeypatch, capsys):
    """Interactive CLI should execute registration commands and print reports."""
    session = Session(subject_id="test", description="Registration CLI test")
    session.add_frame(CoordinateFrame("head", ("R", "A", "S"), "mm"))
    session.add_frame(CoordinateFrame("mri", ("R", "A", "S"), "mm"))

    moving = Image(np.ones((2, 2, 2), dtype=float), np.eye(4), session.get_frame("head"))
    reference = Image(np.ones((2, 2, 2), dtype=float) * 2.0, np.eye(4), session.get_frame("mri"))
    session.add_image("moving", moving)
    session.add_image("reference", reference)

    def fake_register_images(moving_image, reference_image, mode, quality):
        assert moving_image is moving
        assert reference_image is reference
        assert mode == "affine"
        assert quality == "standard"
        return (
            Transform(moving.frame, reference.frame, np.eye(4)),
            RegistrationReport(
                mode=mode,
                quality=quality,
                iterations=12,
                similarity=0.81,
                translation_mm=4.2,
                rotation_deg=1.5,
            ),
        )

    monkeypatch.setattr("registry.command_registry.register_images", fake_register_images)

    class MockConsole(Console):
        def __init__(self):
            self.commands = [
                "register affine moving reference --report --name cli_registration",
                "exit",
            ]
            self.command_index = 0

        def read_input(self):
            if self.command_index >= len(self.commands):
                raise EOFError()
            cmd = self.commands[self.command_index]
            self.command_index += 1
            return cmd

        def print_lines(self, lines):
            for line in lines:
                print(line)

        def print_error(self, msg):
            print(f"Error: {msg}")

    registry = CommandRegistry()
    register_default_commands(registry)
    result = cli_main.run_interactive_cli(session, registry, MockConsole())
    assert result == 0
    assert session.transforms.get_transform("cli_registration").source.name == "head"

    output = capsys.readouterr().out
    assert "Registered affine transform 'cli_registration'" in output
    assert "Iterations: 12" in output
    assert "Similarity: 0.81" in output
