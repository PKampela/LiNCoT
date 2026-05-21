import json
from pathlib import Path

from cli import main as cli_main
from cli.console import Console
from core.frames import CoordinateFrame
from core.transform import Transform
from core.session import Session
from registry.command_registry import CommandRegistry, register_default_commands


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
