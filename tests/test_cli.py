import json
import sys
from pathlib import Path
from io import StringIO

from cli import main as cli_main
from cli.console import Console
from core.session import Session
from registry.command_registry import CommandRegistry, register_default_commands


def test_cli_auto_chain_json(monkeypatch, capsys):
    """Test transform command via interactive CLI with piped input."""
    repo_root = Path(__file__).resolve().parents[1]
    registry_path = repo_root / "transforms.json"
    assert registry_path.exists()
    monkeypatch.chdir(repo_root)

    # Create session and registry
    session = Session(subject_id="test", description="Test session")
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
    # Look for JSON output (last non-empty line should be JSON)
    for line in reversed(output.split('\n')):
        if line.strip():
            try:
                data = json.loads(line)
                assert data["input"]["frame"] == "head"
                assert data["output"]["frame"] == "mni"
                assert data["output"]["coords"] == [0.0, 0.0, 0.0]
                assert [step["name"] for step in data["chain"]] == [
                    "head_to_mri",
                    "mri_to_mni",
                ]
                break
            except json.JSONDecodeError:
                continue
