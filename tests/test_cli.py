import json
import sys
from pathlib import Path

from ..cli import main as cli_main


def test_cli_auto_chain_json(monkeypatch, capsys):
    repo_root = Path(__file__).resolve().parents[1]
    registry_path = repo_root / "transforms.json"
    assert registry_path.exists()
    monkeypatch.chdir(repo_root)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tmscoords",
            "transform",
            "--point",
            "0",
            "0",
            "0",
            "--from",
            "head",
            "--to",
            "mni",
            "--json",
        ],
    )

    result = cli_main.main()
    assert result == 0

    output = json.loads(capsys.readouterr().out)
    assert output["input"]["frame"] == "head"
    assert output["output"]["frame"] == "mni"
    assert output["output"]["coords"] == [0.0, 0.0, 0.0]
    assert [step["name"] for step in output["chain"]] == [
        "head_to_mri",
        "mri_to_mni",
    ]
