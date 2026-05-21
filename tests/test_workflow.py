"""Integration-style workflow tests for the current command surface."""

from pathlib import Path

import numpy as np

from cli.main import build_command_registry
from core.import_service import import_transform
from core.point import Point
from core.session import Session


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_transform_workflow_covers_point_and_cli_commands() -> None:
    session = Session(subject_id="workflow", description="Transform import workflow")
    registry = build_command_registry()

    fif_path = _repo_root() / "dataset" / "trans" / "sample_audvis_raw-trans.fif"
    transform, info = import_transform(session, str(fif_path))

    assert "Imported transform" in info
    assert transform.source.name in session.frames.list_frames()
    assert transform.target.name in session.frames.list_frames()
    assert session.transforms.list_transforms()

    point = Point(np.array([10.0, 20.0, 30.0]), transform.source)
    transformed_point = transform.apply(point)

    assert transformed_point.frame == transform.target
    assert transformed_point.coords.shape == (3,)

    add_result = registry.execute(
        session,
        "point.add",
        ["workflow_point", "10", "20", "30", transform.source.name],
        {},
    )
    transform_result = registry.execute(
        session,
        "transform",
        ["workflow_point", transform.target.name],
        {},
    )
    point_list_result = registry.execute(session, "point.list", [], {})
    frame_list_result = registry.execute(session, "frame.list", [], {})
    transform_list_result = registry.execute(session, "transform.list", [], {})
    summary_result = registry.execute(session, "session.summary", [], {})
    help_result = registry.execute(session, "help", [], {})

    assert "Added point 'workflow_point'" in add_result.message
    assert "Transformed point 'workflow_point'" in transform_result.message
    assert "workflow_point" in point_list_result.message
    assert transform.source.name in frame_list_result.data["frames"]
    assert transform.target.name in frame_list_result.data["frames"]
    assert transform_list_result.data["transforms"]
    assert summary_result.data["points"] == ["workflow_point"]
    assert transform.source.name in summary_result.data["frames"]
    assert transform.target.name in summary_result.data["frames"]
    assert "volume.import" in help_result.message
    assert "view.volume" in help_result.message


def test_volume_workflow_registers_image_frames_and_view_descriptor() -> None:
    session = Session(subject_id="workflow", description="Volume import workflow")
    registry = build_command_registry()

    nifti_path = _repo_root() / "dataset" / "mri" / "T1.nii.gz"
    result = registry.execute(session, "volume.import", [str(nifti_path)], {})

    assert result.data is not None
    assert result.data["image"]["name"] == "T1"
    assert result.data["image"]["frame"] == "T1_mri"
    assert session.get_image("T1").frame.name == "T1_mri"
    assert "T1_voxel" in session.frames.list_frames()
    assert "T1_mri" in session.frames.list_frames()
    assert "T1_voxel_to_mri" in session.transforms.list_transforms()

    view_result = registry.execute(session, "view.volume", ["T1"], {})
    frame_list_result = registry.execute(session, "frame.list", [], {})
    transform_list_result = registry.execute(session, "transform.list", [], {})
    summary_result = registry.execute(session, "session.summary", [], {})

    assert view_result.data is not None
    assert view_result.data["viewer"] == {"type": "volume", "name": "T1"}
    assert view_result.data["volume"]["frame"] == "T1_mri"
    assert "T1_voxel" in frame_list_result.data["frames"]
    assert "T1_mri" in frame_list_result.data["frames"]
    assert "T1_voxel_to_mri" in transform_list_result.data["transforms"]
    assert summary_result.data["images"] == ["T1"]