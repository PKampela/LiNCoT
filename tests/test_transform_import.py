"""Integration tests for transform import workflows."""

from pathlib import Path

import numpy as np
import pytest

from cli.main import build_command_registry
from core.point import Point
from core.session import Session


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sample_fif_path() -> Path:
    return _repo_root() / "dataset" / "trans" / "sample_audvis_raw-trans.fif"


def test_fif_transform_workflow_through_session_and_registry() -> None:
    session = Session(subject_id="workflow", description="FIF transform workflow")
    registry = build_command_registry()

    transform_file = _sample_fif_path()
    if not transform_file.exists():
        pytest.skip(f"Sample data not found: {transform_file}")

    transform, info = session.import_transform(str(transform_file))

    assert "Imported transform" in info
    assert transform.matrix.shape == (4, 4)
    assert transform.source.name in session.frames.list_frames()
    assert transform.target.name in session.frames.list_frames()
    assert session.transforms.list_transforms()

    point = Point(np.array([10.0, 20.0, 30.0]), transform.source)
    transformed_point = transform.apply(point)

    assert transformed_point.frame == transform.target
    assert transformed_point.coords.shape == (3,)
    assert not np.any(np.isnan(transformed_point.coords))

    session.add_point("workflow_point", point)

    transform_result = registry.execute(
        session,
        "transform",
        [transform.source.name, transform.target.name, "10", "20", "30"],
        {"json": True, "show_chain": True},
    )
    point_transform_result = registry.execute(
        session,
        "transform",
        ["workflow_point", transform.target.name],
        {},
    )
    frame_list_result = registry.execute(session, "frame.list", [], {})
    transform_list_result = registry.execute(session, "transform.list", [], {})
    point_list_result = registry.execute(session, "point.list", [], {})
    summary_result = registry.execute(session, "session.summary", [], {})

    assert transform_result.output_format == "json"
    assert transform_result.data is not None
    assert transform_result.data["input"]["frame"] == transform.source.name
    assert transform_result.data["output"]["frame"] == transform.target.name
    assert "Transform chain:" in transform_result.message

    assert "Transformed point 'workflow_point'" in point_transform_result.message
    assert frame_list_result.data["frames"]
    assert transform_list_result.data["transforms"]
    assert point_list_result.data["points"][0]["name"] == "workflow_point"
    assert transform.source.name in summary_result.data["frames"]
    assert transform.target.name in summary_result.data["frames"]
    assert summary_result.data["points"] == ["workflow_point"]


def test_xfm_transform_workflow_creates_frames_and_transforms_point(tmp_path) -> None:
    session = Session(subject_id="workflow", description="XFM transform workflow")
    registry = build_command_registry()

    xfm_path = tmp_path / "talairach.xfm"
    xfm_path.write_text(
        "\n".join(
            [
                "MNI Transform File",
                "% avi2talxfm",
                "Transform_Type = Linear;",
                "Linear_Transform = ",
                "1.022485 -0.008449 -0.036217 5.597427",
                "0.071071 0.914866 0.406098 -19.815094",
                "0.008756 -0.433700 1.028119 -1.547623;",
            ]
        ),
        encoding="utf-8",
    )

    transform, info = session.import_transform(
        str(xfm_path),
        source_frame_name="T1_mri",
        target_frame_name="talairach",
    )

    assert transform.source.name == "T1_mri"
    assert transform.target.name == "talairach"
    assert transform.matrix.shape == (4, 4)
    assert "Created source frame" in info
    assert "Created target frame" in info
    assert "avi2talxfm" in info

    point = Point(np.array([1.0, 2.0, 3.0]), session.get_frame("T1_mri"))
    session.add_point("xfm_point", point)

    point_transform_result = registry.execute(
        session,
        "transform",
        ["xfm_point", "talairach"],
        {},
    )
    frame_list_result = registry.execute(session, "frame.list", [], {})
    transform_list_result = registry.execute(session, "transform.list", [], {})
    summary_result = registry.execute(session, "session.summary", [], {})

    assert "Transformed point 'xfm_point'" in point_transform_result.message
    assert "T1_mri" in frame_list_result.data["frames"]
    assert "talairach" in frame_list_result.data["frames"]
    assert "import_talairach" in transform_list_result.data["transforms"][0]
    assert summary_result.data["points"] == ["xfm_point"]
    assert summary_result.data["frames"]