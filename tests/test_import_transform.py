"""Test transform import workflow."""

from pathlib import Path

import numpy as np
import pytest

from core.frames import CoordinateFrame
from core.import_service import ImportError, UnsupportedFormatError, format_xfm_preview, preview_xfm_transform
from core.point import Point
from core.session import Session


def test_import_transform_fif():
    """Test importing an MNE .fif transform file."""
    session = Session(subject_id="test", description="Import test")
    
    # Import transform from dataset
    transform_file = Path(__file__).parent.parent / "dataset" / "trans" / "sample_audvis_raw-trans.fif"
    if not transform_file.exists():
        pytest.skip(f"Sample data not found: {transform_file}")
    
    transform, info = session.import_transform(str(transform_file))
    
    # Verify transform was created
    assert transform is not None
    assert transform.matrix.shape == (4, 4)
    
    # Verify source and target frames exist
    assert transform.source is not None
    assert transform.target is not None
    
    # Verify frames were registered in session
    assert transform.source.name in session.frames.list_frames()
    assert transform.target.name in session.frames.list_frames()
    
    # Verify transform was registered
    assert len(session.transforms.list_transforms()) > 0
    
    print("✓ Transform import successful")
    print(f"  Imported: {transform.source.name} -> {transform.target.name}")
    print(f"  Info: {info}")


def test_import_nonexistent_file():
    """Test that importing nonexistent file raises error."""
    session = Session(subject_id="test", description="Import test")
    
    with pytest.raises(ImportError):
        session.import_transform("/nonexistent/file.fif")


def test_import_unsupported_format():
    """Test that unsupported formats are rejected."""
    session = Session(subject_id="test", description="Import test")
    
    # Create a dummy .txt file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        temp_path = f.name
    
    try:
        with pytest.raises(UnsupportedFormatError):
            session.import_transform(temp_path)
    finally:
        Path(temp_path).unlink()


def test_import_xfm_transform(tmp_path):
    """Test importing a loosely structured .xfm transform file."""
    session = Session(subject_id="test", description="Import test")

    xfm_path = tmp_path / "talairach.xfm"
    xfm_path.write_text(
        "\n".join(
            [
                "MNI Transform File",
                "% avi2talxfm",
                "",
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
    assert np.allclose(transform.matrix[3], [0.0, 0.0, 0.0, 1.0])
    assert "T1_mri" in session.frames.list_frames()
    assert "talairach" in session.frames.list_frames()
    assert "Created source frame" in info
    assert "Created target frame" in info
    assert "Imported transform" in info
    assert "avi2talxfm" in info


def test_preview_xfm_transform_preserves_metadata(tmp_path):
    """Test reading .xfm metadata lines before import."""
    xfm_path = tmp_path / "talairach.xfm"
    xfm_path.write_text(
        "\n".join(
            [
                "MNI Transform File",
                "% avi2talxfm",
                "Transform_Type = Linear;",
                "Linear_Transform = ",
                "1 0 0 0",
                "0 1 0 0",
                "0 0 1 0",
            ]
        ),
        encoding="utf-8",
    )

    metadata_lines, matrix = preview_xfm_transform(str(xfm_path))

    assert metadata_lines == ["MNI Transform File", "% avi2talxfm", "Transform_Type = Linear;"]
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])


def test_format_xfm_preview_includes_affine_matrix():
    """Test the XFM preview text includes the affine matrix after metadata."""
    metadata_lines = ["MNI Transform File", "% avi2talxfm"]
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0, -6.0],
            [0.0, 0.0, 1.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    preview_text = format_xfm_preview(metadata_lines, matrix)

    assert "MNI Transform File" in preview_text
    assert "Affine matrix:" in preview_text
    assert "  1.000000  0.000000  0.000000  5.000000" in preview_text
    assert "  0.000000  1.000000  0.000000 -6.000000" in preview_text


def test_transform_matrix_validation():
    """Test that invalid matrices are rejected."""
    from core.transform import Transform
    
    # Create dummy frames
    frame1 = CoordinateFrame("frame1", ("R", "A", "S"), "mm")
    frame2 = CoordinateFrame("frame2", ("R", "A", "S"), "mm")
    
    # Test invalid shape - Transform should reject it
    bad_matrix = np.eye(3)  # Wrong shape
    
    with pytest.raises(ValueError):
        Transform(frame1, frame2, bad_matrix)


def test_transform_with_point():
    """Test applying imported transform to a point."""
    session = Session(subject_id="test", description="Import test")
    
    transform_file = Path(__file__).parent.parent / "dataset" / "trans" / "sample_audvis_raw-trans.fif"
    if not transform_file.exists():
        pytest.skip(f"Sample data not found: {transform_file}")
    
    transform, _ = session.import_transform(str(transform_file))
    
    # Create a point in source frame
    source_frame = transform.source
    point = Point(np.array([10.0, 20.0, 30.0]), source_frame)
    
    # Transform it
    transformed_point = transform.apply(point)
    
    # Verify result
    assert transformed_point.frame == transform.target
    assert transformed_point.coords.shape == (3,)
    assert not np.any(np.isnan(transformed_point.coords))
    
    print("✓ Point transformation successful")
    print(f"  Original: {point.coords} in {point.frame.name}")
    print(f"  Transformed: {transformed_point.coords} in {transformed_point.frame.name}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
