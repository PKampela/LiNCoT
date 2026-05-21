"""Tests for MRI image import workflow."""

import numpy as np
import pytest
from nibabel.loadsave import save
from nibabel.nifti1 import Nifti1Image

from cli.main import _bootstrap_session, build_command_registry


@pytest.fixture
def session():
    """Fresh session for each test."""
    return _bootstrap_session()


@pytest.fixture
def registry():
    """Fresh registry for each test."""
    return build_command_registry()


@pytest.fixture
def nifti_file(tmp_path):
    """Create a temporary NIfTI file."""
    data = np.ones((10, 10, 10), dtype=np.float32)
    affine = np.eye(4)
    img = Nifti1Image(data, affine)
    path = tmp_path / "test_brain.nii"
    save(img, path)
    return str(path)


@pytest.fixture
def nifti_gz_file(tmp_path):
    """Create a temporary gzipped NIfTI file."""
    data = np.ones((8, 8, 8), dtype=np.float32)
    affine = np.eye(4)
    affine[:3, 3] = [12.0, -6.0, 3.0]
    img = Nifti1Image(data, affine)
    path = tmp_path / "subject01_T1.nii.gz"
    save(img, path)
    return str(path)


def test_mri_import_registers_image_frames_and_transforms(session, nifti_file):
    """Session.import_image loads MRI data and creates subject-specific frames."""
    image, info = session.import_image(nifti_file)

    assert image is not None
    assert image.frame.name == "test_brain_mri"
    assert session.get_image("test_brain") is image
    assert "test_brain_voxel" in session.frames.list_frames()
    assert "test_brain_mri" in session.frames.list_frames()
    assert "test_brain_voxel_to_mri" in session.transforms.list_transforms()
    assert "test_brain_mri_to_voxel" not in session.transforms.list_transforms()
    assert "Imported MRI image: test_brain" in info


def test_mri_import_detects_nii_gz(session, nifti_gz_file):
    """The import router handles .nii.gz files."""
    image, info = session.import_image(nifti_gz_file)

    assert image.frame.name == "subject01_T1_mri"
    assert session.get_image("subject01_T1") is image
    assert "subject01_T1_voxel_to_mri" in session.transforms.list_transforms()
    assert "subject01_T1_mri_to_voxel" not in session.transforms.list_transforms()
    assert "Orientation:" in info


def test_volume_import_command_alias(session, registry, nifti_file):
    """volume.import routes through the same MRI import workflow."""
    result = registry.execute(session, "volume.import", [nifti_file], {})

    assert result.output_format == "text"
    assert result.data["image"]["name"] == "test_brain"
    assert result.data["image"]["frame"] == "test_brain_mri"
    assert session.get_image("test_brain").frame.name == "test_brain_mri"


def test_mri_import_missing_file_raises(session):
    """Importing a missing MRI file fails cleanly."""
    from core.import_service import ImportError

    with pytest.raises(ImportError):
        session.import_image("/nonexistent/path/to/file.nii")
