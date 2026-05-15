"""Import service for loading transforms and MRI images from file formats.

Routes file imports to appropriate backends based on extension.
Handles format detection, subject-specific frame naming, and validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .image import Image
    from .session import Session
    from .transform import Transform


class ImportError(Exception):
    """Raised when import fails."""

    pass


class UnsupportedFormatError(ImportError):
    """Raised when file format is not supported."""

    pass


def get_file_extension(path: str) -> str:
    """Extract a normalized file extension without leading dot.

    Handles double extensions such as .nii.gz.
    """
    name = Path(path).name.lower()
    if name.endswith(".nii.gz"):
        return "nii.gz"
    return Path(path).suffix.lstrip(".").lower()


def _strip_known_extensions(path: str) -> str:
    """Return a stable stem for image and transform files."""

    name = Path(path).name
    lower_name = name.lower()
    if lower_name.endswith(".nii.gz"):
        return name[:-7]
    if lower_name.endswith(".nii"):
        return name[:-4]
    return Path(path).stem


def _unique_name(preferred: str, existing: list[str]) -> str:
    if preferred not in existing:
        return preferred
    index = 1
    while f"{preferred}_{index}" in existing:
        index += 1
    return f"{preferred}_{index}"


def import_transform(session: Session, path: str) -> tuple[Transform, str]:
    """Import a transform from file into session.

    Routes to appropriate backend based on file extension.
    Validates and registers the transform in the session.

    Args:
        session: The active session
        path: Path to transform file

    Returns:
        Tuple of (Transform object, info message)

    Raises:
        UnsupportedFormatError: If file format is not supported
        ImportError: If import or validation fails
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise ImportError(f"File not found: {path}")

    ext = get_file_extension(path)

    if ext == "fif":
        return _import_transform_fif(session, path)
    else:
        raise UnsupportedFormatError(
            f"Unsupported transform format: .{ext}\n"
            f"Supported formats: .fif"
        )


def import_image(session: Session, path: str) -> tuple["Image", str]:
    """Import an MRI image into the session.

    Routes NIfTI files to the NiBabel backend, creates subject-specific
    coordinate frames and voxel->MRI transforms, and registers the image.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise ImportError(f"File not found: {path}")

    ext = get_file_extension(path)
    if ext in {"nii", "nii.gz"}:
        return _import_nifti_image(session, path)
    raise UnsupportedFormatError(
        f"Unsupported MRI image format: .{ext}\n"
        f"Supported formats: .nii, .nii.gz"
    )


def _import_transform_fif(session: Session, path: str) -> tuple[Transform, str]:
    """Import MNE .fif transform file.

    Args:
        session: The active session
        path: Path to .fif file

    Returns:
        Tuple of (Transform object, info message)

    Raises:
        ImportError: If import or validation fails
    """
    try:
        from backends.mne_backend import load_transform_with_frame_mapping
    except ImportError as exc:
        raise ImportError(f"MNE backend not available: {exc}") from exc

    try:
        transform, source_frame_name, target_frame_name = load_transform_with_frame_mapping(path, session.frames)
    except Exception as exc:
        raise ImportError(f"Failed to load MNE transform: {exc}") from exc

    # Validate transform
    _validate_transform(transform)

    # Generate a name for the transform
    file_stem = Path(path).stem
    transform_name = f"import_{file_stem}"

    # Check for name collision
    if transform_name in session.transforms.list_transforms():
        i = 1
        while f"{transform_name}_{i}" in session.transforms.list_transforms():
            i += 1
        transform_name = f"{transform_name}_{i}"

    # Register in session
    session.transforms.register_transform(transform_name, transform)

    info_msg = (
        f"Imported transform: {transform_name}\n"
        f"  Source: {source_frame_name}\n"
        f"  Target: {target_frame_name}\n"
        f"  File: {Path(path).name}"
    )

    return transform, info_msg


def _import_nifti_image(session: Session, path: str) -> tuple["Image", str]:
    """Import a NIfTI MRI image with subject-specific frames and transforms."""

    try:
        from backends.nibabel_backend import load_nifti, load_nifti_image, voxel_to_world_transform
    except ImportError as exc:
        raise ImportError(f"NiBabel backend not available: {exc}") from exc

    info = load_nifti(path)
    image_name = _unique_name(_strip_known_extensions(path), session.list_images())

    voxel_frame_name = f"{image_name}_voxel"
    mri_frame_name = f"{image_name}_mri"

    voxel_frame = session.frames.get_frame(voxel_frame_name) if voxel_frame_name in session.frames.list_frames() else None
    if voxel_frame is None:
        from .frames import CoordinateFrame

        voxel_frame = CoordinateFrame(
            name=voxel_frame_name,
            axes=("i", "j", "k"),
            units="voxel",
            description=f"Voxel coordinates for {image_name}",
        )
        session.add_frame(voxel_frame)

    mri_frame = session.frames.get_frame(mri_frame_name) if mri_frame_name in session.frames.list_frames() else None
    if mri_frame is None:
        from .frames import CoordinateFrame

        mri_frame = CoordinateFrame(
            name=mri_frame_name,
            axes=("R", "A", "S"),
            units="mm",
            description=f"MRI coordinates for {image_name}",
        )
        session.add_frame(mri_frame)

    image = load_nifti_image(path, mri_frame)
    session.add_image(image_name, image)

    transform = voxel_to_world_transform(info, voxel_frame, mri_frame)
    _validate_transform(transform)

    forward_name = _unique_name(f"{image_name}_voxel_to_mri", session.transforms.list_transforms())
    session.add_transform(forward_name, transform)

    inverse_name = _unique_name(f"{image_name}_mri_to_voxel", session.transforms.list_transforms())
    session.add_transform(inverse_name, transform.invert())

    try:
        import nibabel as nib

        orientation = "".join(nib.orientations.aff2axcodes(np.asarray(image.affine, dtype=float)))
    except Exception:
        orientation = "unknown"

    voxel_size = np.linalg.norm(np.asarray(image.affine, dtype=float)[:3, :3], axis=0)
    voxel_size = np.where(voxel_size == 0.0, 1.0, voxel_size)

    info_msg = (
        f"Imported MRI image: {image_name}\n"
        f"  File: {Path(path).name}\n"
        f"  Frame: {mri_frame.name}\n"
        f"  Voxel frame: {voxel_frame.name}\n"
        f"  Dimensions: {image.shape[0]} x {image.shape[1]} x {image.shape[2]}\n"
        f"  Voxel size: {voxel_size[0]:0.3f} x {voxel_size[1]:0.3f} x {voxel_size[2]:0.3f} mm\n"
        f"  Orientation: {orientation}\n"
        f"  Data type: {image.data.dtype}\n"
        f"  Intensity range: {float(np.min(image.data)):0.3f} -> {float(np.max(image.data)):0.3f}\n"
        f"  Transform: {forward_name}"
    )
    return image, info_msg


def _validate_transform(transform: Transform) -> None:
    """Validate transform matrix properties.

    Args:
        transform: The transform to validate

    Raises:
        ImportError: If validation fails
    """
    import numpy as np

    matrix = transform.matrix

    # Check shape
    if matrix.shape != (4, 4):
        raise ImportError(
            f"Invalid transform matrix shape: {matrix.shape}. Expected (4, 4)"
        )

    # Check bottom row (should be [0, 0, 0, 1])
    expected_bottom = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(matrix[3, :], expected_bottom):
        raise ImportError(
            f"Invalid homogeneous transform: bottom row is {matrix[3, :]}, "
            f"expected {expected_bottom}"
        )

    # Check determinant (should be non-zero and not extremely small)
    rotation_part = matrix[:3, :3]
    det = np.linalg.det(rotation_part)

    if abs(det) < 1e-6:
        raise ImportError(
            f"Transform is singular or near-singular (det={det:.2e}). "
            f"This may cause numerical issues."
        )

    # Warn if determinant is far from 1 (indicates scaling)
    if abs(det - 1.0) > 0.1:
        # Don't fail, just log - scaling might be intentional
        pass

    # Check for NaN or Inf
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        raise ImportError("Transform matrix contains NaN or Inf values")


__all__ = [
    "import_transform",
    "import_image",
    "get_file_extension",
    "ImportError",
    "UnsupportedFormatError",
]
