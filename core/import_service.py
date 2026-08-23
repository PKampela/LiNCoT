"""Import service for loading transforms and MRI images from file formats.

Routes file imports to appropriate backends based on extension.
Handles format detection, subject-specific frame naming, and validation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from mne import surface
import numpy as np

from core.asset import AssetMetadata

from .frames import CoordinateFrame
from .transform import Transform
from .image import Image
from .point import Point
from .session import Session
from .surface import Surface

if TYPE_CHECKING:
    from .image import Image
    from .session import Session


class ImportError(Exception):
    """Raised when import fails."""

    pass


class UnsupportedFormatError(ImportError):
    """Raised when file format is not supported."""

    pass


_SURFACE_EXTENSIONS = {"pial", "white", "sphere", "surf", "inflated", "orig", "smoothwm"}


def get_file_extension(path: Path) -> str:
    """Extract a normalized file extension without leading dot.

    Handles double extensions such as .nii.gz.
    """
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return "nii.gz"
    return path.suffix.lstrip(".").lower()


def _strip_known_extensions(path: Path) -> str:
    """Return a stable stem for image and transform files."""

    name = path.name
    lower_name = name.lower()
    if lower_name.endswith(".nii.gz"):
        return name[:-7]
    if lower_name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _infer_surface_name(path: Path) -> str:
    name = path.name
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "surf":
        return path.stem
    return name.replace(".", "_")


def _infer_surface_frame_name(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    stem = path.stem.lower()
    if suffix == "surf" and any(token in stem for token in ("skin", "skull")):
        return "head"
    return "surface_ras"


def _unique_name(preferred: str, existing: list[str]) -> str:
    if preferred not in existing:
        return preferred
    index = 1
    while f"{preferred}_{index}" in existing:
        index += 1
    return f"{preferred}_{index}"

def create_asset_metadata(
    path: Path,
    *,
    linked: bool = True,
    imported_by: str = "user",
) -> AssetMetadata:
    return AssetMetadata(
        source_path=path.resolve(),
        linked=linked,
        imported_by=imported_by,
    )


def _extract_xfm_payload(path: Path) -> tuple[list[str], np.ndarray]:
    """Extract metadata lines and the affine matrix from an .xfm file.

    The file format is loosely structured: top metadata/comment lines are preserved
    for display, then the Linear_Transform block is parsed into a 4x4 affine.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    metadata_lines: list[str] = []
    matrix_rows: list[list[float]] = []
    in_linear_block = False

    for raw_line in lines:
        line = raw_line.strip()
        if not in_linear_block:
            if line.startswith("Linear_Transform"):
                in_linear_block = True
            else:
                metadata_lines.append(raw_line)
            continue

        if not line or line.startswith("#") or line.startswith(";"):
            continue

        if line.endswith(";"):
            line = line[:-1].strip()

        if not line:
            continue

        values = [float(value) for value in re.split(r"\s+", line) if value]
        if values:
            matrix_rows.append(values)

    if not matrix_rows:
        raise ImportError(f"No Linear_Transform matrix found in .xfm file: {path}")

    row_lengths = {len(row) for row in matrix_rows}
    if row_lengths - {4}:
        raise ImportError(
            f"Invalid .xfm transform rows in {path.name}: expected rows of 4 values"
        )

    if len(matrix_rows) == 3:
        matrix_rows.append([0.0, 0.0, 0.0, 1.0])
    elif len(matrix_rows) != 4:
        raise ImportError(
            f"Invalid .xfm transform in {path.name}: expected 3 or 4 rows, got {len(matrix_rows)}"
        )

    matrix = np.asarray(matrix_rows, dtype=float)
    if matrix.shape != (4, 4):
        raise ImportError(
            f"Invalid .xfm transform in {path.name}: expected a 4x4 matrix, got {matrix.shape}"
        )

    return metadata_lines, matrix


def import_transform(
    session: Session,
    path: Path,
    source_frame_name: str | None = None,
    target_frame_name: str | None = None,
) -> tuple[Transform, str]:
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
    if not path.exists():
        raise ImportError(f"File not found: {path}")

    ext = get_file_extension(path)

    if ext == "fif":
        return _import_transform_fif(session, path)
    if ext == "xfm":
        if source_frame_name is None or target_frame_name is None:
            raise ImportError(
                "XFM transforms require explicit source and target frames"
            )
        return _import_transform_xfm(session, path, source_frame_name, target_frame_name)
    else:
        raise UnsupportedFormatError(
            f"Unsupported transform format: .{ext}\n"
            f"Supported formats: .fif, .xfm"
        )

def import_multiple_transforms(
    session: Session,
    paths: list[Path],
    source_frame_name: str | None = None,
    target_frame_name: str | None = None,
) -> list[tuple[Transform, str]]:
    """Import multiple transforms from files into the session."""
    transforms = []
    for path in paths:
        transform, info = import_transform(
            session,
            path,
            source_frame_name=source_frame_name,
            target_frame_name=target_frame_name
        )
        transforms.append((transform, info))
    return transforms


def import_image(session: Session, path: Path) -> tuple["Image", str]:
    """Import an MRI image into the session.

    Routes NIfTI files to the NiBabel backend, creates subject-specific
    coordinate frames and voxel->MRI transforms, and registers the image.
    """
    if not path.exists():
        raise ImportError(f"File not found: {path}")

    ext = get_file_extension(path)
    if ext in {"nii", "nii.gz"}:
        return _import_nifti_image(session, path)
    raise UnsupportedFormatError(
        f"Unsupported MRI image format: .{ext}\n"
        f"Supported formats: .nii, .nii.gz"
    )

def import_multiple_images(session: Session, paths: list[Path]) -> list[tuple["Image", str]]:
    """Import multiple MRI images from files into the session."""
    images = []
    for path in paths:
        image, info = import_image(session, path)
        images.append((image, info))
    return images


def import_surface(
    session: Session,
    path: Path,
    frame_name: str | None = None,
    surface_name: str | None = None,
) -> tuple["Surface", str, str]:
    """Import a triangular surface mesh into the session."""

    if not path.exists():
        raise ImportError(f"File not found: {path}")

    ext = get_file_extension(path)
    if ext not in _SURFACE_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported surface format: .{ext}\n"
            f"Supported formats: .pial, .white, .sphere, .surf, .inflated, .orig, .smoothwm"
        )

    inferred_name = surface_name or _infer_surface_name(path)
    inferred_frame_name = frame_name or _infer_surface_frame_name(path)

    try:
        from backends.surface_backend import load_surface_geometry
    except ImportError as exc:
        raise ImportError(f"Surface backend not available: {exc}") from exc

    frame_created = False
    try:
        frame = session.frames.get_frame(inferred_frame_name)
    except KeyError:
        frame = CoordinateFrame(
            name=inferred_frame_name,
            axes=("R", "A", "S"),
            units="mm",
            description=f"Imported surface frame from {path.name}",
        )
        session.frames.add_frame(frame)
        frame_created = True

    try:
        vertices, faces = load_surface_geometry(path)
        surface = Surface(
            vertices=vertices,
            faces=faces,
            frame=frame,
            asset=create_asset_metadata(path)
        )
    except Exception as exc:
        raise ImportError(f"Failed to load surface: {exc}") from exc

    

    register_name = _unique_name(inferred_name, session.surfaces.names_all())
    session.add_surface(register_name, surface)

    info_lines = [
        f"Imported surface: {register_name}",
        f"  Frame: {surface.frame.name}",
        f"  File: {path.name}",
        f"  Vertices: {surface.vertices.shape[0]}",
        f"  Faces: {surface.faces.shape[0]}",
    ]
    if frame_created:
        info_lines.insert(1, f"  Created frame: {frame.name}")
    if surface_name is None:
        info_lines.append(f"  Inferred name from file: {inferred_name}")
    if frame_name is None:
        info_lines.append(f"  Inferred frame from file: {inferred_frame_name}")

    return surface, register_name, "\n".join(info_lines)

def import_multiple_surfaces(
    session: Session,
    paths: list[Path],
    frame_names: list[str] | None = None,
    surface_names: list[str] | None = None,
) -> list[tuple["Surface", str, str]]:
    """Import multiple triangular surface meshes into the session."""
    surfaces = []
    for i, path in enumerate(paths):
        surface_name = surface_names[i] if surface_names and i < len(surface_names) else None
        frame_name = frame_names[i] if frame_names and i < len(frame_names) else None
        surface, register_name, info = import_surface(
            session,
            path,
            frame_name=frame_name,
            surface_name=surface_name
        )
        surfaces.append((surface, register_name, info))
    return surfaces


def _import_transform_fif(session: Session, path: Path) -> tuple[Transform, str]:
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
        source_frame, target_frame, matrix, source_frame_name, target_frame_name = load_transform_with_frame_mapping(path, session.frames)
        transform = Transform(source=source_frame, target=target_frame, matrix=matrix, asset=create_asset_metadata(path))
    except Exception as exc:
        raise ImportError(f"Failed to load MNE transform: {exc}") from exc

    # Validate transform
    _validate_transform(transform)

    # Generate a name for the transform
    file_stem = path.stem
    transform_name = f"import_{file_stem}"

    # Check for name collision
    if transform_name in session.transforms.names():
        i = 1
        while f"{transform_name}_{i}" in session.transforms.names():
            i += 1
        transform_name = f"{transform_name}_{i}"

    # Register in session
    session.transforms.register_transform(transform_name, transform)

    info_msg = (
        f"Imported transform: {transform_name}\n"
        f"  Source: {source_frame_name}\n"
        f"  Target: {target_frame_name}\n"
        f"  File: {path.name}"
    )

    return transform, info_msg


def preview_xfm_transform(path: Path) -> tuple[list[str], np.ndarray]:
    """Read an .xfm file without importing it, for GUI preview/prompting."""

    if not path.exists():
        raise ImportError(f"File not found: {path}")

    if get_file_extension(path) != "xfm":
        raise UnsupportedFormatError(f"Unsupported transform format preview: {path.suffix}")

    return _extract_xfm_payload(path)


def format_xfm_preview(metadata_lines: list[str], matrix: np.ndarray) -> str:
    """Format .xfm metadata and affine matrix for display."""

    preview_lines = [line for line in metadata_lines if line.strip()]
    preview_lines.append("Affine matrix:")
    preview_lines.extend(
        "  " + " ".join(f"{value: .6f}" for value in row)
        for row in matrix
    )
    return "\n".join(preview_lines)


def _import_transform_xfm(
    session: Session,
    path: Path,
    source_frame_name: str,
    target_frame_name: str,
) -> tuple[Transform, str]:
    """Import a loosely structured .xfm transform file."""

    metadata_lines, matrix = _extract_xfm_payload(path)

    def _get_or_create_frame(frame_name: str, role: str) -> tuple[CoordinateFrame, bool]:
        try:
            return session.get_frame(frame_name), False
        except KeyError:
            frame = CoordinateFrame(
                name=frame_name,
                axes=("R", "A", "S"),
                units="mm",
                description=f"Created for imported .xfm {role} frame",
            )
            session.add_frame(frame)
            return frame, True

    source_frame, source_created = _get_or_create_frame(source_frame_name, "source")
    target_frame, target_created = _get_or_create_frame(target_frame_name, "target")

    transform = Transform(source=source_frame, target=target_frame, matrix=matrix, asset=create_asset_metadata(path))
    _validate_transform(transform)

    transform_name = _unique_name(f"import_{path.stem}", session.transforms.names())
    session.add_transform(transform_name, transform)

    info_lines = [
        f"Imported transform: {transform_name}",
        f"  Source: {source_frame_name}",
        f"  Target: {target_frame_name}",
        f"  File: {path.name}",
    ]
    if source_created:
        info_lines.append(f"  Created source frame: {source_frame_name}")
    if target_created:
        info_lines.append(f"  Created target frame: {target_frame_name}")
    if metadata_lines:
        info_lines.append("  Metadata:")
        info_lines.extend([f"    {line}" for line in metadata_lines if line.strip()])
    info_lines.append("  Affine matrix:")
    info_lines.extend([f"    {' '.join(f'{value: .6f}' for value in row)}" for row in matrix])

    return transform, "\n".join(info_lines)


def _import_nifti_image(session: Session, path: Path) -> tuple["Image", str]:
    """Import a NIfTI MRI image with subject-specific frames and transforms."""

    try:
        from backends.nibabel_backend import load_nifti_image, voxel_to_world_transform
    except ImportError as exc:
        raise ImportError(f"NiBabel backend not available: {exc}") from exc

    image_name = _unique_name(_strip_known_extensions(path), session.list_images())

    voxel_frame_name = f"{image_name}_voxel"
    mri_frame_name = f"{image_name}_mri"

    voxel_frame = session.frames.get_frame(voxel_frame_name) if voxel_frame_name in session.frames.names() else None
    if voxel_frame is None:
        from .frames import CoordinateFrame

        voxel_frame = CoordinateFrame(
            name=voxel_frame_name,
            axes=("i", "j", "k"),
            units="voxel",
            description=f"Voxel coordinates for {image_name}",
        )
        session.add_frame(voxel_frame)

    mri_frame = session.frames.get_frame(mri_frame_name) if mri_frame_name in session.frames.names() else None
    if mri_frame is None:
        from .frames import CoordinateFrame

        mri_frame = CoordinateFrame(
            name=mri_frame_name,
            axes=("R", "A", "S"),
            units="mm",
            description=f"MRI coordinates for {image_name}",
        )
        session.add_frame(mri_frame)

    data, affine = load_nifti_image(path)
    image = Image(
        data=data,
        affine=affine,
        voxel_frame=voxel_frame,
        world_frame=mri_frame,
        asset=create_asset_metadata(path)
    )
    if image.voxel_frame.units != "voxel":
        raise ValueError("Image voxel frame must use voxel coordinates.")

    if image.world_frame.units != "mm":
        raise ValueError("Image world frame must use millimetres.")

    session.add_image(image_name, image)

    transform = voxel_to_world_transform(
        image,
        image.voxel_frame,
        image.world_frame,
    )
    _validate_transform(transform)

    forward_name = _unique_name(f"{image_name}_voxel_to_mri", session.transforms.names())
    session.add_transform(forward_name, transform)

    try:
        import nibabel as nib

        orientation = "".join(nib.orientations.aff2axcodes(np.asarray(image.affine, dtype=float)))
    except Exception:
        orientation = "unknown"

    voxel_size = np.linalg.norm(np.asarray(image.affine, dtype=float)[:3, :3], axis=0)
    voxel_size = np.where(voxel_size == 0.0, 1.0, voxel_size)

    info_msg = (
        f"Imported MRI image: {image_name}\n"
        f"  File: {path.name}\n"
        f"  World frame: {mri_frame.name}\n"
        f"  Voxel frame: {voxel_frame.name}\n"
        f"  Dimensions: {image.shape[0]} x {image.shape[1]} x {image.shape[2]}\n"
        f"  Voxel size: {voxel_size[0]:0.3f} x {voxel_size[1]:0.3f} x {voxel_size[2]:0.3f} mm\n"
        f"  Orientation: {orientation}\n"
        f"  Data type: {image.data.dtype}\n"
        f"  Intensity range: {float(np.min(image.data)):0.3f} -> {float(np.max(image.data)):0.3f}\n"
        f"  Transform: {forward_name}"
    )

    import nibabel as nib


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
    "import_surface",
    "import_multiple_transforms",
    "import_multiple_surfaces",
    "import_multiple_images",
    "preview_xfm_transform",
    "import_image",
    "get_file_extension",
    "ImportError",
    "UnsupportedFormatError",
]
