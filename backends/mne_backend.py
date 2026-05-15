"""MNE backend wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, cast

import numpy as np

from core.frames import CoordinateFrame
from core.transform import Transform

if TYPE_CHECKING:
    from registry.frame_registry import FrameRegistry


# MNE FIFF coordinate frame constants
_MNE_FRAME_ID_TO_NAME = {
    1: "mri",          # FIFFV_COORD_MRI
    2: "head",         # FIFFV_COORD_HEAD
    3: "mni_head",     # FIFFV_COORD_MNI_HEAD
    4: "mni_tal",      # FIFFV_COORD_MNI_TAL
    5: "mni_mgh_tal",  # FIFFV_COORD_MNI_MGH_TAL
    10: "mri_voxel",   # FIFFV_COORD_MRI_VOXEL
    11: "surface_ras", # FIFFV_COORD_SURFACE_RAS
    12: "ctf",         # FIFFV_COORD_CTF_HEAD
    13: "ctf_device",  # FIFFV_COORD_CTF_DEVICE
    14: "4d_head",     # FIFFV_COORD_4D_HEAD
    15: "4d_device",   # FIFFV_COORD_4D_DEVICE
}


def _get_mne_frame_id(mne_trans_dict: dict) -> tuple[int, int]:
    """Extract source and destination frame IDs from MNE transform dict.

    Args:
        mne_trans_dict: MNE transform dictionary (from mne.read_trans)

    Returns:
        Tuple of (from_frame_id, to_frame_id)
    """
    from_id = mne_trans_dict.get("from", None)
    to_id = mne_trans_dict.get("to", None)

    if from_id is None or to_id is None:
        raise ValueError(
            f"MNE transform missing frame IDs: from={from_id}, to={to_id}"
        )

    return int(from_id), int(to_id)


def _get_frame_name_from_mne_id(frame_id: int) -> str:
    """Map MNE frame ID to internal frame name.

    Args:
        frame_id: MNE FIFFV_COORD_* constant

    Returns:
        Internal frame name (e.g., "head", "mri")

    Raises:
        ValueError: If frame ID is unknown
    """
    if frame_id not in _MNE_FRAME_ID_TO_NAME:
        raise ValueError(
            f"Unknown MNE frame ID: {frame_id}. "
            f"Known IDs: {sorted(_MNE_FRAME_ID_TO_NAME.keys())}"
        )
    return _MNE_FRAME_ID_TO_NAME[frame_id]


def load_mne_transform(
    path: str,
    source_frame: Optional[CoordinateFrame] = None,
    target_frame: Optional[CoordinateFrame] = None,
) -> Transform:
    """Load an MNE transform file and convert it into a Transform.

    This function does not expose MNE objects to the caller.
    """

    import mne

    mne_trans = mne.read_trans(path)
    matrix = np.asarray(cast(dict, mne_trans)["trans"], dtype=float)

    if source_frame is None or target_frame is None:
        raise ValueError("source_frame and target_frame must be provided")

    return Transform(source=source_frame, target=target_frame, matrix=matrix)


def load_transform_with_frame_mapping(
    path: str,
    frame_registry: FrameRegistry,
) -> tuple[Transform, str, str]:
    """Load an MNE transform and auto-map frames based on MNE frame IDs.

    This is the recommended way to import MNE transforms, as it automatically
    detects and maps the MNE frame coordinate systems to internal CoordinateFrame
    objects based on their FIFF IDs.

    Args:
        path: Path to MNE transform (.fif) file
        frame_registry: FrameRegistry to look up frames by name

    Returns:
        Tuple of (Transform, source_frame_name, target_frame_name)

    Raises:
        ValueError: If frames are not registered or frame IDs are unknown
        FileNotFoundError: If file does not exist
    """
    import mne

    mne_trans = mne.read_trans(path)
    matrix = np.asarray(cast(dict, mne_trans)["trans"], dtype=float)

    # Get frame IDs from MNE transform
    from_id, to_id = _get_mne_frame_id(mne_trans)

    # Map to frame names
    from_name = _get_frame_name_from_mne_id(from_id)
    to_name = _get_frame_name_from_mne_id(to_id)

    # Look up or create frames
    try:
        source_frame = frame_registry.get_frame(from_name)
    except KeyError:
        # Create the frame if it doesn't exist
        # Using standard RAS (Right-Anterior-Superior) convention for neuro frames
        source_frame = CoordinateFrame(
            name=from_name,
            axes=("R", "A", "S"),
            units="mm",
            description=f"MNE frame: {from_name}",
        )
        frame_registry.register_frame(source_frame)

    try:
        target_frame = frame_registry.get_frame(to_name)
    except KeyError:
        # Create the frame if it doesn't exist
        target_frame = CoordinateFrame(
            name=to_name,
            axes=("R", "A", "S"),
            units="mm",
            description=f"MNE frame: {to_name}",
        )
        frame_registry.register_frame(target_frame)

    transform = Transform(source=source_frame, target=target_frame, matrix=matrix)

    return transform, from_name, to_name


def head_to_mri_transform(path: str, head_frame: CoordinateFrame, mri_frame: CoordinateFrame) -> Transform:
    """Load a head -> MRI transform."""

    return load_mne_transform(path, source_frame=head_frame, target_frame=mri_frame)


def mri_to_mni_transform(path: str, mri_frame: CoordinateFrame, mni_frame: CoordinateFrame) -> Transform:
    """Load an MRI -> MNI transform."""

    return load_mne_transform(path, source_frame=mri_frame, target_frame=mni_frame)
