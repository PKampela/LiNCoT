"""MNE backend wrapper."""

from __future__ import annotations

from typing import Optional, cast

import numpy as np

from ..core.frames import CoordinateFrame
from ..core.transform import Transform


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


def head_to_mri_transform(path: str, head_frame: CoordinateFrame, mri_frame: CoordinateFrame) -> Transform:
    """Load a head -> MRI transform."""

    return load_mne_transform(path, source_frame=head_frame, target_frame=mri_frame)


def mri_to_mni_transform(path: str, mri_frame: CoordinateFrame, mni_frame: CoordinateFrame) -> Transform:
    """Load an MRI -> MNI transform."""

    return load_mne_transform(path, source_frame=mri_frame, target_frame=mni_frame)
