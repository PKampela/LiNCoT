"""Parity tests that validate MNE transform imports against MNE's own reader.

These tests are skipped when the MNE sample data or `mne` package
are not available, to keep CI and developer runs robust.
"""

from pathlib import Path

import numpy as np
import pytest

from core.session import Session

try:
    import mne  # type: ignore
except Exception:  # pragma: no cover - skip when mne not installed
    mne = None  # type: ignore

from backends import mne_backend


def _sample_fif_path() -> Path:
    return Path(__file__).resolve().parents[1] / "dataset" / "trans" / "sample_audvis_raw-trans.fif"


def test_mne_transform_matrix_matches_backend():
    fif = _sample_fif_path()
    if not fif.exists():
        pytest.skip(f"Sample .fif not present: {fif}")
    if mne is None:
        pytest.skip("MNE-Python not installed")

    # Read matrix via MNE (oracle)
    mne_trans = mne.read_trans(str(fif))
    mne_matrix = np.asarray(mne_trans["trans"], dtype=float)

    # Load via our backend adapter (should produce same matrix)
    session = Session()
    transform, source_name, target_name = mne_backend.load_transform_with_frame_mapping(str(fif), session.frames)

    assert transform.matrix.shape == (4, 4)
    assert np.allclose(transform.matrix, mne_matrix)

    # Confirm the frames were registered and named as reported
    assert source_name in session.frames.list_frames()
    assert target_name in session.frames.list_frames()
