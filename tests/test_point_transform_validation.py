"""Tests that validate point transformations numerically against the raw affine.

These use the repository sample .fif transform as the oracle when available.
"""

from pathlib import Path

import numpy as np
import pytest

from core.point import Point
from core.session import Session

try:
    import mne  # type: ignore
except Exception:  # pragma: no cover - skip when mne not installed
    mne = None  # type: ignore


def _sample_fif_path() -> Path:
    return Path(__file__).resolve().parents[1] / "dataset" / "trans" / "sample_audvis_raw-trans.fif"


def test_point_transform_matches_affine_and_roundtrip():
    fif = _sample_fif_path()
    if not fif.exists():
        pytest.skip(f"Sample .fif not present: {fif}")
    if mne is None:
        pytest.skip("MNE-Python not installed")

    # Oracle matrix from MNE
    mne_trans = mne.read_trans(str(fif))
    oracle = np.asarray(mne_trans["trans"], dtype=float)

    # Import through the session path so frames are registered the same way
    session = Session()
    transform, info = session.import_transform(str(fif))

    # Create a point in the source frame and transform
    src = transform.source
    coords = np.array([10.0, 20.0, 30.0])
    p = Point(coords, src)
    out = transform.apply(p)

    expected = (oracle @ np.append(coords, 1.0))[:3]
    assert np.allclose(out.coords, expected, atol=1e-6)

    # Round-trip via invert
    inv = transform.invert()
    back = inv.apply(out)
    assert np.allclose(back.coords, coords, atol=1e-6)
