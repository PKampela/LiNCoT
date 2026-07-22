"""Generate a quantitative validation report comparing results to oracles.

This test computes numeric metrics and writes them to
`reports/validation_report.json` so you can include them in your project
report. Tests that require MNE or sample data are skipped when unavailable.
"""

from pathlib import Path
import json
import numpy as np
import pytest

from core.session import Session

try:
    import mne  # type: ignore
except Exception:  # pragma: no cover - skip when mne not installed
    mne = None  # type: ignore

from backends import mne_backend
from backends.nibabel_backend import load_nifti, _select_affine
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import save
from core.point import Point
from core.transform import Transform
from core.frames import CoordinateFrame


REPORT_PATH = Path("reports") / "validation_report.json"


def _sample_fif_path() -> Path:
    return Path(__file__).resolve().parents[1] / "dataset" / "trans" / "sample_audvis_raw-trans.fif"


def _ensure_reports_dir():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def test_generate_quantitative_validation_report(tmp_path: Path):
    """Compute metrics and save a JSON report for inclusion in documentation."""
    report = {
        "mne_available": mne is not None,
        "fif_comparison": None,
        "point_errors": None,
        "roundtrip_error": None,
        "nifti_affine_comparison": None,
    }

    fif = _sample_fif_path()
    if not fif.exists() or mne is None:
        pytest.skip("MNE or sample .fif not available; skipping numeric parity report")

    # --- .fif matrix comparison ---
    mne_trans = mne.read_trans(str(fif))
    mne_matrix = np.asarray(mne_trans["trans"], dtype=float)

    session = Session()
    our_transform, src_name, tgt_name = mne_backend.load_transform_with_frame_mapping(str(fif), session.frames)
    our_matrix = our_transform.matrix

    diff = np.abs(our_matrix - mne_matrix)
    fif_metrics = {
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
        "rms_diff": float(np.sqrt(np.mean(diff ** 2))),
    }
    report["fif_comparison"] = fif_metrics

    # --- frame mapping validation ---
    try:
        from_id = int(mne_trans.get("from"))
        to_id = int(mne_trans.get("to"))
        expected_from = mne_backend._get_frame_name_from_mne_id(from_id)
        expected_to = mne_backend._get_frame_name_from_mne_id(to_id)
    except Exception:
        expected_from = None
        expected_to = None

    report["frame_mapping"] = {
        "mne_from_id": int(mne_trans.get("from")),
        "mne_to_id": int(mne_trans.get("to")),
        "expected_from_name": expected_from,
        "expected_to_name": expected_to,
        "actual_from_name": src_name,
        "actual_to_name": tgt_name,
    }

    # --- point transform error (grid of points) ---
    rng = np.random.default_rng(12345)
    points = rng.uniform(-50.0, 50.0, size=(25, 3))
    diffs = []
    for coords in points:
        expected = (mne_matrix @ np.append(coords, 1.0))[:3]
        p = Point(coords, our_transform.source)
        out = our_transform.apply(p)
        diffs.append(np.abs(out.coords - expected))
    diffs = np.vstack(diffs)
    pt_max = float(np.max(diffs))
    pt_mean = float(np.mean(diffs))
    pt_rms = float(np.sqrt(np.mean(diffs ** 2)))
    point_metrics = {"max_abs_error": pt_max, "mean_abs_error": pt_mean, "rms_error": pt_rms}
    report["point_errors"] = point_metrics

    # --- round-trip error ---
    inv = our_transform.invert()
    back = inv.apply(out)
    rt_error = np.abs(back.coords - coords)
    rt_metrics = {
        "max_abs_error": float(np.max(rt_error)),
        "mean_abs_error": float(np.mean(rt_error)),
        "rms_error": float(np.sqrt(np.mean(rt_error ** 2))),
    }
    report["roundtrip_error"] = rt_metrics

    # --- chained transform parity ---
    # Create a small synthetic secondary transform (translation) and compose
    trans2_matrix = np.eye(4)
    trans2_matrix[:3, 3] = [2.0, -3.0, 5.0]
    # trans2: from our_transform.target -> "tmp_frame"
    tmp_frame = CoordinateFrame("tmp_frame", ("R", "A", "S"), "mm")
    trans2 = Transform(source=our_transform.target, target=tmp_frame, matrix=trans2_matrix)

    # Sequential application vs matrix multiplication
    seq_diffs = []
    for coords in points:
        p = Point(coords, our_transform.source)
        first = our_transform.apply(p)
        seq = trans2.apply(first)

        composed_matrix = trans2.matrix @ our_transform.matrix
        expected = (composed_matrix @ np.append(coords, 1.0))[:3]
        seq_diffs.append(np.abs(seq.coords - expected))
    seq_diffs = np.vstack(seq_diffs)
    chain_metrics = {
        "max_abs_diff": float(np.max(seq_diffs)),
        "mean_abs_diff": float(np.mean(seq_diffs)),
        "rms_diff": float(np.sqrt(np.mean(seq_diffs ** 2))),
    }
    report["chain_parity"] = chain_metrics

    # --- NIfTI affine and metadata comparison ---
    # Create a synthetic NIfTI and compare backend's affine with nibabel's affine
    shape = (8, 8, 8)
    data = np.zeros(shape, dtype=np.float32)
    data[3, 3, 3] = 1.0
    affine = np.eye(4, dtype=float)
    affine[:3, 3] = [5.0, -2.0, 1.0]
    img = Nifti1Image(data, affine)
    # set explicit qform/sform codes to exercise precedence behavior
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    p = tmp_path / "rep_test.nii"
    save(img, p)

    info = load_nifti(str(p))
    nib_img = nib.load(str(p))
    header = nib_img.header
    nib_affine = np.asarray(nib_img.affine, dtype=float)
    backend_affine = info.affine
    a_diff = np.abs(backend_affine - nib_affine)
    # header fields useful for validation
    nifti_header_fields = {
        "sform_code": int(header.get("sform_code", 0)),
        "qform_code": int(header.get("qform_code", 0)),
        "pixdim": tuple(header.get_zooms()),
        "xyzt_units": header.get_xyzt_units(),
        "datatype": str(header.get_data_dtype()),
    }
    nifti_metrics = {
        "max_abs_affine_diff": float(np.max(a_diff)),
        "mean_abs_affine_diff": float(np.mean(a_diff)),
        "rms_affine_diff": float(np.sqrt(np.mean(a_diff ** 2))),
        "shape": info.shape,
        "path": str(p),
    }
    nifti_metrics["header"] = nifti_header_fields
    report["nifti_affine_comparison"] = nifti_metrics

    # Write JSON report
    _ensure_reports_dir()
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_default)

    # Also assert that values are within reasonable tolerances so the test fails visibly
    assert fif_metrics["max_abs_diff"] < 1e-6
    assert point_metrics["rms_error"] < 1e-6
    assert rt_metrics["rms_error"] < 1e-6
    assert nifti_metrics["max_abs_affine_diff"] < 1e-8
