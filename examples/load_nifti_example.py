#!/usr/bin/env python
"""Example script demonstrating how to load a NIfTI volume via CLI or programmatically.

This example uses the MNE-Python sample dataset, which includes a T1-weighted 
anatomical MRI volume.
"""

from pathlib import Path
import numpy as np

# Example 1: Using the CLI (run in terminal)
# First, get the sample data path (this downloads the dataset if needed)
# python -c "import mne; print(mne.datasets.sample.data_path() / 'subjects' / 'sample' / 'mri' / 'T1.mgz')"
# Then use the CLI:
# tmscoords load-volume <path_to_T1.mgz> --frame scanner --register-transform

# Example 2: Programmatic approach using the session and nibabel backend
from ..core.session import Session
from ..core.frames import CoordinateFrame
from ..backends.nibabel_backend import load_nifti_image, voxel_to_world_transform, load_nifti


def get_sample_t1_path():
    """Get path to the MNE sample dataset T1 MRI volume."""
    try:
        import mne
        data_path = mne.datasets.sample.data_path()
        t1_path = data_path / 'subjects' / 'sample' / 'mri' / 'T1.mgz'
        
        if not t1_path.exists():
            print(f"Sample T1 not found at {t1_path}")
            print("The MNE sample dataset may need to be downloaded.")
            print("Run: mne.datasets.sample.data_path() to download it.")
            return None
        
        return str(t1_path)
    except ImportError:
        print("MNE-Python is not installed. Install it with: pip install mne")
        return None


def load_volume_programmatically(nifti_path: str):
    """Load a NIfTI volume programmatically into a session."""
    
    # Create a session
    session = Session(subject_id="sample", description="MNE sample dataset TMS session")
    
    # Define the coordinate frame (e.g., scanner coordinates)
    scanner_frame = CoordinateFrame(name="scanner", units="mm", axes=("x", "y", "z"))
    session.add_frame(scanner_frame)
    
    # Load the NIfTI image
    print(f"Loading NIfTI file: {nifti_path}")
    image = load_nifti_image(nifti_path, scanner_frame)
    
    # Add the image to the session
    volume_name = "T1"
    session.add_image(volume_name, image)
    print(f"\nAdded image '{volume_name}' to session")
    print(f"  Shape: {image.shape}")
    print(f"  Frame: {image.frame.name}")
    print(f"  Data type: {image.data.dtype}")
    print(f"  Data range: [{image.data.min():.1f}, {image.data.max():.1f}]")
    print(f"  Affine shape: {image.affine.shape}")
    
    # Create voxel-to-world transform
    info = load_nifti(nifti_path)
    voxel_frame = CoordinateFrame(name="T1_voxel", units="voxel", axes=("i", "j", "k"))
    session.add_frame(voxel_frame)
    
    transform = voxel_to_world_transform(info, voxel_frame, scanner_frame)
    transform_name = "T1_voxel_to_scanner"
    session.add_transform(transform_name, transform)
    print(f"\nRegistered transform '{transform_name}'")
    print(f"  Source: {transform.source.name}")
    print(f"  Target: {transform.target.name}")
    print(f"  Matrix shape: {transform.matrix.shape}")
    
    return session


if __name__ == "__main__":
    print("=" * 70)
    print("Loading NIfTI Volume Example - MNE Sample Dataset")
    print("=" * 70)
    
    # Get the sample T1 path
    t1_path = get_sample_t1_path()
    
    if t1_path:
        print(f"\nFound sample T1 at: {t1_path}")
        print("\n" + "=" * 70)
        print("METHOD 1: CLI Usage")
        print("=" * 70)
        print(f"tmscoords load-volume {t1_path} --frame scanner --register-transform")
        
        print("\n" + "=" * 70)
        print("METHOD 2: Programmatic Usage")
        print("=" * 70)
        session = load_volume_programmatically(t1_path)
        
        print("\n" + "=" * 70)
        print("Session Summary")
        print("=" * 70)
        print(f"Subject ID: {session.subject_id}")
        print(f"Images: {list(session.images.keys())}")
        print(f"Frames: {session.frames.list_frames()}")
        print(f"Transforms: {session.transforms.list_transforms()}")
    else:
        print("\n" + "=" * 70)
        print("Note: MNE sample dataset not available")
        print("=" * 70)
        print("To use this example, install MNE-Python and download the sample dataset:")
        print("  pip install mne")
        print("  python -c 'import mne; mne.datasets.sample.data_path()'")
        print("\nAlternatively, use any NIfTI file:")
        print("  tmscoords load-volume /path/to/your/image.nii.gz --frame scanner --register-transform")
