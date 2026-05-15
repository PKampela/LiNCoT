"""
Download and organize MNE sample dataset for TMS workflows.

This script:
1. Downloads the MNE sample dataset
2. Organizes files into a structured "dataset" folder
3. Converts MRI volume from .mgz to .nii.gz
4. Copies cortical surfaces, scalp, skull surfaces, and registration files
"""

import os
import shutil
from pathlib import Path

import mne
import nibabel as nib
from nibabel.freesurfer import load as load_freesurfer
from nibabel.nifti1 import Nifti1Image

# Step 1: Download MNE sample dataset
print("Downloading MNE sample dataset...")
sample_data_path = mne.datasets.sample.data_path(verbose=True)
print(f"\nSample data downloaded to: {sample_data_path}\n")

# Step 2: Create dataset folder structure
base_dir = Path(__file__).parent / "dataset"
base_dir.mkdir(exist_ok=True)

# Create subdirectories
dirs = {
    "mri": base_dir / "mri",
    "surfaces": base_dir / "surfaces",
    "bem": base_dir / "bem",
    "trans": base_dir / "trans",
}

for dir_path in dirs.values():
    dir_path.mkdir(exist_ok=True)

print("Created dataset folder structure:")
for key, path in dirs.items():
    print(f"  - {path.relative_to(base_dir)}/")

# Step 3: Convert and copy MRI volume
print("\n" + "="*70)
print("Processing MRI Volume (T1.mgz -> T1.nii.gz)")
print("="*70)

mgz_path = Path(sample_data_path) / "subjects" / "sample" / "mri" / "T1.mgz"
nifti_output = dirs["mri"] / "T1.nii.gz"

if mgz_path.exists():
    print(f"Loading: {mgz_path}")
    # Load MGZ file and convert to NIfTI
    img = nib.load(str(mgz_path))
    # Save as NIfTI
    nib.save(img, str(nifti_output))
    print(f"✓ Saved as: {nifti_output}")
    print(f"  Shape: {img.shape}")
    print(f"  Data type: {img.get_data_dtype()}")
else:
    print(f"✗ MGZ file not found: {mgz_path}")

# Step 4: Copy cortical surfaces
print("\n" + "="*70)
print("Processing Cortical Surfaces (hemispheres)")
print("="*70)

surf_dir = Path(sample_data_path) / "subjects" / "sample" / "surf"
cortical_surfaces = ["lh.pial", "rh.pial", "lh.white", "rh.white", "lh.sphere", "rh.sphere"]

for surface_name in cortical_surfaces:
    surf_path = surf_dir / surface_name
    if surf_path.exists():
        dest_path = dirs["surfaces"] / surface_name
        shutil.copy2(str(surf_path), str(dest_path))
        print(f"✓ {surface_name}")
    else:
        print(f"✗ {surface_name} not found")

# Step 5: Copy BEM surfaces (scalp and skull)
print("\n" + "="*70)
print("Processing BEM Surfaces (scalp & skull)")
print("="*70)

bem_dir = Path(sample_data_path) / "subjects" / "sample" / "bem"
bem_surfaces = [
    "outer_skin.surf",
    "outer_skull.surf",
    "inner_skull.surf",
]

for bem_name in bem_surfaces:
    bem_path = bem_dir / bem_name
    if bem_path.exists():
        dest_path = dirs["bem"] / bem_name
        shutil.copy2(str(bem_path), str(dest_path))
        print(f"✓ {bem_name}")
    else:
        print(f"✗ {bem_name} not found")

# Also try to copy -bem.fif files (MNE BEM solutions)
bem_fif_files = list(bem_dir.glob("*.fif"))
for bem_fif_path in bem_fif_files:
    if "bem" in bem_fif_path.name:
        dest_path = dirs["bem"] / bem_fif_path.name
        shutil.copy2(str(bem_fif_path), str(dest_path))
        print(f"✓ {bem_fif_path.name}")

# Step 6: Copy registration/transformation files
print("\n" + "="*70)
print("Processing Registration Files (*-trans.fif)")
print("="*70)

mri_dir = Path(sample_data_path) / "subjects" / "sample" / "mri"
trans_files = list(mri_dir.glob("*-trans.fif"))

if not trans_files:
    # Sometimes in MEG folder
    meg_dir = Path(sample_data_path) / "MEG" / "sample"
    trans_files = list(meg_dir.glob("*-trans.fif"))

for trans_path in trans_files:
    dest_path = dirs["trans"] / trans_path.name
    shutil.copy2(str(trans_path), str(dest_path))
    print(f"✓ {trans_path.name}")

# Step 7: Summary
print("\n" + "="*70)
print("DOWNLOAD AND ORGANIZATION COMPLETE")
print("="*70)

print(f"\nDataset location: {base_dir}")
print("\nFolder structure:")
for key, dir_path in dirs.items():
    file_count = len(list(dir_path.glob("*")))
    print(f"  - {dir_path.name}/ ({file_count} files)")

print("\nFiles organized by type:")
print("  MRI Volume:")
print("    - T1.nii.gz (Anatomical MRI)")
print("\n  Cortical Surfaces:")
print("    - lh.pial, rh.pial (Hemisphere cortex surfaces)")
print("    - lh.white, rh.white (White matter surfaces)")
print("    - lh.sphere, rh.sphere (Spherical surfaces)")
print("\n  BEM Surfaces:")
print("    - outer_skin.surf (Scalp surface)")
print("    - outer_skull.surf, inner_skull.surf (Skull surfaces)")
print("\n  Registration Files:")
print("    - *-trans.fif (Head ↔ MRI transformations)")

print("\nReady for TMS workflows!")
