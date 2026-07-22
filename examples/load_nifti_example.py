#!/usr/bin/env python
"""Example script demonstrating how to load a NIfTI volume with nibabel and TMSLabs.

This example shows:
1. Loading a NIfTI file using nibabel
2. Creating coordinate frames for voxel and world spaces
3. Creating a Transform object from the NIfTI affine matrix
4. Visualizing points in voxel space (pre-transform) and world space (post-transform)

Visualization uses matplotlib only (required):
    pip install matplotlib
"""

import sys
from typing import cast

import numpy as np
import matplotlib.pyplot as plt
from nibabel import loadsave
from nibabel.nifti1 import Nifti1Image

# Import TMSLabs modules
from core.frames import CoordinateFrame
from core.transform import Transform
from core.point import Point


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
        print("MNE-Python is not installed.")
        print("To use MNE sample data: pip install mne")
        return None


def load_and_inspect_nifti(nifti_path: str):
    """Load a NIfTI volume and create voxel-to-world transform."""
    
    print(f"\nLoading NIfTI file: {nifti_path}")
    
    # Load the NIfTI image using nibabel
    img = cast(Nifti1Image, loadsave.load(nifti_path))
    data = np.asarray(img.dataobj)
    affine = np.asarray(img.affine, dtype=float)
    
    print(f"\nImage properties:")
    print(f"  Shape: {img.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Data range: [{data.min():.1f}, {data.max():.1f}]")
    print(f"\nAffine matrix (voxel to world):")
    print(affine)
    
    # Define coordinate frames
    voxel_frame = CoordinateFrame(
        name="T1_voxel",
        axes=("i", "j", "k"),
        units="voxel",
        description="Voxel coordinates"
    )
    
    scanner_frame = CoordinateFrame(
        name="scanner",
        axes=("R", "A", "S"),
        units="mm",
        description="Scanner world coordinates (RAS)"
    )
    
    # Create the voxel-to-world transform using the NIfTI affine
    voxel_to_world = Transform(
        source=voxel_frame,
        target=scanner_frame,
        matrix=affine
    )
    
    print(f"\nCreated transform: {voxel_to_world.source.name} -> {voxel_to_world.target.name}")
    
    # Example: Transform a voxel coordinate to world coordinates
    # Let's take the center voxel
    center_voxel = np.array([s // 2 for s in img.shape[:3]], dtype=float)
    center_point_voxel = Point(center_voxel, voxel_frame)
    center_point_world = voxel_to_world.apply(center_point_voxel)
    
    print(f"\nExample transformation:")
    print(f"  Center voxel: {center_voxel} ({voxel_frame.name})")
    print(f"  World coords: {center_point_world.coords} {scanner_frame.units} ({scanner_frame.name})")
    
    return img, voxel_to_world, voxel_frame, scanner_frame


def visualize_slice_with_points(
    img: Nifti1Image,
    voxel_frame: CoordinateFrame,
    scanner_frame: CoordinateFrame,
    voxel_to_world: Transform,
) -> None:
    """Display native and transformed slices with separate forward and inverse views."""

    print("\nDisplaying native -> MNI and MNI -> native views...")
    try:
        from nibabel.processing import resample_to_output

        data = np.asarray(img.dataobj)
        shape = np.array(img.shape[:3], dtype=float)
        world_img = resample_to_output(img, voxel_sizes=(1.0, 1.0, 1.0))
        world_data = np.asarray(world_img.dataobj)
        world_affine = np.asarray(world_img.affine, dtype=float)
        world_shape = np.array(world_img.shape[:3], dtype=int)

        # Define a few sample points and their transformed coordinates.
        voxel_points_dict = {
            "center": shape / 2,
            "corner_1": np.array([(shape[0] / 2), 100, 100], dtype=float),
            "corner_2": np.array([100, 100, (shape[0] / 2)], dtype=float),
        }

        world_points_dict = {}
        inverse_transform = voxel_to_world.invert()  # world -> voxel

        for name, voxel_coord in voxel_points_dict.items():
            point_voxel = Point(voxel_coord, voxel_frame)
            point_world = voxel_to_world.apply(point_voxel)
            world_points_dict[name] = point_world.coords
            print(f"  {name:10s}: native {voxel_coord} -> MNI {point_world.coords}")

        # Create figure with one row per direction.
        fig, axes = plt.subplots(3, 3, figsize=(16, 13), constrained_layout=True)
        fig.suptitle(
            "Native to MNI and back: separate forward and inverse views",
            fontsize=14,
            fontweight="bold",
        )

        center_idx = (shape / 2).astype(int)

        # Color map for points.
        colors = {"center": "red", "corner_1": "blue", "corner_2": "green"}
        markers = {"center": "o", "corner_1": "s", "corner_2": "^"}

        def _style_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
            axis.set_title(title)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)

        def _style_world_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
            axis.set_title(title)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2, linestyle="--", linewidth=0.7)
            axis.set_facecolor("#fbfbfb")

        def _slice_extent(plane: str, slice_index: int) -> tuple[float, float, float, float]:
            # For transposed slices, extent is (left, right, bottom, top) in the transposed coordinate system
            if plane == "axial":
                # Slice shape after [:, :, z].T is (A, R), so extent y is first (A), x is second (R)
                corners = [
                    (world_affine @ np.array([0, 0, slice_index, 1.0]))[:3],
                    (world_affine @ np.array([world_shape[0] - 1, 0, slice_index, 1.0]))[:3],
                    (world_affine @ np.array([0, world_shape[1] - 1, slice_index, 1.0]))[:3],
                    (world_affine @ np.array([world_shape[0] - 1, world_shape[1] - 1, slice_index, 1.0]))[:3],
                ]
                x_coords = [c[0] for c in corners]
                y_coords = [c[1] for c in corners]
                return (min(x_coords), max(x_coords), min(y_coords), max(y_coords))
            if plane == "sagittal":
                # Slice shape after [x, :, :].T is (S, A), so extent y is first (S), x is second (A)
                corners = [
                    (world_affine @ np.array([slice_index, 0, 0, 1.0]))[:3],
                    (world_affine @ np.array([slice_index, world_shape[1] - 1, 0, 1.0]))[:3],
                    (world_affine @ np.array([slice_index, 0, world_shape[2] - 1, 1.0]))[:3],
                    (world_affine @ np.array([slice_index, world_shape[1] - 1, world_shape[2] - 1, 1.0]))[:3],
                ]
                y_coords = [c[1] for c in corners]  # A coordinate
                z_coords = [c[2] for c in corners]  # S coordinate
                return (min(y_coords), max(y_coords), min(z_coords), max(z_coords))
            # Coronal: slice shape after [:, y, :].T is (S, R)
            corners = [
                (world_affine @ np.array([0, slice_index, 0, 1.0]))[:3],
                (world_affine @ np.array([world_shape[0] - 1, slice_index, 0, 1.0]))[:3],
                (world_affine @ np.array([0, slice_index, world_shape[2] - 1, 1.0]))[:3],
                (world_affine @ np.array([world_shape[0] - 1, slice_index, world_shape[2] - 1, 1.0]))[:3],
            ]
            x_coords = [c[0] for c in corners]
            z_coords = [c[2] for c in corners]
            return (min(x_coords), max(x_coords), min(z_coords), max(z_coords))

        def _draw_native_points(axis, plane: str, slice_index: int) -> None:
            for name, voxel_coord in voxel_points_dict.items():
                if plane == "axial" and abs(voxel_coord[2] - slice_index) < 1:
                    axis.plot(
                        voxel_coord[0],
                        voxel_coord[1],
                        marker=markers[name],
                        color=colors[name],
                        markersize=10,
                        markeredgewidth=1.5,
                        markeredgecolor="white",
                        linestyle="none",
                    )
                    axis.text(
                        voxel_coord[0] + 6,
                        voxel_coord[1],
                        name,
                        color=colors[name],
                        fontsize=9,
                        fontweight="bold",
                    )
                elif plane == "sagittal" and abs(voxel_coord[0] - slice_index) < 1:
                    axis.plot(
                        voxel_coord[1],
                        voxel_coord[2],
                        marker=markers[name],
                        color=colors[name],
                        markersize=10,
                        markeredgewidth=1.5,
                        markeredgecolor="white",
                        linestyle="none",
                    )
                    axis.text(
                        voxel_coord[1] + 6,
                        voxel_coord[2],
                        name,
                        color=colors[name],
                        fontsize=9,
                        fontweight="bold",
                    )
                elif plane == "coronal" and abs(voxel_coord[1] - slice_index) < 1:
                    axis.plot(
                        voxel_coord[0],
                        voxel_coord[2],
                        marker=markers[name],
                        color=colors[name],
                        markersize=10,
                        markeredgewidth=1.5,
                        markeredgecolor="white",
                        linestyle="none",
                    )
                    axis.text(
                        voxel_coord[0] + 6,
                        voxel_coord[2],
                        name,
                        color=colors[name],
                        fontsize=9,
                        fontweight="bold",
                    )

        def _draw_inverse_points(axis, plane: str, slice_index: int) -> None:
            for name, world_coord in world_points_dict.items():
                point_world = Point(world_coord, scanner_frame)
                voxel_back = inverse_transform.apply(point_world).coords
                if plane == "axial" and abs(voxel_back[2] - slice_index) < 1:
                    axis.plot(
                        voxel_back[0],
                        voxel_back[1],
                        marker=markers[name],
                        color=colors[name],
                        markersize=10,
                        markeredgewidth=1.5,
                        markeredgecolor=colors[name],
                        markerfacecolor="none",
                        linestyle="none",
                    )
                    axis.text(
                        voxel_back[0] + 6,
                        voxel_back[1],
                        f"{name}'",
                        color=colors[name],
                        fontsize=9,
                        fontweight="bold",
                        style="italic",
                    )
                elif plane == "sagittal" and abs(voxel_back[0] - slice_index) < 1:
                    axis.plot(
                        voxel_back[1],
                        voxel_back[2],
                        marker=markers[name],
                        color=colors[name],
                        markersize=10,
                        markeredgewidth=1.5,
                        markeredgecolor=colors[name],
                        markerfacecolor="none",
                        linestyle="none",
                    )
                    axis.text(
                        voxel_back[1] + 6,
                        voxel_back[2],
                        f"{name}'",
                        color=colors[name],
                        fontsize=9,
                        fontweight="bold",
                        style="italic",
                    )
                elif plane == "coronal" and abs(voxel_back[1] - slice_index) < 1:
                    axis.plot(
                        voxel_back[0],
                        voxel_back[2],
                        marker=markers[name],
                        color=colors[name],
                        markersize=10,
                        markeredgewidth=1.5,
                        markeredgecolor=colors[name],
                        markerfacecolor="none",
                        linestyle="none",
                    )
                    axis.text(
                        voxel_back[0] + 6,
                        voxel_back[2],
                        f"{name}'",
                        color=colors[name],
                        fontsize=9,
                        fontweight="bold",
                        style="italic",
                    )

        def _draw_world_points(axis, plane: str, slice_index: float) -> None:
            for name, world_coord in world_points_dict.items():
                # Only plot points that intersect this slice plane (within 1mm)
                if plane == "axial" and abs(world_coord[2] - slice_index) < 1:
                    x_value, y_value = world_coord[0], world_coord[1]
                    axis.scatter(
                        x_value,
                        y_value,
                        marker=markers[name],
                        s=70,
                        color=colors[name],
                        edgecolors="white",
                        linewidths=1.0,
                        zorder=3,
                    )
                    axis.text(
                        x_value + 1.5,
                        y_value,
                        name,
                        color=colors[name],
                        fontsize=9,
                        fontweight="bold",
                    )
                elif plane == "sagittal" and abs(world_coord[0] - slice_index) < 1:
                    x_value, y_value = world_coord[1], world_coord[2]
                    axis.scatter(
                        x_value,
                        y_value,
                        marker=markers[name],
                        s=70,
                        color=colors[name],
                        edgecolors="white",
                        linewidths=1.0,
                        zorder=3,
                    )
                    axis.text(
                        x_value + 1.5,
                        y_value,
                        name,
                        color=colors[name],
                        fontsize=9,
                        fontweight="bold",
                    )
                elif plane == "coronal" and abs(world_coord[1] - slice_index) < 1:
                    x_value, y_value = world_coord[0], world_coord[2]
                    axis.scatter(
                        x_value,
                        y_value,
                        marker=markers[name],
                        s=70,
                        color=colors[name],
                        edgecolors="white",
                        linewidths=1.0,
                        zorder=3,
                    )
                    axis.text(
                        x_value + 1.5,
                        y_value,
                        name,
                        color=colors[name],
                        fontsize=9,
                        fontweight="bold",
                    )

            axis.axvline(0, color="0.7", linewidth=1, linestyle=":")
            axis.axhline(0, color="0.7", linewidth=1, linestyle=":")
            axis.text(
                0.02,
                0.96,
                f"slice @ {slice_index:.1f} mm",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="0.35",
            )

        # ===== ROW 1: Native space =====
        z_idx = center_idx[2]
        axial_slice = data[:, :, z_idx]
        axes[0, 0].imshow(axial_slice, cmap="gray")
        _style_axis(axes[0, 0], f"Native axial (Z={z_idx})", "X (voxels)", "Y (voxels)")
        _draw_native_points(axes[0, 0], "axial", z_idx)

        x_idx = center_idx[0]
        sagittal_slice = data[x_idx, :, :]
        axes[0, 1].imshow(sagittal_slice, cmap="gray")
        _style_axis(axes[0, 1], f"Native sagittal (X={x_idx})", "Y (voxels)", "Z (voxels)")
        _draw_native_points(axes[0, 1], "sagittal", x_idx)

        y_idx = center_idx[1]
        coronal_slice = data[:, y_idx, :]
        axes[0, 2].imshow(coronal_slice, cmap="gray")
        _style_axis(axes[0, 2], f"Native coronal (Y={y_idx})", "X (voxels)", "Z (voxels)")
        _draw_native_points(axes[0, 2], "coronal", y_idx)

        # ===== ROW 2: Native -> MNI =====
        center_world = voxel_to_world.apply(Point(center_idx.astype(float), voxel_frame)).coords

        world_z_idx = world_shape[2] // 2
        world_x_idx = world_shape[0] // 2
        world_y_idx = world_shape[1] // 2

        world_coronal_slice = world_data[:, world_y_idx, :]
        axes[1, 0].imshow(world_coronal_slice.T, cmap="gray", origin="lower", extent=_slice_extent("coronal", world_y_idx))
        _style_world_axis(axes[1, 0], f"Native -> MNI coronal (A={center_world[1]:.1f} mm)", "R (mm)", "S (mm)")
        _draw_world_points(axes[1, 0], "coronal", center_world[1])

        world_sagittal_slice = world_data[world_x_idx, :, :]
        axes[1, 1].imshow(world_sagittal_slice.T, cmap="gray", origin="lower", extent=_slice_extent("sagittal", world_x_idx))
        _style_world_axis(axes[1, 1], f"Native -> MNI sagittal (R={center_world[0]:.1f} mm)", "A (mm)", "S (mm)")
        _draw_world_points(axes[1, 1], "sagittal", center_world[0])

        world_axial_slice = world_data[:, :, world_z_idx]
        axes[1, 2].imshow(world_axial_slice.T, cmap="gray", origin="lower", extent=_slice_extent("axial", world_z_idx))
        _style_world_axis(axes[1, 2], f"Native -> MNI axial (S={center_world[2]:.1f} mm)", "R (mm)", "A (mm)")
        _draw_world_points(axes[1, 2], "axial", center_world[2])

        # ===== ROW 3: MNI -> Native =====
        axes[2, 0].imshow(axial_slice, cmap="gray", alpha=0.55)
        _style_axis(axes[2, 0], f"MNI -> Native axial (Z={z_idx})", "X (voxels)", "Y (voxels)")
        _draw_inverse_points(axes[2, 0], "axial", z_idx)

        axes[2, 1].imshow(sagittal_slice, cmap="gray", alpha=0.55)
        _style_axis(axes[2, 1], f"MNI -> Native sagittal (X={x_idx})", "Y (voxels)", "Z (voxels)")
        _draw_inverse_points(axes[2, 1], "sagittal", x_idx)

        axes[2, 2].imshow(coronal_slice, cmap="gray", alpha=0.55)
        _style_axis(axes[2, 2], f"MNI -> Native coronal (Y={y_idx})", "X (voxels)", "Z (voxels)")
        _draw_inverse_points(axes[2, 2], "coronal", y_idx)

        # Add legend.
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=8, label="center (native)", markeredgecolor="white"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor="blue", markersize=8, label="corner_1 (native)", markeredgecolor="white"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="green", markersize=8, label="corner_2 (native)", markeredgecolor="white"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markersize=8, label="inverse(MNI) -> native", markeredgewidth=1.5, markeredgecolor="gray"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9, frameon=True)

        plt.show()
    except Exception as e:
        print(f"  Error during visualization: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("Loading NIfTI Volume Example")
    print("=" * 70)
    
    # Try to get the sample T1 path from MNE
    t1_path = get_sample_t1_path()
    
    if t1_path:
        print(f"\nFound sample T1 at: {t1_path}")
        img, transform, voxel_frame, scanner_frame = load_and_inspect_nifti(t1_path)
        
        # Perform visualization
        print("\n" + "=" * 70)
        print("Visualization: Pre-Transform (Voxel) and Post-Transform (World)")
        print("=" * 70)
        
        # Show 2D slices with pre/post transform points
        visualize_slice_with_points(img, voxel_frame, scanner_frame, transform)
        
        print("\n" + "=" * 70)
        print("Success! Loaded and visualized MNE sample T1 MRI")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("Note: MNE sample dataset not available")
        print("=" * 70)
        print("\nTo use this example with MNE sample data:")
        print("  1. Install MNE-Python: pip install mne")
        print("  2. Download sample data: python -c 'import mne; mne.datasets.sample.data_path()'")
        print("\nAlternatively, you can use any NIfTI file (.nii or .nii.gz):")
        print("  - Modify the script to point to your NIfTI file path")
        print("  - Run: python examples/load_nifti_example.py")

