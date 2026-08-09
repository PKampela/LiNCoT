from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.optimize import OptimizeResult, minimize
from scipy.ndimage import zoom

from dataclasses import dataclass

from .frames import CoordinateFrame
from .image import Image
from .sampling import sample_registration_points, world_to_voxel, voxel_to_world
from .transform import Transform


@dataclass(frozen=True)
class RegistrationSettings:

    pyramid_sizes: tuple[int, ...]
    sample_steps: tuple[int, ...] #Currently there is a bug where only the first sample step is used, so this should be a single value for now. Fix later to allow for different sample steps at different pyramid levels.

    translation: bool = True
    rigid: bool = True
    affine: bool = True

    translation_maxiter: int = 100
    rigid_maxiter: int = 100
    affine_maxiter: int = 100

    xtol: float = 1e-2
    ftol: float = 1e-3


@dataclass(frozen=True)
class RegistrationReport:
    quality: str
    iterations: int
    similarity: float
    translation_mm: float
    rotation_deg: float


FAST = RegistrationSettings(
    pyramid_sizes=(32, 64),
    sample_steps=(4, 2),
    translation=True,
    rigid=True,
    affine=False,

    translation_maxiter = 30,
    rigid_maxiter = 30,
    affine_maxiter = 40,
)

STANDARD = RegistrationSettings(
    pyramid_sizes=(32, 64, 128),
    sample_steps=(4, 2, 1),
    translation=True,
    rigid=True,
    affine=True,

    translation_maxiter = 50,
    rigid_maxiter = 50,
    affine_maxiter = 60,
)

ACCURATE = RegistrationSettings(
    pyramid_sizes=(32, 64, 128, 128),
    sample_steps=(4, 2, 1, 1),
    translation=True,
    rigid=True,
    affine=True,

    translation_maxiter = 100,
    rigid_maxiter = 100,
    affine_maxiter = 100,
)

QUALITY_PRESETS: dict[str, RegistrationSettings] = {
    "fast": FAST,
    "standard": STANDARD,
    "accurate": ACCURATE,
}
# ---------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------


from scipy.ndimage import zoom
import numpy as np

from core.image import Image


def downsample_image(image: Image, size: int) -> Image:
    """
    Downsample an image while preserving its world-space coordinate system.

    The voxel spacing is increased so that physical dimensions remain
    approximately unchanged. The world origin is preserved.
    """

    shape = np.asarray(image.data.shape[:3], dtype=float)

    max_dim = float(np.max(shape))
    if size > max_dim:
        raise ValueError(
            f"Requested size {size} exceeds largest image dimension {int(max_dim)}."
        )

    factor = max_dim / float(size)

    zoom_factor = 1.0 / factor

    resampled = zoom(
        image.data,
        zoom=(zoom_factor, zoom_factor, zoom_factor),
        order=1,
        mode="constant",
        cval=0.0,
    )

    affine = image.affine.copy()

    # Increase voxel spacing
    affine[:3, :3] *= factor

    # Preserve world origin
    # (No translation correction.)

    return Image(
        data=np.asarray(resampled),
        affine=affine,
        voxel_frame=image.voxel_frame,
        world_frame=image.world_frame,
    )


# ---------------------------------------------------------------------
# Similarity metric
# ---------------------------------------------------------------------


def normalized_cross_correlation(
    reference: np.ndarray,
    moving: np.ndarray,
) -> float:
    """
    Compute normalized cross correlation.

    Returns
    -------
    float
        Value between approximately -1 and 1.
        Larger values indicate better alignment.
    """

    a = reference.astype(float).ravel()
    b = moving.astype(float).ravel()

    a -= a.mean()
    b -= b.mean()

    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))

    if denominator == 0:
        return 0.0

    return float(np.sum(a * b) / denominator)


def get_quality_preset(name: str) -> RegistrationSettings:
    try:
        return QUALITY_PRESETS[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown registration quality '{name}'. Expected one of: fast, standard, accurate."
        ) from exc


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------


def center_of_mass_translation(
    moving: Image,
    reference: Image,
) -> tuple[float, float, float]:
    """
    Estimate an initial translation by aligning image centres of mass.
    """

    moving_voxel = np.asarray(center_of_mass(moving.data))
    reference_voxel = np.asarray(center_of_mass(reference.data))

    moving_world = (
        moving.affine
        @ np.append(moving_voxel, 1.0)
    )[:3]

    reference_world = (
        reference.affine
        @ np.append(reference_voxel, 1.0)
    )[:3]

    translation = reference_world - moving_world

    return tuple(translation)


# ---------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------


def affine_matrix(parameters: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 affine matrix.

    Parameters are

        tx ty tz
        rx ry rz
        sx sy sz

    Rotations are given in radians.
    """

    tx, ty, tz, rx, ry, rz, sx, sy, sz = parameters

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)],
        ]
    )

    Ry = np.array(
        [
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ]
    )

    Rz = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1],
        ]
    )

    rotation = Rz @ Ry @ Rx

    scaling = np.diag([sx, sy, sz])

    matrix = np.eye(4)

    matrix[:3, :3] = rotation @ scaling
    matrix[:3, 3] = [tx, ty, tz]

    return matrix

def registration_matrix(
    parameters: np.ndarray,
    moving_com_world: np.ndarray,
) -> np.ndarray:
    """
    Build the world-space transform used during registration.
    Rotation and scaling occur about the moving image centre of mass.
    """

    affine = affine_matrix(parameters)

    T1 = np.eye(4)
    T1[:3, 3] = -moving_com_world

    T2 = np.eye(4)
    T2[:3, 3] = moving_com_world

    return T2 @ affine @ T1


# ---------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------

_registration_calls = 0
def registration_cost(
    parameters: np.ndarray,
    moving: Image,
    reference: Image,
    moving_com_world: np.ndarray,
    sample_step: int = 1,
) -> float:
    """
    Objective minimized during optimization.
    """
    global _registration_calls
    _registration_calls += 1

    if _registration_calls % 25 == 0:
        print(f"Cost evaluation {_registration_calls}")
        
    matrix = registration_matrix(parameters, moving_com_world)

    transform = Transform(
        source=moving.world_frame,
        target=reference.world_frame,
        matrix=matrix,
    )

    reference_values, moving_values = sample_registration_points(
        moving_image=moving,
        reference_image=reference,
        moving_to_reference=transform,
        step=sample_step,
        debug = (_registration_calls % 50 == 0),
    )
    if _registration_calls % 25 == 0:
        print(f"Resampled shape: {moving_values.shape}")
    similarity = normalized_cross_correlation(
        reference_values,
        moving_values,
    )

    return -similarity


# ---------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------


def _translation_registration(
    initial: np.ndarray,
    moving: Image,
    reference: Image,
    moving_com_world: np.ndarray,
    sample_step: int,
    settings: RegistrationSettings,
) -> OptimizeResult:
    """
    Optimize translation only while keeping rotation and scaling fixed.
    """

    initial = initial.copy()

    bounds = [
        (None, None),
        (None, None),
        (None, None),

        (initial[3], initial[3]),
        (initial[4], initial[4]),
        (initial[5], initial[5]),

        (initial[6], initial[6]),
        (initial[7], initial[7]),
        (initial[8], initial[8]),
    ]

    return minimize(
        registration_cost,
        initial,
        args=(moving, reference, moving_com_world, sample_step),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": settings.translation_maxiter,
            "xtol": settings.xtol,
            "ftol": settings.ftol,
        },
    )


def _rigid_registration(
    initial: np.ndarray,
    moving: Image,
    reference: Image,
    moving_com_world: np.ndarray,
    sample_step: int,
    settings: RegistrationSettings,
) -> OptimizeResult:
    """
    Optimize translation and rotation.
    Scaling remains fixed.
    """

    initial = initial.copy()

    bounds = [
        (None, None),
        (None, None),
        (None, None),

        (-np.pi/2, np.pi/2),
        (-np.pi/2, np.pi/2),
        (-np.pi/2, np.pi/2),

        (initial[6], initial[6]),
        (initial[7], initial[7]),
        (initial[8], initial[8]),
    ]

    return minimize(
        registration_cost,
        initial,
        args=(moving, reference, moving_com_world, sample_step),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": settings.rigid_maxiter,
            "xtol": settings.xtol,
            "ftol": settings.ftol,
        },
    )

def _affine_registration(
    initial: np.ndarray,
    moving: Image,
    reference: Image,
    moving_com_world: np.ndarray,
    sample_step: int,
    settings: RegistrationSettings,
) -> OptimizeResult:
    """
    Optimize full affine transform.
    """

    initial = initial.copy()

    bounds = [
        (None, None),
        (None, None),
        (None, None),

        (-np.pi/2, np.pi/2),
        (-np.pi/2, np.pi/2),
        (-np.pi/2, np.pi/2),

        (0.7, 1.4),
        (0.7, 1.4),
        (0.7, 1.4),
    ]

    return minimize(
        registration_cost,
        initial,
        args=(moving, reference, moving_com_world, sample_step),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": settings.affine_maxiter,
            "xtol": settings.xtol,
            "ftol": settings.ftol,
        },
    )


import time

def _run_registration_pipeline(
    moving: Image,
    reference: Image,
    settings: RegistrationSettings,
) -> tuple[np.ndarray, OptimizeResult | None]:

    parameters = np.array([
        *center_of_mass_translation(moving, reference),
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ])

    print("=" * 70)
    print("Registration started")
    print(f"Moving image    : {moving.data.shape}")
    print(f"Reference image : {reference.data.shape}")
    print(f"Pyramid         : {settings.pyramid_sizes}")
    print(f"Translation     : {settings.translation}")
    print(f"Rigid           : {settings.rigid}")
    print(f"Affine          : {settings.affine}")
    print()
    print("Initial parameters:")
    print(parameters)
    print("=" * 70)

    last_result = None

    for level, (size, step) in enumerate(
    zip(settings.pyramid_sizes, settings.sample_steps),
    start=1,):

        print()
        print(f"========== Pyramid level {level} ==========")
        print(f"Target size : {size}")

        t0 = time.perf_counter()

        moving_level = downsample_image(moving, size)
        reference_level = downsample_image(reference, size)

        moving_com_voxel = center_of_mass(moving_level.data)

        moving_com_world = voxel_to_world(
            moving_level,
            np.asarray(moving_com_voxel)[None],
        )[0]

        print(type(moving_level))
        print(type(reference_level))

        print(f"Moving level    : {moving_level.data.shape}")
        print(f"Reference level : {reference_level.data.shape}")

        print(f"Downsample took {time.perf_counter()-t0:.2f} s")

        if settings.translation:

            print("\nRunning translation registration...")

            t = time.perf_counter()

            last_result = _translation_registration(
                parameters,
                moving_level,
                reference_level,
                moving_com_world,
                sample_step=step,
                settings=settings,
            )

            parameters = last_result.x

            transform = Transform(
                source=moving_level.world_frame,
                target=reference_level.world_frame,
                matrix=registration_matrix(parameters, moving_com_world)
            )


            show_registration_debug(
                moving=moving_level,
                reference=reference_level,
                transform=transform,
                title=f"Level {level} - Translation",
            )

            print(f"Finished in {time.perf_counter()-t:.2f} s")
            print("Parameters:", parameters)
            print("Cost:", last_result.fun)

        if settings.rigid:

            print("\nRunning rigid registration...")

            t = time.perf_counter()

            last_result = _rigid_registration(
                parameters,
                moving_level,
                reference_level,
                moving_com_world,
                settings=settings,
                sample_step=step
            )

            parameters = last_result.x

            transform = Transform(
                source=moving_level.world_frame,
                target=reference_level.world_frame,
                matrix=registration_matrix(parameters, moving_com_world),
            )


            show_registration_debug(
                moving=moving_level,
                reference=reference_level,
                transform=transform,
                title=f"Level {level} - rigid",
            )

            print(f"Finished in {time.perf_counter()-t:.2f} s")
            print("Parameters:", parameters)
            print("Cost:", last_result.fun)

        if settings.affine:

            print("\nRunning affine registration...")

            t = time.perf_counter()

            last_result = _affine_registration(
                parameters,
                moving_level,
                reference_level,
                moving_com_world,
                settings=settings,
                sample_step=step
            )

            parameters = last_result.x

            transform = Transform(
                source=moving_level.world_frame,
                target=reference_level.world_frame,
                matrix=registration_matrix(parameters, moving_com_world),
            )


            show_registration_debug(
                moving=moving_level,
                reference=reference_level,
                transform=transform,
                title=f"Level {level} - affine",
            )

            print(f"Finished in {time.perf_counter()-t:.2f} s")
            print("Parameters:", parameters)
            print("Cost:", last_result.fun)

    print("\nRegistration finished.")
    print("Final parameters:")
    print(parameters)

    return parameters, last_result


def _registration_report(
    parameters: np.ndarray,
    result: OptimizeResult | None,
    quality: str,
) -> RegistrationReport:
    translation_mm = float(np.linalg.norm(parameters[:3]))
    rotation_deg = float(np.linalg.norm(parameters[3:6]) * (180.0 / np.pi))
    iterations = int(getattr(result, "nit", 0) or 0)
    similarity = float(-getattr(result, "fun", 0.0)) if result is not None else 0.0
    return RegistrationReport(
        quality=quality,
        iterations=iterations,
        similarity=similarity,
        translation_mm=translation_mm,
        rotation_deg=rotation_deg,
    )


def registration_report_lines(report: RegistrationReport) -> list[str]:
    return [
        f"Iterations: {report.iterations}",
        f"Similarity: {report.similarity:0.2f}",
        f"Translation: {report.translation_mm:0.1f} mm",
        f"Rotation: {report.rotation_deg:0.1f}°",
    ]


def _validate_registration_inputs(moving: Image, reference: Image) -> None:
    if moving.data is None or reference.data is None:
        raise ValueError("Empty image data provided for registration.")

    if moving.data.ndim < 2 or reference.data.ndim < 2:
        raise ValueError("Images must be at least 2D.")

    if np.std(moving.data) == 0 or np.std(reference.data) == 0:
        raise ValueError("Cannot register constant-valued images.")

    if np.linalg.matrix_rank(moving.affine[:3, :3]) < 3:
        raise ValueError("Moving affine is rank deficient.")


def affine_registration(
    moving: Image,
    reference: Image,
    settings: RegistrationSettings = STANDARD,
) -> Transform:
    """
    Register two images using a multi-resolution affine registration
    pipeline.

    Parameters
    ----------
    moving
        Image to be transformed.

    reference
        Target image.

    settings
        Registration configuration controlling pyramid levels,
        enabled optimisation stages, and convergence criteria.

    Returns
    -------
    Transform
        Affine transform mapping moving.frame -> reference.frame.
    """
    _validate_registration_inputs(moving, reference)

    parameters, _result = _run_registration_pipeline(moving, reference, settings)

    moving_com_voxel = center_of_mass(moving.data)

    moving_com_world = voxel_to_world(
        moving,
        np.asarray(moving_com_voxel)[None],
    )[0]

    return Transform(
        source=moving.world_frame,
        target=reference.world_frame,
        matrix=registration_matrix(parameters, moving_com_world),
    )


def rigid_registration(
    moving: Image,
    reference: Image,
    settings: RegistrationSettings = STANDARD,
) -> Transform:
    rigid_settings = RegistrationSettings(
        pyramid_sizes=settings.pyramid_sizes,
        sample_steps=settings.sample_steps,
        translation=True,
        rigid=True,
        affine=False,
        translation_maxiter=settings.translation_maxiter,
        rigid_maxiter=settings.rigid_maxiter,
        affine_maxiter=settings.affine_maxiter,
        xtol=settings.xtol,
        ftol=settings.ftol,

    )
    return affine_registration(moving, reference, rigid_settings)


def register_images(
    moving: Image,
    reference: Image,
    quality: str = "standard",
) -> tuple[Transform, RegistrationReport]:
    _validate_registration_inputs(moving, reference)
    settings = get_quality_preset(quality)
    parameters, result = _run_registration_pipeline(moving, reference, settings)
    moving_com_voxel = center_of_mass(moving.data)
    moving_com_world = voxel_to_world(
        moving,
        np.asarray(moving_com_voxel)[None],
    )[0]
    transform = Transform(
        source=moving.world_frame,
        target=reference.world_frame,
        matrix=registration_matrix(parameters, moving_com_world),
    )
    report = _registration_report(parameters, result, quality)
    return transform, report

import numpy as np
import matplotlib.pyplot as plt



def show_registration_debug(
    moving: Image,
    reference: Image,
    transform: Transform,
    title: str = "Registration debug",
):
    """
    Compact registration diagnostics.

    Focuses on:
        - orientation
        - voxel spacing
        - transform sanity
        - registration quality
        - visual comparison
    """

    import nibabel as nib

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    # -------------------------------------------------------------
    # Basic image information
    # -------------------------------------------------------------

    print("\nReference")
    print(f"  Shape       : {reference.shape}")
    print(f"  Affine       : {reference.affine}")
    print(f"  Orientation : {nib.aff2axcodes(reference.affine)}")

    print("\nMoving")
    print(f"  Shape       : {moving.shape}")
    print(f"  Affine       : {moving.affine}")
    print(f"  Orientation : {nib.aff2axcodes(moving.affine)}")

    ref_spacing = np.linalg.norm(reference.affine[:3, :3], axis=0)
    mov_spacing = np.linalg.norm(moving.affine[:3, :3], axis=0)

    print("\nVoxel spacing (mm)")
    print("  Reference :", np.round(ref_spacing, 2))
    print("  Moving    :", np.round(mov_spacing, 2))

    # -------------------------------------------------------------
    # Sample moving image
    # -------------------------------------------------------------

    reference_values, moving_resampled = sample_registration_points(
        moving_image=moving,
        reference_image=reference,
        moving_to_reference=transform,
        step=1,
    )

    reference_volume = reference.data.astype(np.float32)
    moving_volume = moving_resampled.astype(np.float32)

    # -------------------------------------------------------------
    # Transform centre of reference image
    # -------------------------------------------------------------

    centre_voxel = (np.array(reference.shape) - 1) / 2

    centre_world = voxel_to_world(
        reference,
        centre_voxel[None],
    )[0]

    inverse = np.linalg.inv(transform.matrix)

    moving_world = (
        inverse
        @ np.append(centre_world, 1)
    )[:3]

    moving_voxel = world_to_voxel(
        moving,
        moving_world[None],
    )[0]

    print("\nCentre mapping")
    print(f"  Reference voxel : {np.round(centre_voxel,2)}")
    print(f"  Moving voxel    : {np.round(moving_voxel,2)}")

    inside = np.all(
        (moving_voxel >= 0)
        &
        (moving_voxel < np.array(moving.shape))
    )

    print(f"  Inside moving   : {inside}")

    # -------------------------------------------------------------
    # Sampling statistics
    # -------------------------------------------------------------

    x = np.arange(reference.shape[0])
    y = np.arange(reference.shape[1])
    z = np.arange(reference.shape[2])

    grid = np.meshgrid(
        x,
        y,
        z,
        indexing="ij",
    )

    voxel_ref = np.stack(
        [g.ravel() for g in grid],
        axis=1,
    )

    world_ref = voxel_to_world(
        reference,
        voxel_ref,
    )

    moving_world = (
        inverse
        @ np.column_stack(
            (world_ref, np.ones(len(world_ref)))
        ).T
    ).T[:, :3]

    moving_voxel = world_to_voxel(
        moving,
        moving_world,
    )

    inside = np.all(
        (moving_voxel >= 0)
        &
        (moving_voxel < np.array(moving.shape)),
        axis=1,
    )

    print("\nSampling")
    print(f"  Inside voxels : {inside.mean()*100:.1f}%")

    # -------------------------------------------------------------
    # NCC
    # -------------------------------------------------------------

    ref = reference_volume.copy()
    mov = moving_volume.copy()

    ref -= ref.min()
    mov -= mov.min()

    if ref.max():
        ref /= ref.max()

    if mov.max():
        mov /= mov.max()

    ncc = np.corrcoef(
        ref.ravel(),
        mov.ravel(),
    )[0, 1]

    print(f"\nWhole-volume NCC : {ncc:.4f}")

    # -------------------------------------------------------------
    # Display
    # -------------------------------------------------------------

    cx = ref.shape[0] // 2
    cy = ref.shape[1] // 2
    cz = ref.shape[2] // 2

    slices = [
        (ref[cx,:,:], mov[cx,:,:], "Sagittal"),
        (ref[:,cy,:], mov[:,cy,:], "Coronal"),
        (ref[:,:,cz], mov[:,:,cz], "Axial"),
    ]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(10,10),
        constrained_layout=True,
    )

    fig.suptitle(title)

    headers = [
        "Reference",
        "Moving",
        "Overlay",
    ]

    for j, h in enumerate(headers):
        axes[0, j].set_title(h)

    for row, (r, m, label) in enumerate(slices):

        r = np.rot90(r)
        m = np.rot90(m)

        overlay = np.zeros((*r.shape,3))
        overlay[...,0] = r
        overlay[...,1] = m

        axes[row,0].imshow(r, cmap="gray", origin="lower")
        axes[row,1].imshow(m, cmap="gray", origin="lower")
        axes[row,2].imshow(overlay, origin="lower")

        axes[row,0].set_ylabel(label)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.show(block=False)