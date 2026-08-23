from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass, map_coordinates
from scipy.optimize import OptimizeResult, minimize
from scipy.ndimage import zoom

from dataclasses import dataclass

from .frames import CoordinateFrame
from .image import Image
from .sampling import sample_registration_points, world_to_voxel, voxel_to_world, RegistrationSamples, sample_moving_image_at_reference_points
from .transform import Transform


@dataclass(frozen=True)
class RegistrationSettings:

    pyramid_sizes: tuple[int, ...]
    sample_steps: tuple[int, ...]

    translation: bool = True
    rigid: bool = True
    affine: bool = True

    translation_maxiter: int = 50
    rigid_maxiter: int = 50
    affine_maxiter: int = 50

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

    translation_maxiter = 20,
    rigid_maxiter = 30,
    affine_maxiter = 50,
)

STANDARD = RegistrationSettings(
    pyramid_sizes=(32, 64, 128),
    sample_steps=(4, 2, 1),
    translation=True,
    rigid=True,
    affine=True,

    translation_maxiter = 30,
    rigid_maxiter = 50,
    affine_maxiter = 80,
)

ACCURATE = RegistrationSettings(
    pyramid_sizes=(32, 64, 128, 192),
    sample_steps=(4, 2, 1, 1),
    translation=True,
    rigid=True,
    affine=True,

    translation_maxiter = 50,
    rigid_maxiter = 80,
    affine_maxiter = 100,

    xtol = 1e-3,
    ftol = 1e-4,
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
    Downsample a 3D image while preserving its world-space geometry.

    The largest output dimension is approximately ``size`` voxels.
    The output grid remains aligned with the original image's voxel
    axes and world-space orientation.

    No anatomical reorientation is performed. The orientation encoded
    by ``image.affine`` is preserved.
    """
    if image.data is None:
        raise ValueError("Cannot downsample an image without voxel data")

    if image.data.ndim != 3:
        raise ValueError("Registration downsampling requires a 3D image")

    original_shape = np.asarray(image.data.shape, dtype=int)

    max_dim = int(original_shape.max())

    if size <= 0:
        raise ValueError("Downsampling size must be positive")

    if size >= max_dim:
        return Image(
            data=np.asarray(image.data),
            affine=image.affine.copy(),
            voxel_frame=image.voxel_frame,
            world_frame=image.world_frame,
        )

    # ---------------------------------------------------------------
    # Determine the downsampling factor.
    #
    # Example:
    #   256 x 256 x 160
    #   size = 64
    #
    # factor = 4
    # output ≈ 64 x 64 x 40
    # ---------------------------------------------------------------

    factor = max_dim / float(size)

    output_shape = np.maximum(
        1,
        np.round(original_shape / factor).astype(int),
    )

    # ---------------------------------------------------------------
    # Preserve the physical voxel axes.
    #
    # The original affine columns describe the world-space direction
    # and spacing of each voxel axis.
    #
    # We increase the voxel spacing by the corresponding reduction
    # factor.
    # ---------------------------------------------------------------

    original_axis_vectors = image.affine[:3, :3]

    original_spacing = np.linalg.norm(
        original_axis_vectors,
        axis=0,
    )

    new_spacing = original_spacing * factor

    directions = original_axis_vectors / original_spacing

    new_affine = image.affine.copy()
    new_affine[:3, :3] = directions * new_spacing

    # ---------------------------------------------------------------
    # Preserve the physical centre of the image.
    #
    # This is important. Simply keeping affine[:3, 3] unchanged while
    # changing the voxel spacing moves the centre of the represented
    # volume.
    # ---------------------------------------------------------------

    original_center_voxel = (original_shape - 1) / 2.0
    original_center_world = (
        image.affine
        @ np.append(original_center_voxel, 1.0)
    )[:3]

    output_center_voxel = (output_shape - 1) / 2.0

    new_affine[:3, 3] = (
        original_center_world
        - new_affine[:3, :3] @ output_center_voxel
    )

    # ---------------------------------------------------------------
    # Resample by constructing output-grid voxel coordinates and
    # mapping them back into the original voxel space.
    # ---------------------------------------------------------------

    grid = np.meshgrid(
        np.arange(output_shape[0], dtype=float),
        np.arange(output_shape[1], dtype=float),
        np.arange(output_shape[2], dtype=float),
        indexing="ij",
    )

    output_voxels = np.stack(
        [g.ravel() for g in grid],
        axis=1,
    )

    output_world = voxel_to_world(
        Image(
            data=None,
            affine=new_affine,
            voxel_frame=image.voxel_frame,
            world_frame=image.world_frame,
            shape_metadata=tuple(output_shape),
        ),
        output_voxels,
    )

    input_voxels = world_to_voxel(
        image,
        output_world,
    )

    # scipy.ndimage.map_coordinates expects one coordinate array
    # per input dimension.
    resampled = map_coordinates(
        image.data,
        [
            input_voxels[:, 0],
            input_voxels[:, 1],
            input_voxels[:, 2],
        ],
        order=1,
        mode="constant",
        cval=0.0,
    )

    resampled = resampled.reshape(tuple(output_shape))

    return Image(
        data=np.asarray(resampled),
        affine=new_affine,
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

def registration_cost(
    parameters: np.ndarray,
    moving: Image,
    moving_com_world: np.ndarray,
    samples: RegistrationSamples,
) -> float:

    matrix = registration_matrix(
        parameters,
        moving_com_world,
    )

    moving_values = sample_moving_image_at_reference_points(
        moving_image=moving,
        reference_points_world=samples.reference_points_world,
        moving_to_reference_matrix=matrix,
    )

    similarity = normalized_cross_correlation(
        samples.reference_values,
        moving_values,
    )

    return -similarity


def prepare_registration_samples(
    reference_image: Image,
    step: int = 1,
) -> RegistrationSamples:
    """
    Precompute the fixed reference sampling locations and intensities.

    These values are independent of the registration parameters and
    therefore only need to be calculated once per pyramid level.
    """

    if reference_image.data is None:
        raise ValueError(
            "Registration sampling requires an image with voxel data"
        )

    x = np.arange(0, reference_image.shape[0], step)
    y = np.arange(0, reference_image.shape[1], step)
    z = np.arange(0, reference_image.shape[2], step)

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

    reference_values = reference_image.data[
        voxel_ref[:, 0].astype(int),
        voxel_ref[:, 1].astype(int),
        voxel_ref[:, 2].astype(int),
    ]

    reference_points_world = voxel_to_world(
        reference_image,
        voxel_ref,
    )

    return RegistrationSamples(
        reference_frame=reference_image.world_frame,
        reference_points_world=reference_points_world,
        reference_values=reference_values,
    )



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

    initial = initial.copy()

    samples = prepare_registration_samples(
        reference,
        step=sample_step,
    )

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
        args=(
            moving,
            moving_com_world,
            samples,
        ),
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
    samples = prepare_registration_samples(
        reference,
        step=sample_step,
    )

    initial = initial.copy()
    max_rotation = np.deg2rad(30.0)

    bounds = [
        (None, None),
        (None, None),
        (None, None),

        (-max_rotation, max_rotation),
        (-max_rotation, max_rotation),
        (-max_rotation, max_rotation),

        (initial[6], initial[6]),
        (initial[7], initial[7]),
        (initial[8], initial[8]),
    ]

    return minimize(
        registration_cost,
        initial,
        args=(moving, moving_com_world, samples),
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

    samples = prepare_registration_samples(
        reference,
        step=sample_step,
    )

    initial = initial.copy()
    max_rotation = np.deg2rad(30.0)

    bounds = [
        (None, None),
        (None, None),
        (None, None),

        (-max_rotation, max_rotation),
        (-max_rotation, max_rotation),
        (-max_rotation, max_rotation),

        (0.7, 1.4),
        (0.7, 1.4),
        (0.7, 1.4),
    ]

    return minimize(
        registration_cost,
        initial,
        args=(moving, moving_com_world, samples),
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
) -> tuple[np.ndarray, OptimizeResult | None, np.ndarray]:

    parameters = np.array([
        *center_of_mass_translation(moving, reference),
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ])

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

        print("\n--- Geometry after downsampling ---")
        import nibabel as nib
        for name, image in [
            ("moving", moving_level),
            ("reference", reference_level),
        ]:
            print(name)
            print("  shape:", image.shape)
            print("  axcodes:", nib.aff2axcodes(image.affine))
            print("  affine:")
            print(image.affine)

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

            # show_registration_debug(
            #     moving=moving_level,
            #     reference=reference_level,
            #     transform=transform,
            #     title=f"Level {level} - affine",
            # )


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

            print("Parameters:", parameters)
            print(
                "Rigid rotation (deg):",
                np.round(np.rad2deg(parameters[3:6]), 3),
            )

            transform = Transform(
                source=moving_level.world_frame,
                target=reference_level.world_frame,
                matrix=registration_matrix(parameters, moving_com_world),
            )

            # show_registration_debug(
            #     moving=moving_level,
            #     reference=reference_level,
            #     transform=transform,
            #     title=f"Level {level} - affine",
            # )


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

            print("Parameters:", parameters)
            print(
                "Rigid rotation (deg):",
                np.round(np.rad2deg(parameters[3:6]), 3),
            )

            transform = Transform(
                source=moving_level.world_frame,
                target=reference_level.world_frame,
                matrix=registration_matrix(parameters, moving_com_world),
            )



            # show_registration_debug(
            #     moving=moving_level,
            #     reference=reference_level,
            #     transform=transform,
            #     title=f"Level {level} - affine",
            # )

            print(f"Finished in {time.perf_counter()-t:.2f} s")
            print("Parameters:", parameters)
            print("Cost:", last_result.fun)

    print("\nRegistration finished.")
    print("Final parameters:")
    print(parameters)

    print("Final matrix:")
    print(transform.matrix)

    return parameters, last_result, moving_com_world


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

    parameters, _result, moving_com_world = _run_registration_pipeline(moving, reference, settings)

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
    parameters, result, moving_com_world = _run_registration_pipeline(moving, reference, settings)

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
    Display registration diagnostics using anatomical orientations.

    The moving image is resampled onto the reference voxel grid using
    the supplied moving->reference world-space transform.  Reference
    and moving slices are then displayed using the reference anatomical
    planes rather than assuming that raw array axes are sagittal,
    coronal and axial.

    Rows:
        Axial
        Coronal
        Sagittal

    Columns:
        Reference
        Moving
        Overlay
    """

    import nibabel as nib
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------
    # Basic image information
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("\nReference")
    print(f"  Shape       : {reference.shape}")
    print(f"  Affine      :\n{reference.affine}")
    print(f"  Orientation : {nib.aff2axcodes(reference.affine)}")

    print("\nMoving")
    print(f"  Shape       : {moving.shape}")
    print(f"  Affine      :\n{moving.affine}")
    print(f"  Orientation : {nib.aff2axcodes(moving.affine)}")

    ref_spacing = np.linalg.norm(
        reference.affine[:3, :3],
        axis=0,
    )
    mov_spacing = np.linalg.norm(
        moving.affine[:3, :3],
        axis=0,
    )

    print("\nVoxel spacing (mm)")
    print("  Reference :", np.round(ref_spacing, 2))
    print("  Moving    :", np.round(mov_spacing, 2))

    # -------------------------------------------------------------
    # Sample moving image onto reference voxel grid
    # -------------------------------------------------------------

    reference_values, moving_resampled = sample_registration_points(
        moving_image=moving,
        reference_image=reference,
        moving_to_reference=transform,
        step=1,
    )

    reference_volume = np.asarray(
        reference_values,
        dtype=np.float32,
    ).reshape(reference.shape)

    moving_volume = np.asarray(
        moving_resampled,
        dtype=np.float32,
    )

    # -------------------------------------------------------------
    # Transform centre of reference image
    # -------------------------------------------------------------

    centre_voxel = (
        np.asarray(reference.shape, dtype=float) - 1.0
    ) / 2.0

    centre_world = voxel_to_world(
        reference,
        centre_voxel[None],
    )[0]

    inverse = np.linalg.inv(transform.matrix)

    moving_world = (
        inverse @ np.append(centre_world, 1.0)
    )[:3]

    moving_voxel = world_to_voxel(
        moving,
        moving_world[None],
    )[0]

    print("\nCentre mapping")
    print(
        f"  Reference voxel : "
        f"{np.round(centre_voxel, 2)}"
    )
    print(
        f"  Reference world : "
        f"{np.round(centre_world, 2)}"
    )
    print(
        f"  Moving voxel    : "
        f"{np.round(moving_voxel, 2)}"
    )
    print(
        f"  Moving world    : "
        f"{np.round(moving_world, 2)}"
    )

    inside = np.all(
        (moving_voxel >= 0)
        & (moving_voxel < np.asarray(moving.shape)),
    )

    print(f"  Inside moving   : {inside}")

    # -------------------------------------------------------------
    # Sampling diagnostics
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

    moving_world_all = (
        inverse
        @ np.column_stack(
            (world_ref, np.ones(len(world_ref)))
        ).T
    ).T[:, :3]

    moving_voxel_all = world_to_voxel(
        moving,
        moving_world_all,
    )

    inside = np.all(
        (moving_voxel_all >= 0)
        & (
            moving_voxel_all
            < np.asarray(moving.shape)
        ),
        axis=1,
    )

    print("\nSampling")
    print(
        f"  Inside voxels : "
        f"{inside.mean() * 100:.1f}%"
    )

    # -------------------------------------------------------------
    # NCC
    # -------------------------------------------------------------

    ref = reference_volume.copy()
    mov = moving_volume.copy()

    ref -= ref.min()
    mov -= mov.min()

    if ref.max() > 0:
        ref /= ref.max()

    if mov.max() > 0:
        mov /= mov.max()

    ncc = np.corrcoef(
        ref.ravel(),
        mov.ravel(),
    )[0, 1]

    print(f"\nWhole-volume NCC : {ncc:.4f}")

    # -------------------------------------------------------------
    # Anatomical orientation and deterministic plane extraction
    # -------------------------------------------------------------

    ref_codes = nib.aff2axcodes(reference.affine)

    print("\nReference anatomical orientation:")
    print(f"  Axis 0 : {ref_codes[0]}")
    print(f"  Axis 1 : {ref_codes[1]}")
    print(f"  Axis 2 : {ref_codes[2]}")

    affine_basis = np.asarray(reference.affine, dtype=float)[:3, :3]

    def _axis_mapping_from_affine() -> dict[str, tuple[int, int, str]]:
        # Choose one-to-one voxel-axis assignment that best aligns with
        # world X/Y/Z directions. This is robust for permuted/oblique affines.
        best_perm: tuple[int, int, int] | None = None
        best_score = float("-inf")

        for a0 in (0, 1, 2):
            for a1 in (0, 1, 2):
                if a1 == a0:
                    continue
                for a2 in (0, 1, 2):
                    if a2 == a0 or a2 == a1:
                        continue

                    score = (
                        abs(float(affine_basis[0, a0]))
                        + abs(float(affine_basis[1, a1]))
                        + abs(float(affine_basis[2, a2]))
                    )

                    if score > best_score:
                        best_score = score
                        best_perm = (a0, a1, a2)

        if best_perm is None:
            raise ValueError("Failed to derive anatomical axis mapping from affine")

        lr_axis, ap_axis, si_axis = best_perm

        lr_sign = 1 if float(affine_basis[0, lr_axis]) >= 0.0 else -1
        ap_sign = 1 if float(affine_basis[1, ap_axis]) >= 0.0 else -1
        si_sign = 1 if float(affine_basis[2, si_axis]) >= 0.0 else -1

        return {
            "lr": (lr_axis, lr_sign, "R" if lr_sign > 0 else "L"),
            "ap": (ap_axis, ap_sign, "A" if ap_sign > 0 else "P"),
            "si": (si_axis, si_sign, "S" if si_sign > 0 else "I"),
        }

    axis_mapping = _axis_mapping_from_affine()

    print("\nResolved anatomical mapping:")
    for group in ("lr", "ap", "si"):
        voxel_axis, sign, code = axis_mapping[group]
        print(
            f"  {group.upper()}: voxel axis {voxel_axis}, code {code}, sign {sign}"
        )

    def _extract_oriented_slice(
        volume: np.ndarray,
        fixed_axis: int,
        slice_index: int,
        row_axis: int,
        col_axis: int,
        row_sign: int,
        col_sign: int,
    ) -> np.ndarray:
        slice_data = np.take(volume, slice_index, axis=fixed_axis)

        remaining_axes = [
            axis for axis in range(3)
            if axis != fixed_axis
        ]

        row_position = remaining_axes.index(row_axis)
        col_position = remaining_axes.index(col_axis)

        oriented = np.transpose(
            slice_data,
            axes=(row_position, col_position),
        )

        if row_sign > 0:
            oriented = np.flip(oriented, axis=0)
        if col_sign > 0:
            oriented = np.flip(oriented, axis=1)

        return np.asarray(oriented)

    # Row order is fixed and independent of image orientation.
    plane_specs = (
        ("Axial", "si", "ap", "lr"),
        ("Coronal", "ap", "si", "lr"),
        ("Sagittal", "lr", "si", "ap"),
    )

    centre = np.asarray(reference.shape, dtype=int) // 2
    slices: list[tuple[np.ndarray, np.ndarray, str]] = []

    for label, fixed_group, row_group, col_group in plane_specs:
        fixed_axis, _fixed_sign, _fixed_code = axis_mapping[fixed_group]
        row_axis, row_sign, _row_code = axis_mapping[row_group]
        col_axis, col_sign, _col_code = axis_mapping[col_group]

        slice_index = int(centre[fixed_axis])

        reference_slice = _extract_oriented_slice(
            ref,
            fixed_axis=fixed_axis,
            slice_index=slice_index,
            row_axis=row_axis,
            col_axis=col_axis,
            row_sign=row_sign,
            col_sign=col_sign,
        )

        moving_slice = _extract_oriented_slice(
            mov,
            fixed_axis=fixed_axis,
            slice_index=slice_index,
            row_axis=row_axis,
            col_axis=col_axis,
            row_sign=row_sign,
            col_sign=col_sign,
        )

        slices.append((reference_slice, moving_slice, label))

    # -------------------------------------------------------------
    # Display
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(10, 10),
        constrained_layout=True,
    )

    fig.suptitle(title)

    headers = (
        "Reference",
        "Moving",
        "Overlay",
    )

    for column, header in enumerate(headers):
        axes[0, column].set_title(header)

    for row, (reference_slice, moving_slice, label) in enumerate(slices):

        reference_slice = np.asarray(reference_slice)
        moving_slice = np.asarray(moving_slice)

        overlay = np.zeros(
            (*reference_slice.shape, 3),
            dtype=np.float32,
        )

        overlay[..., 0] = reference_slice
        overlay[..., 1] = moving_slice

        axes[row, 0].imshow(
            reference_slice,
            cmap="gray",
            origin="lower",
        )

        axes[row, 1].imshow(
            moving_slice,
            cmap="gray",
            origin="lower",
        )

        axes[row, 2].imshow(
            overlay,
            origin="lower",
        )

        axes[row, 0].set_ylabel(label)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.show(block=False)