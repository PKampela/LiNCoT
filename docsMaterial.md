## Working notes for documentation

This file is intentionally more detailed than the README. It is meant to be a staging area for future docs, reference material, and content that should not crowd the project landing page for TMSLabs.

## Important architectural information

Nibabel is used to canonise all mri images into RAS format

## Command reference

The CLI is interactive and stateful. It uses the general pattern `<command> <object> [parameters] [--flags]`.

Available commands:

- `point add <name> <x> <y> <z> <frame>` - add a point to the session.
- `point list` - list all stored points.
- `frame list` - list all registered frames.
- `transform <point> <target> [--chain name1 name2] [--show-chain] [--show-matrix] [--explain] [--json]` - transform a stored point into another frame.
- `transform list` - list all registered transforms.
- `volume load <path> [--json]` - load an MRI volume into the session.
- `volume import <path> [--json]` - alias for `volume load`.
- `surface list` - list all loaded surfaces.
- `view surface <name> [--json]` - open a surface viewer for a loaded surface.
- `view volume <name> [--json]` - open a volume viewer for a loaded volume.
- `session summary` - show a compact summary of the current session.
- `help` - list the available commands.

Example CLI session:

```text
> frame list
> point add target 12 34 56 head
> transform target mni --show-chain
> session summary
> exit
```

## Python API notes

The Python API exposes the underlying transform and import helpers used by the CLI and GUI. These are useful for scripting and for implementing a proper docs site later.

Transform helpers:

- `load_mne_transform`
- `head_to_mri_transform`
- `mri_to_mni_transform`

Volume helpers:

- `load_nifti`
- `load_nifti_image`
- `voxel_to_world_transform`
- `world_to_voxel_transform`
- `transform_image`
- `sample_volume`

Surface helpers:

- `transform_surface`
- `closest_vertex`
- `distance_to_surface`
- `bounding_box`
- `face_normals`
- `project_to_surface`
- `projection_normal`

## Basic workflows

These examples are intentionally short and can later be expanded into real tutorials.

### TMS workflow

```text
> volume load subject.nii.gz --name subject_mri --register-transform
> point add motor_hotspot 10 20 30 head
> transform motor_hotspot mri --show-chain --show-matrix
> view volume subject_mri
> session summary
> exit
```

### Python transform example

```python
import numpy as np

from core.frames import CoordinateFrame
from core.point import Point
from core.transform import Transform

head = CoordinateFrame("head", ("R", "A", "S"), "mm")
mri = CoordinateFrame("mri", ("R", "A", "S"), "mm")

transform = Transform(head, mri, np.eye(4))
point = Point(np.array([1.0, 2.0, 3.0]), head)
result = transform.apply(point)

print(result.frame.name)
print(result.coords)
```

## Notes for future docs

- Move the command reference into a proper generated CLI reference when a docs site exists.
- Split Python helpers into API pages grouped by transforms, volumes, and surfaces.
- Add rendered examples and screenshots for the GUI viewer workflow when the docs system is ready.

## Development notes

Build, test, and troubleshooting notes can live here until a formal docs site exists.

### Build and publish

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine upload dist/*
```

### Testing

```bash
python -m pip install -e .
python -m pytest -q
```

### Troubleshooting

- Prefer absolute package imports and avoid `sys.path` hacks.
- On Windows, create the recommended conda environment before installing if binary dependency issues appear.
