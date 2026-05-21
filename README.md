# TMSCoords

TMSCoords is a lightweight neuroimaging coordinate transformation tool for TMS workflows. It keeps coordinate frames explicit, treats transforms as first-class objects, and provides both a Python API and a stateful interactive CLI.

## Design goals

- Every point carries its coordinate frame.
- Transformations are explicit, inspectable, and invertible.
- Backends are isolated from the core domain model.
- The CLI is session-based, so points, images, frames, and loaded transforms can be reused across commands.

For a fuller command reference and Python API notes, see [docsMaterial.md](docsMaterial.md).

## Environment

The repository includes a Conda environment definition in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate tmslabs
```

## Neuroimaging operations

The codebase separates its neuroimaging operations into a few clear areas so you can use the pieces independently or together.

### Coordinate frames and points

- Define named coordinate frames with explicit axes and units.
- Create points that always carry their originating frame.
- Add points to a session and list the points that are already stored.

Typical operations:

- frame list
- point add
- point list

### Transform operations

- Apply a transform between two frames to a stored point.
- Resolve multi-step transform chains automatically from the registered transforms.
- Inspect the selected chain, the composed matrix, or an explanation of how the chain was chosen.

Typical operations:

- transform
- transform list

See [docsMaterial.md](docsMaterial.md) for command details and Python helpers.

### Volume and image operations

- Load a NIfTI volume into the current session.
- Optionally register a voxel-to-world transform for the loaded image.
- Resample an image through an affine transform.
- Sample image values at arbitrary world-space coordinates.
- Open a volume viewer for an image already in the session.

Typical operations:

- volume load
- volume import
- view volume

### Surface operations

- List loaded surfaces in the session.
- Transform a surface mesh through an affine transform.
- Query the closest vertex to a point.
- Compute point-to-surface distance.
- Compute a surface bounding box.
- Open a surface viewer for an already loaded mesh.

Typical operations:

- surface list
- view surface

### Session and registry operations

- Show a compact summary of the current session state.
- List the available commands through the built-in help command.
- Load default frames at startup.
- Load transform definitions from transforms.json when present.

Typical operations:

- session summary
- help

## Running the project

The launcher supports both interfaces:

```bash
python launcher.py --cli
python launcher.py --gui
```

`--gui` is the default when no flag is provided. The CLI can also be started directly with:

```bash
python -m cli.main
```

## CLI overview

The current CLI is interactive and stateful. You start a session, then enter commands at the prompt.

- Command syntax: `<command> <object> [parameters] [--flags]`
- Exit commands: `quit` and `exit`
- For point transforms, the source frame is inferred from the stored point. A source frame can still be provided as a validation check when needed.

Example session:

```text
> frame list
> point add target 12 34 56 head
> transform target mni --show-chain
> session summary
> exit
```

## TMS workflow example

The following session shows a realistic TMS-style flow: load an MRI volume, register its voxel-to-world transform, define a scalp target in head space, convert it into MRI space, and inspect the session.

```text
> volume load subject.nii.gz --name subject_mri --register-transform
> point add motor_hotspot 10 20 30 head
> transform motor_hotspot mri --show-chain --show-matrix
> view volume subject_mri
> session summary
> exit
```

In a full workflow, the transformed point can be used to guide coil placement, compare against anatomical surfaces, or sample values from the MRI volume at the target location.

For the full command and API notes, see [docsMaterial.md](docsMaterial.md).

## Installation

Recommended (developer) install using pip in editable mode:

```bash
python -m pip install -e .
```

Standard install (non-editable):

```bash
python -m pip install .
```

If you prefer conda for binary dependencies (`PySide6`, `pyvista`, `mne`), create the environment from the supplied file:

```bash
conda env create -f environment.yml
conda activate tmslabs
python -m pip install -e .
```

## Quick usage

CLI (after install) — runs the interactive session or start directly from source:

```bash
# installed entrypoint
tmscoords --cli

# or from source
python launcher.py --cli
```

Start the GUI:

```bash
tmscoords --gui
# or
python launcher.py --gui
```

For build notes, tests, troubleshooting, and API details, see [docsMaterial.md](docsMaterial.md).

