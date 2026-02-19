# TMSCoords

A lightweight coordinate transformation tool for TMS workflows. This project wraps established neuroimaging libraries to make coordinate transformations safer and more explicit.

## Design goals

- Every point carries its coordinate frame.
- Transformations are explicit, inspectable, and invertible.
- Backends are isolated from user-facing APIs.
- Single-point or small-set transformations only.

## Quick example

```python
from ..core.frames import CoordinateFrame
from ..core.point import Point
from ..core.transform import Transform

import numpy as np

head = CoordinateFrame("head", ("R", "A", "S"), "mm")
mri = CoordinateFrame("mri", ("R", "A", "S"), "mm")

transform = Transform(head, mri, np.eye(4))
point = Point(np.array([1.0, 2.0, 3.0]), head)
result = transform.apply(point)
print(result.coords)
```

## CLI (minimal)

By default, the CLI looks for a transform registry named `transforms.json` in the
current working directory and auto-resolves a valid transform chain.

```
python -m tmscoords.cli.main transform \
  --point 12 34 56 \
  --from head \
  --to mni
```

To override resolution with an explicit chain:

```
python -m tmscoords.cli.main transform \
  --point 12 34 56 \
  --from head \
  --to mni \
  --transform-file transforms.json \
  --transform-name head_to_mri mri_to_mni
```

Optional flags:

- `--show-matrix`: print the composed affine matrix.
- `--show-chain`: print a concise summary of the chain used.
- `--explain`: narrate how the chain was resolved.
- `--json`: emit structured JSON output instead of text.
