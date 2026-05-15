import os

import numpy as np

from core.import_service import import_transform
from core.point import Point
from core.session import Session


def run_test() -> None:
    print("1. Creating a Session...")
    session = Session()

    fif_path = os.path.join("dataset", "trans", "sample_audvis_raw-trans.fif")
    print(f"2. Importing a transform from {fif_path}...")
    transform, info = import_transform(session, fif_path)
    print(info)

    source_frame = transform.source
    point = Point(np.array([10.0, 20.0, 30.0]), source_frame)

    print(f"3. Transforming point {point.coords} in {point.frame.name}...")
    transformed_point = transform.apply(point)

    print("4. Results:")
    print(f"   Original point: {point.coords} in {point.frame.name}")
    print(f"   Transformed point: {transformed_point.coords} in {transformed_point.frame.name}")

    print("5. Verification:")
    if transformed_point.frame == transform.target:
        print("   ✓ Frame correctly transformed")
    else:
        print("   ✗ Frame transformation failed")

    print("\nWorkflow test completed successfully!")


if __name__ == "__main__":
    run_test()
