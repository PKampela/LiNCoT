import os
import sys

from core.session import Session


def main() -> int:
    current_dir = os.getcwd()
    fif_path = os.path.join(current_dir, "dataset", "trans", "sample_audvis_raw-trans.fif")

    session = Session()
    print("Created Session object")

    transform, message = session.import_transform(fif_path)
    print(f"Imported transform: {message}")

    all_transforms = session.transforms.list_transforms()
    print(f"Transforms in session: {all_transforms}")

    if not all_transforms:
        print("FAIL: No transforms registered in session")
        return 1

    source_frame = transform.source
    target_frame = transform.target
    print(f"Source Frame Name: {source_frame.name if source_frame else 'None'}")
    print(f"Target Frame Name: {target_frame.name if target_frame else 'None'}")

    if not source_frame or not target_frame:
        print("FAIL: Source or Target frame is missing")
        return 1

    matrix = transform.matrix
    print("Transform matrix shape:", matrix.shape)
    print("Transform matrix:\n", matrix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
