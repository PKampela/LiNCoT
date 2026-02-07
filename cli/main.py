"""CLI entry point for tmscoords."""

from __future__ import annotations

import argparse
from typing import List, Mapping

import numpy as np

from ..core.chain import TransformChain
from ..core.frames import CoordinateFrame
from ..core.point import Point
from ..registry.frames import FrameRegistry, default_frames
from ..registry.transforms import TransformRegistry


def _build_default_frame_registry() -> FrameRegistry:
    registry = FrameRegistry()
    registry.register_many(default_frames())
    return registry


def _load_transform_chain(
    transform_file: str,
    transform_names: List[str],
    frames: Mapping[str, CoordinateFrame],
) -> TransformChain:
    registry = TransformRegistry.load(transform_file, frames)
    transforms = [registry.get_transform(name) for name in transform_names]
    return TransformChain(transforms)


def _cmd_transform(args: argparse.Namespace) -> int:
    frames_registry = _build_default_frame_registry()
    frames = {name: frames_registry.get_frame(name) for name in frames_registry.list_frames()}

    if args.source not in frames or args.target not in frames:
        raise ValueError("Unknown source/target frame")

    point = Point(np.asarray(args.point, dtype=float), frames[args.source])

    chain = _load_transform_chain(args.transform_file, args.transform_name, frames)
    result = chain.apply(point)

    if result.frame.name != args.target:
        raise ValueError(
            f"Transform chain result frame '{result.frame.name}' does not match requested "
            f"target frame '{args.target}'."
        )
    print(f"Input point ({point.frame.name}): {point.coords.tolist()} {point.frame.units}")
    print(f"Output point ({result.frame.name}): {result.coords.tolist()} {result.frame.units}")

    if args.show_matrix:
        composed = chain.compose()
        print("Composed affine:")
        print(composed.matrix)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tmscoords")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transform_parser = subparsers.add_parser("transform", help="Transform a point")
    transform_parser.add_argument("--point", nargs=3, type=float, required=True)
    transform_parser.add_argument("--from", dest="source", required=True)
    transform_parser.add_argument("--to", dest="target", required=True)
    transform_parser.add_argument("--transform-file", required=True)
    transform_parser.add_argument("--transform-name", nargs="+", required=True)
    transform_parser.add_argument("--show-matrix", action="store_true")
    transform_parser.set_defaults(func=_cmd_transform)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
