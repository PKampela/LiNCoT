"""CLI entry point for tmscoords."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from ..core.chain import TransformChain
from ..core.frames import CoordinateFrame
from ..core.point import Point
from ..registry.frames import FrameRegistry, default_frames
from ..registry.transforms import TransformRegistry
from ..core.session import Session 


class TransformResolutionError(ValueError):
    def __init__(self, message: str, explain_lines: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.explain_lines = explain_lines or []


def _default_transform_file() -> str:
    return str(Path.cwd() / "transforms.json")


def _build_default_frame_registry() -> FrameRegistry:
    registry = FrameRegistry()
    registry.register_many(default_frames())
    return registry


def _load_transform_registry(
    transform_file: Optional[str],
    frames: Mapping[str, CoordinateFrame],
) -> TransformRegistry:
    path = transform_file or _default_transform_file()
    if not Path(path).exists():
        raise FileNotFoundError(f"Transform registry not found: {path}")
    return TransformRegistry.load(path, frames)


def _build_transform_graph(
    registry: TransformRegistry,
) -> Dict[str, List[Tuple[str, str]]]:
    graph: Dict[str, List[Tuple[str, str]]] = {}
    for name in registry.list_transforms():
        transform = registry.get_transform(name)
        graph.setdefault(transform.source.name, []).append((transform.target.name, name))
    for source in graph:
        graph[source].sort(key=lambda item: (item[0], item[1]))
    return graph


def _find_shortest_paths(
    graph: Mapping[str, List[Tuple[str, str]]],
    source: str,
    target: str,
    limit: int = 50,
) -> List[List[str]]:
    queue: deque[Tuple[str, List[str]]] = deque([(source, [])])
    visited_depth: Dict[str, int] = {source: 0}
    shortest_len: Optional[int] = None
    results: List[List[str]] = []

    while queue:
        node, path = queue.popleft()
        depth = len(path)
        if shortest_len is not None and depth > shortest_len:
            break
        if node == target:
            shortest_len = depth
            results.append(path)
            if len(results) >= limit:
                break
            continue
        for next_node, transform_name in graph.get(node, []):
            next_depth = depth + 1
            if shortest_len is not None and next_depth > shortest_len:
                continue
            prev_depth = visited_depth.get(next_node)
            if prev_depth is None or prev_depth >= next_depth:
                visited_depth[next_node] = next_depth
                queue.append((next_node, path + [transform_name]))
    return results


def _chain_steps(
    transform_names: Iterable[str],
    registry: TransformRegistry,
) -> List[Dict[str, str]]:
    steps: List[Dict[str, str]] = []
    for name in transform_names:
        transform = registry.get_transform(name)
        steps.append(
            {
                "source": transform.source.name,
                "target": transform.target.name,
                "name": name,
            }
        )
    return steps


def _path_string(steps: List[Dict[str, str]]) -> str:
    if not steps:
        return ""
    nodes = [steps[0]["source"]] + [step["target"] for step in steps]
    return " -> ".join(nodes)


def _units_summary(steps: List[Dict[str, str]], registry: TransformRegistry) -> str:
    units = []
    for step in steps:
        source_units = registry.get_transform(step["name"]).source.units
        target_units = registry.get_transform(step["name"]).target.units
        if not units:
            units.append(source_units)
        if units[-1] != target_units:
            units.append(target_units)
    if len(set(units)) == 1 and units:
        return f"Units consistent ({units[0]})"
    if units:
        return "Units inconsistent (" + " -> ".join(units) + ")"
    return "Units unavailable"


def _resolve_transform_chain(
    registry: TransformRegistry,
    source: str,
    target: str,
    transform_names: Optional[List[str]],
) -> Tuple[TransformChain, List[Dict[str, str]], List[str]]:
    explain_lines: List[str] = [
        f"Resolving transform from '{source}' to '{target}':",
        "",
        f"- Found {len(registry.list_transforms())} registered transforms",
    ]

    if transform_names:
        try:
            steps = _chain_steps(transform_names, registry)
            chain = TransformChain([registry.get_transform(name) for name in transform_names])
        except (KeyError, ValueError) as exc:
            raise TransformResolutionError(str(exc)) from exc
        if chain.source.name != source or chain.target.name != target:
            raise TransformResolutionError(
                f"Specified transform chain does not match requested source/target: "
                f"{chain.source.name} -> {chain.target.name}"
            )
        explain_lines.extend(
            [
                "- Using user-specified transform chain",
                f"- Selected chain: {_path_string(steps)}",
                "- All frames matched expected source and target",
                f"- {_units_summary(steps, registry)}",
            ]
        )
        return chain, steps, explain_lines

    graph = _build_transform_graph(registry)
    paths = _find_shortest_paths(graph, source, target)
    if not paths:
        outgoing = graph.get(source, [])
        available = ", ".join([name for _, name in outgoing])
        explain_lines.extend(
            [
                f"No valid transform chain found from '{source}' to '{target}'.",
                f"Available outgoing transforms from '{source}': {available or 'none'}",
            ]
        )
        if len(outgoing) == 1:
            explain_lines.append(
                f"Missing transform: {outgoing[0][0]} -> {target}"
            )
        raise TransformResolutionError(
            f"No valid transform chain found from '{source}' to '{target}'.",
            explain_lines,
        )

    selected = paths[0]
    steps = _chain_steps(selected, registry)
    chain = TransformChain([registry.get_transform(name) for name in selected])
    explain_lines.extend(
        [
            f"- Constructed {len(paths)} possible chains",
            f"- Selected shortest valid chain: {_path_string(steps)}",
            "- All frames matched expected source and target",
            f"- {_units_summary(steps, registry)}",
        ]
    )
    return chain, steps, explain_lines


def _emit_output(
    payload: Dict[str, object],
    text_lines: List[str],
    as_json: bool,
) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    for line in text_lines:
        print(line)


def _cmd_transform(args: argparse.Namespace) -> int:
    frames_registry = _build_default_frame_registry()
    frames = {name: frames_registry.get_frame(name) for name in frames_registry.list_frames()}

    if args.source not in frames or args.target not in frames:
        message = "Unknown source/target frame"
        if args.json:
            _emit_output({"error": {"message": message}}, [], True)
        else:
            print(message, file=sys.stderr)
        return 2

    try:
        registry = _load_transform_registry(args.transform_file, frames)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        if args.json:
            _emit_output({"error": {"message": str(exc)}}, [], True)
        else:
            print(str(exc), file=sys.stderr)
        return 2

    point = Point(np.asarray(args.point, dtype=float), frames[args.source])

    try:
        chain, steps, explain_lines = _resolve_transform_chain(
            registry,
            args.source,
            args.target,
            args.transform_name,
        )
    except TransformResolutionError as exc:
        if args.json:
            payload: Dict[str, object] = {"error": {"message": str(exc)}}
            if args.explain and exc.explain_lines:
                payload["explain"] = exc.explain_lines
            _emit_output(payload, [], True)
        else:
            if args.explain and exc.explain_lines:
                print("\n".join(exc.explain_lines), file=sys.stderr)
            else:
                print(str(exc), file=sys.stderr)
        return 2

    result = chain.apply(point)

    if result.frame.name != args.target:
        message = (
            f"Transform chain result frame '{result.frame.name}' does not match requested "
            f"target frame '{args.target}'."
        )
        if args.json:
            _emit_output({"error": {"message": message}}, [], True)
        else:
            print(message, file=sys.stderr)
        return 2

    payload: Dict[str, object] = {
        "input": {
            "frame": point.frame.name,
            "coords": point.coords.tolist(),
            "units": point.frame.units,
        },
        "output": {
            "frame": result.frame.name,
            "coords": result.coords.tolist(),
            "units": result.frame.units,
        },
        "chain": steps,
    }

    text_lines: List[str] = []
    text_lines.append(
        f"Input point ({point.frame.name}): {point.coords.tolist()} {point.frame.units}"
    )
    text_lines.append(
        f"Output point ({result.frame.name}): {result.coords.tolist()} {result.frame.units}"
    )

    if args.show_chain:
        text_lines.append("Transform chain:")
        for step in steps:
            text_lines.append(
                f"  {step['source']} -> {step['target']}   (affine: {step['name']})"
            )

    if args.show_matrix:
        composed = chain.compose()
        payload["composed_matrix"] = composed.matrix.tolist()
        text_lines.append("Composed affine:")
        text_lines.append(str(composed.matrix))

    if args.explain:
        payload["explain"] = explain_lines
        text_lines = explain_lines + [""] + text_lines

    _emit_output(payload, text_lines, args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tmscoords")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transform_parser = subparsers.add_parser("transform", help="Transform a point")
    transform_parser.add_argument("--point", nargs=3, type=float, required=True)
    transform_parser.add_argument("--from", dest="source", required=True)
    transform_parser.add_argument("--to", dest="target", required=True)
    transform_parser.add_argument("--transform-file")
    transform_parser.add_argument("--transform-name", nargs="+")
    transform_parser.add_argument("--show-matrix", action="store_true")
    transform_parser.add_argument("--show-chain", action="store_true")
    transform_parser.add_argument("--explain", action="store_true")
    transform_parser.add_argument("--json", action="store_true")
    transform_parser.set_defaults(func=_cmd_transform)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    session = Session(subject_id="default", description="Example session" )
    return args.func(args, session)


if __name__ == "__main__":
    raise SystemExit(main())
