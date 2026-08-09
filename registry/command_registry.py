"""Command registry shared by terminal CLI and GUI console."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core.import_service import import_image as import_mri_image
from core.chain import TransformChain
from core.registration import register_images, registration_report_lines
from core.point import Point
from core.transform import Transform
from core.session import Session
from registry.transform_registry import TransformRegistry



@dataclass(frozen=True)
class CommandResult:
    message: str
    data: dict | None = None
    output_format: str = "text"  # "text" or "json"


CommandHandler = Callable[[Session, Sequence[str], Dict[str, Any]], CommandResult]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    handler: CommandHandler
    help_text: str


@dataclass
class CommandRegistry:
    _commands: Dict[str, CommandSpec] = field(default_factory=dict)

    def register(self, name: str, handler: CommandHandler, help_text: str) -> None:
        key = name.lower()
        if key in self._commands:
            raise ValueError(f"Command '{name}' already registered")
        self._commands[key] = CommandSpec(name=key, handler=handler, help_text=help_text)

    def get(self, name: str) -> CommandSpec:
        key = name.lower()
        try:
            return self._commands[key]
        except KeyError as exc:
            raise KeyError(f"Unknown command '{name}'") from exc

    def execute(self, session: Session, name: str, args: Sequence[str], kwargs: Dict[str, Any] | None = None) -> CommandResult:
        spec = self.get(name)
        return spec.handler(session, args, kwargs or {})

    def list_commands(self) -> List[CommandSpec]:
        return [self._commands[k] for k in sorted(self._commands.keys())]


class CommandExecutionError(ValueError):
    """Raised when a known command receives invalid arguments."""


def _expect_arg_count(command_name: str, args: Sequence[str], expected: int) -> None:
    if len(args) != expected:
        raise CommandExecutionError(
            f"{command_name} expects {expected} arguments, got {len(args)}"
        )


def _cmd_point_add(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    _expect_arg_count("point.add", args, 5)
    name, x, y, z, frame_name = args
    try:
        frame = session.get_frame(frame_name)
    except KeyError as exc:
        raise CommandExecutionError(f"Unknown frame: {frame_name}") from exc
    try:
        coords = [float(x), float(y), float(z)]
    except ValueError as exc:
        raise CommandExecutionError(f"Invalid coordinates: {exc}") from exc
    point = Point(np.asarray(coords, dtype=float), frame)
    session.add_point(name, point)
    return CommandResult(
        message=f"Added point '{name}' in frame '{frame_name}'",
        data={"name": name, "coords": point.coords.tolist(), "frame": frame_name},
    )


def _cmd_point_list(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    _expect_arg_count("point.list", args, 0)
    names = session.list_points()
    if not names:
        return CommandResult("No points in session", data={"points": []})
    return CommandResult(
        message="Points: " + ", ".join(names),
        data={
            "points": [
                {
                    "name": name,
                    "coords": session.get_point(name).coords.tolist(),
                    "frame": session.get_point(name).frame.name,
                }
                for name in names
            ]
        },
    )


def _cmd_frame_list(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    _expect_arg_count("frame.list", args, 0)
    frames = session.frames.names()
    return CommandResult(
        message="Frames: " + (", ".join(frames) if frames else "none"),
        data={"frames": frames},
    )


def _cmd_transform_list(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    _expect_arg_count("transform.list", args, 0)
    transforms = session.transforms.names()
    return CommandResult(
        message="Transforms: " + (", ".join(transforms) if transforms else "none"),
        data={"transforms": transforms},
    )


def _cmd_surface_list(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    _expect_arg_count("surface.list", args, 0)
    surfaces = session.surfaces.names_all()
    return CommandResult(
        message="Surfaces: " + (", ".join(surfaces) if surfaces else "none"),
        data={"surfaces": surfaces},
    )


def _cmd_surface_import(
    session: Session,
    args: Sequence[str],
    kwargs: Dict[str, Any],
) -> CommandResult:

    if not args:
        raise CommandExecutionError(
            "surface import requires at least one file"
        )

    paths = [Path(p) for p in args]

    names = kwargs.get("name")
    frames = kwargs.get("frame")

    if isinstance(names, str):
        names = [names]

    if isinstance(frames, str):
        frames = [frames]

    if names and len(names) != len(paths):
        raise CommandExecutionError(
            f"Received {len(names)} names for {len(paths)} surfaces"
        )

    if frames and len(frames) != len(paths):
        raise CommandExecutionError(
            f"Received {len(frames)} frames for {len(paths)} surfaces"
        )

    try:
        results = session.import_surfaces(
            paths,
            frame_names=frames,
            surface_names=names,
        )

    except Exception as exc:
        raise CommandExecutionError(
            f"Failed to import surfaces: {exc}"
        ) from exc


    payload = {
        "surfaces": [
            {
                "name": registered_name,
                "path": str(path),
                "frame": surface.frame.name,
                "vertices": int(surface.vertices.shape[0]),
                "faces": int(surface.faces.shape[0]),
            }
            for (surface, registered_name, _), path
            in zip(results, paths)
        ]
    }

    message = "\n\n".join(
        info
        for _, _, info in results
    )

    return CommandResult(
        message=message,
        data=payload,
        output_format="json" if kwargs.get("json") else "text",
    )


def _cmd_view_surface(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    _expect_arg_count("view.surface", args, 1)
    surface_name = args[0]
    try:
        surface = session.get_surface(surface_name)
    except KeyError as exc:
        raise CommandExecutionError(f"Unknown surface: {surface_name}") from exc

    payload = {
        "viewer": {"type": "surface", "name": surface_name},
        "surface": {
            "name": surface_name,
            "frame": surface.frame.name,
            "vertices": int(surface.vertices.shape[0]),
            "faces": int(surface.faces.shape[0]),
        },
    }
    return CommandResult(
        message=f"Opened surface viewer for '{surface_name}'",
        data=payload,
        output_format="json" if kwargs.get("json") else "text",
    )


def _cmd_view_volume(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    _expect_arg_count("view.volume", args, 1)
    image_name = args[0]
    try:
        image = session.get_image(image_name)
    except KeyError as exc:
        raise CommandExecutionError(f"Unknown volume: {image_name}") from exc

    payload = {
        "viewer": {"type": "volume", "name": image_name},
        "volume": {
            "name": image_name,
            "voxel_frame": image.voxel_frame.name,
            "world_frame": image.world_frame.name,
            "shape": tuple(int(value) for value in image.shape),
            "dtype": str(image.data.dtype),
        },
    }
    return CommandResult(
        message=f"Opened volume viewer for '{image_name}'",
        data=payload,
        output_format="json" if kwargs.get("json") else "text",
    )


def _cmd_session_summary(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    _expect_arg_count("session.summary", args, 0)
    summary = {
        "subject_id": session.subject.subject_id,
        "description": session.project.description,
        "frames": session.frames.names(),
        "points": session.list_points(),
        "images": session.list_images(),
        "transforms": session.transforms.names(),
        "surfaces": session.surfaces.names_all(),
    }
    return CommandResult(
        message=(
            f"Session summary: {len(summary['frames'])} frames, "
            f"{len(summary['points'])} points, {len(summary['images'])} images"
        ),
        data=summary,
    )


def _unique_transform_name(session: Session, preferred: str) -> str:
    existing = session.transforms.names()
    if preferred not in existing:
        return preferred
    index = 1
    while f"{preferred}_{index}" in existing:
        index += 1
    return f"{preferred}_{index}"


def _cmd_register(
    session: Session,
    args: Sequence[str],
    kwargs: Dict[str, Any],
) -> CommandResult:

    command_name = "register"
    _expect_arg_count(command_name, args, 2)

    moving_name, reference_name = args

    try:
        moving_image = session.get_image(moving_name)
    except KeyError as exc:
        raise CommandExecutionError(
            f"Unknown moving image: {moving_name}"
        ) from exc

    try:
        reference_image = session.get_image(reference_name)
    except KeyError as exc:
        raise CommandExecutionError(
            f"Unknown reference image: {reference_name}"
        ) from exc

    quality = str(kwargs.get("quality", "standard"))

    explicit_name = kwargs.get("name")
    transform_name = str(explicit_name).strip() if explicit_name else ""

    if not transform_name:
        transform_name = _unique_transform_name(
            session,
            f"register_{moving_name}_to_{reference_name}",
        )

    try:
        transform, report = register_images(
            moving_image,
            reference_image,
            quality=quality,
        )

    except Exception as exc:
        raise CommandExecutionError(
            f"Failed to register images: {exc}"
        ) from exc

    session.add_transform(transform_name, transform)

    report_lines = [
        f"Registered transform '{transform_name}'",
        f"  Moving: {moving_name}",
        f"  Reference: {reference_name}",
        f"  Quality: {report.quality}",
    ] + [
        f"  {line}"
        for line in registration_report_lines(report)
    ]

    report_text = "\n".join(report_lines)

    payload = {
        "transform": {
            "name": transform_name,
            "source": transform.source.name,
            "target": transform.target.name,
            "matrix": transform.matrix.tolist(),
        },
        "registration": {
            "quality": report.quality,
            "iterations": report.iterations,
            "similarity": report.similarity,
            "translation_mm": report.translation_mm,
            "rotation_deg": report.rotation_deg,
        },
        "report": report_text,
    }

    message = report_lines[0]
    if kwargs.get("report"):
        message = report_text

    return CommandResult(
        message=message,
        data=payload,
        output_format="json" if kwargs.get("json") else "text",
    )


def _cmd_help(session: Session, args: Sequence[str], kwargs: Dict[str, Any], registry: CommandRegistry) -> CommandResult:
    _expect_arg_count("help", args, 0)
    lines = [f"{spec.name}: {spec.help_text}" for spec in registry.list_commands()]
    return CommandResult("Available commands:\n" + "\n".join(lines))


# Transform command helpers

class TransformResolutionError(ValueError):
    """Error during transform chain resolution."""
    def __init__(self, message: str, explain_lines: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.explain_lines = explain_lines or []


@dataclass(frozen=True)
class _ResolvedStep:
    source: str
    target: str
    name: str
    inverted: bool = False


def _build_transform_graph(
    registry: TransformRegistry,
) -> Dict[str, List[Tuple[str, str, bool]]]:
    graph: Dict[str, List[Tuple[str, str, bool]]] = {}
    for name in registry.names():
        transform = registry.get_transform(name)
        graph.setdefault(transform.source.name, []).append((transform.target.name, name, False))
        graph.setdefault(transform.target.name, []).append((transform.source.name, name, True))
    for source in graph:
        graph[source].sort(key=lambda item: (item[0], item[1], item[2]))
    return graph


def _find_shortest_paths(
    graph: Mapping[str, List[Tuple[str, str, bool]]],
    source: str,
    target: str,
    limit: int = 50,
) -> List[List[Tuple[str, bool]]]:
    queue: deque[Tuple[str, List[Tuple[str, bool]]]] = deque([(source, [])])
    visited_depth: Dict[str, int] = {source: 0}
    shortest_len: Optional[int] = None
    results: List[List[Tuple[str, bool]]] = []

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
        for next_node, transform_name, inverted in graph.get(node, []):
            next_depth = depth + 1
            if shortest_len is not None and next_depth > shortest_len:
                continue
            prev_depth = visited_depth.get(next_node)
            if prev_depth is None or prev_depth >= next_depth:
                visited_depth[next_node] = next_depth
                queue.append((next_node, path + [(transform_name, inverted)]))
    return results


def _chain_steps(
    transform_names: Iterable[Tuple[str, bool]],
    registry: TransformRegistry,
) -> Tuple[List[Transform], List[Dict[str, object]]]:
    transforms: List[Transform] = []
    steps: List[Dict[str, object]] = []
    for name, inverted in transform_names:
        transform = registry.get_transform(name)
        if inverted:
            transform = transform.invert()
        transforms.append(transform)
        steps.append(
            {
                "source": transform.source.name,
                "target": transform.target.name,
                "name": name,
                "inverted": inverted,
            }
        )
    return transforms, steps


def _path_string(steps: List[Dict[str, object]]) -> str:
    if not steps:
        return ""
    nodes = [str(steps[0]["source"])] + [str(step["target"]) for step in steps]
    return " -> ".join(nodes)


def _units_summary(steps: List[Dict[str, object]], registry: TransformRegistry) -> str:
    units = []
    for step in steps:
        source_units = registry.get_transform(str(step["name"])).source.units
        target_units = registry.get_transform(str(step["name"])).target.units
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
) -> Tuple[TransformChain, List[Dict[str, object]], List[str]]:
    explain_lines: List[str] = [
        f"Resolving transform from '{source}' to '{target}':",
        "",
        f"- Found {len(registry)} registered transforms",
    ]

    if transform_names:
        try:
            chain_transforms, steps = _chain_steps([(name, False) for name in transform_names], registry)
            chain = TransformChain(chain_transforms)
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
        available = ", ".join([name for _, name, _ in outgoing])
        explain_lines.extend(
            [
                f"No valid transform chain found from '{source}' to '{target}'.",
                f"Available outgoing transforms from '{source}': {available or 'none'}",
            ]
        )
        if len(outgoing) == 1:
            explain_lines.append(f"Missing transform: {outgoing[0][0]} -> {target}")
        raise TransformResolutionError(
            f"No valid transform chain found from '{source}' to '{target}'.",
            explain_lines,
        )

    selected = paths[0]
    chain_transforms, steps = _chain_steps(selected, registry)
    chain = TransformChain(chain_transforms)
    explain_lines.extend(
        [
            f"- Constructed {len(paths)} possible chains",
            f"- Selected shortest valid chain: {_path_string(steps)}",
            "- All frames matched expected source and target",
            f"- {_units_summary(steps, registry)}",
        ]
    )
    return chain, steps, explain_lines


def _get_point_for_transform(
    session: Session,
    point_name: str,
    source_frame: str | None = None,
) -> Point:
    """Retrieve a point and validate its source frame."""

    try:
        point = session.get_point(point_name)
    except KeyError as exc:
        raise CommandExecutionError(
            f"Unknown point: {point_name}"
        ) from exc

    if source_frame is not None:
        if point.frame.name != source_frame:
            raise CommandExecutionError(
                f"Point '{point_name}' is in frame "
                f"'{point.frame.name}', not '{source_frame}'"
            )

    return point

def _apply_point_transform(
    session: Session,
    point: Point,
    target_frame: str,
    transform_names: list[str] | None = None,
):
    """Apply a transform chain to a point."""

    if point.frame.name == target_frame:
        return point, [], [
            f"Source and target are identical: '{target_frame}'"
        ]

    try:
        chain, steps, explain_lines = _resolve_transform_chain(
            session.transforms,
            point.frame.name,
            target_frame,
            transform_names,
        )

    except TransformResolutionError as exc:
        raise CommandExecutionError(
            str(exc)
        ) from exc

    result = chain.apply(point)

    if result.frame.name != target_frame:
        raise CommandExecutionError(
            f"Transform produced frame '{result.frame.name}', "
            f"expected '{target_frame}'"
        )

    return result, steps, explain_lines

def _resolve_output_point_name(
    session: Session,
    requested_name: str | None,
    original_name: str,
    target_frame: str,
) -> str:

    if requested_name:
        return requested_name

    return f"{original_name}_{target_frame}"

def _point_payload(
    point_name: str,
    point: Point,
):
    return {
        "name": point_name,
        "frame": point.frame.name,
        "coords": point.coords.tolist(),
        "units": point.frame.units,
    }

def _cmd_transform(
    session: Session,
    args: Sequence[str],
    kwargs: Dict[str, Any],
) -> CommandResult:

    if len(args) not in {2, 3}:
        raise CommandExecutionError(
            "transform expects: "
            "transform <point> <target> "
            "or transform <point> <source> <target>"
        )

    chain_spec = kwargs.get("chain")

    transform_names = (
        chain_spec
        if isinstance(chain_spec, list)
        else [chain_spec]
        if chain_spec
        else None
    )

    point_name = args[0]

    if len(args) == 2:
        target_frame = args[1]
        point = _get_point_for_transform(
            session,
            point_name,
        )

    else:
        source_frame = args[1]
        target_frame = args[2]

        point = _get_point_for_transform(
            session,
            point_name,
            source_frame,
        )


    result, steps, explain_lines = _apply_point_transform(
        session,
        point,
        target_frame,
        transform_names,
    )


    output_name = _resolve_output_point_name(
        session,
        kwargs.get("name"),
        point_name,
        target_frame,
    )


    if output_name in session.list_points():
        raise CommandExecutionError(
            f"Point '{output_name}' already exists"
        )


    session.create_transformed_point(
        output_name,
        result,
    )


    payload = {
        "input": _point_payload(
            point_name,
            point,
        ),
        "output": _point_payload(
            output_name,
            result,
        ),
        "chain": steps,
    }


    text = [
        f"Created transformed point '{output_name}'",
        f"Input frame: {point.frame.name}",
        f"Output frame: {result.frame.name}",
        f"Coordinates: {result.coords.tolist()}",
    ]


    if kwargs.get("show_chain"):
        text.append("Transform chain:")
        for step in steps:
            name = (
                f"{step['name']} (inverse)"
                if step.get("inverted")
                else str(step["name"])
            )

            text.append(
                f"  {step['source']} -> {step['target']} ({name})"
            )


    if kwargs.get("explain"):
        payload["explain"] = explain_lines
        text = explain_lines + [""] + text


    return CommandResult(
        message="\n".join(text),
        data=payload,
        output_format="json" if kwargs.get("json") else "text",
    )


def _cmd_volume_import(
    session: Session,
    args: Sequence[str],
    kwargs: Dict[str, Any],
) -> CommandResult:

    if not args:
        raise CommandExecutionError(
            "volume import requires at least one file"
        )

    paths = [Path(p) for p in args]

    try:
        results = session.import_images(paths)

    except Exception as exc:
        raise CommandExecutionError(
            f"Failed to import volumes: {exc}"
        ) from exc


    payload = {
        "images": [
            {
                "name": info.split(":",1)[-1].strip(),
                "path": str(path),
                "voxel_frame": image.voxel_frame.name,
                "world_frame": image.world_frame.name,
                "shape": tuple(int(v) for v in image.shape),
                "affine_shape": image.affine.shape,
            }
            for (image, info), path
            in zip(results, paths)
        ]
    }

    message = "\n\n".join(
        info
        for _, info in results
    )

    return CommandResult(
        message=message,
        data=payload,
        output_format="json" if kwargs.get("json") else "text",
    )

def _cmd_transform_import(
    session: Session,
    args: Sequence[str],
    kwargs: Dict[str, Any],
) -> CommandResult:

    if not args:
        raise CommandExecutionError(
            "transform import requires at least one file"
        )

    paths = [Path(p) for p in args]

    source_frame = kwargs.get("source_frame")
    target_frame = kwargs.get("target_frame")

    try:
        results = session.import_transforms(
            paths,
            source_frame_name=source_frame,
            target_frame_name=target_frame,
        )

    except Exception as exc:
        raise CommandExecutionError(
            f"Failed to import transforms: {exc}"
        ) from exc


    payload = {
        "transforms": [
            {
                "path": str(path),
                "source": transform.source.name,
                "target": transform.target.name,
            }
            for (transform, _), path
            in zip(results, paths)
        ]
    }


    message = "\n\n".join(
        info
        for _, info in results
    )

    return CommandResult(
        message=message,
        data=payload,
        output_format="json" if kwargs.get("json") else "text",
    )


def register_default_commands(registry: CommandRegistry) -> None:
    registry.register("point.add", _cmd_point_add, "point add <name> <x> <y> <z> <frame>")
    registry.register("point.list", _cmd_point_list, "point list")
    registry.register("frame.list", _cmd_frame_list, "frame list")
    registry.register("transform.import", _cmd_transform_import, "transform.import <path1> <path2> ... [--source-frame frame] [--target-frame frame] [--json]")
    registry.register("transform", _cmd_transform, "transform <point> <target> [--chain name1 name2] [--show-chain] [--show-matrix] [--explain] [--json]")
    registry.register("transform.list", _cmd_transform_list, "transform list")
    registry.register("volume.import", _cmd_volume_import, "volume import <path1> <path2> ... [--json]")
    registry.register("surface.import", _cmd_surface_import, "surface import <path1> <path2> ... [--name name1 name2 ...] [--frame frame1 frame2 ...] [--json]")
    registry.register("surface.list", _cmd_surface_list, "surface list")
    registry.register("view.surface", _cmd_view_surface, "view surface <name> [--json]")
    registry.register("view.volume", _cmd_view_volume, "view volume <name> [--json]")
    registry.register("register", _cmd_register, "register <moving> <reference> [--quality fast|standard|accurate] [--name name] [--report] [--json]")
    registry.register("session.summary", _cmd_session_summary, "session summary")

    def _help_wrapper(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
        return _cmd_help(session, args, kwargs, registry)

    registry.register("help", _help_wrapper, "help")
