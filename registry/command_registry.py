"""Command registry shared by terminal CLI and GUI console."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core.import_service import import_image as import_mri_image
from core.import_service import import_surface as import_surface_file
from core.chain import TransformChain
from core.frames import CoordinateFrame
from core.registration import register_images, registration_report_lines
from core.point import Point
from core.session import Session
from registry.frame_registry import FrameRegistry
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
    frames = session.frames.list_frames()
    return CommandResult(
        message="Frames: " + (", ".join(frames) if frames else "none"),
        data={"frames": frames},
    )


def _cmd_transform_list(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    _expect_arg_count("transform.list", args, 0)
    transforms = session.transforms.list_transforms()
    return CommandResult(
        message="Transforms: " + (", ".join(transforms) if transforms else "none"),
        data={"transforms": transforms},
    )


def _cmd_surface_list(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    _expect_arg_count("surface.list", args, 0)
    surfaces = session.surfaces.list_surfaces()
    return CommandResult(
        message="Surfaces: " + (", ".join(surfaces) if surfaces else "none"),
        data={"surfaces": surfaces},
    )


def _cmd_surface_import(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    _expect_arg_count("surface.import", args, 1)
    path = args[0]
    frame_name = kwargs.get("frame")
    surface_name = kwargs.get("name")

    try:
        surface, info_msg = import_surface_file(
            session,
            path,
            frame_name=frame_name,
            surface_name=surface_name,
        )
    except Exception as exc:
        raise CommandExecutionError(f"Failed to import surface: {exc}") from exc

    first_line = info_msg.splitlines()[0] if info_msg else f"Imported surface: {Path(path).stem}"
    resolved_name = first_line.split(":", 1)[-1].strip() if ":" in first_line else Path(path).stem

    payload: Dict[str, object] = {
        "surface": {
            "name": resolved_name,
            "path": path,
            "frame": surface.frame.name,
            "vertices": int(surface.vertices.shape[0]),
            "faces": int(surface.faces.shape[0]),
        }
    }

    return CommandResult(
        message=info_msg,
        data=payload,
        output_format="json" if kwargs.get("json") else "text",
    )


def _cmd_surface_import(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    _expect_arg_count("surface.import", args, 1)
    path = args[0]
    frame_name = kwargs.get("frame")
    surface_name = kwargs.get("name")
    try:
        surface, info_msg = import_surface_file(
            session,
            path,
            frame_name=frame_name,
            surface_name=surface_name,
        )
    except Exception as exc:
        raise CommandExecutionError(f"Failed to import surface: {exc}") from exc

    payload: Dict[str, object] = {
        "surface": {
            "name": surface_name or Path(path).name.replace(".", "_"),
            "path": path,
            "frame": surface.frame.name,
            "vertices": int(surface.vertices.shape[0]),
            "faces": int(surface.faces.shape[0]),
        }
    }

    return CommandResult(
        message=info_msg,
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
            "frame": image.frame.name,
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
        "subject_id": session.subject_id,
        "description": session.description,
        "frames": session.frames.list_frames(),
        "points": session.list_points(),
        "images": session.list_images(),
        "transforms": session.transforms.list_transforms(),
        "surfaces": session.surfaces.list_surfaces(),
    }
    return CommandResult(
        message=(
            f"Session summary: {len(summary['frames'])} frames, "
            f"{len(summary['points'])} points, {len(summary['images'])} images"
        ),
        data=summary,
    )


def _unique_transform_name(session: Session, preferred: str) -> str:
    existing = session.transforms.list_transforms()
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
    for name in registry.list_transforms():
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
        f"- Found {len(registry.list_transforms())} registered transforms",
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
        available = ", ".join([name for _, name in outgoing])
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


def _cmd_transform(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    """Transform either an existing point or a raw coordinate triple.

    Accepted forms:
        transform <point> <target>
        transform <point> <source> <target>
        transform <source> <target> <x> <y> <z>
    """
    transform_registry = session.transforms
    if not transform_registry.list_transforms():
        raise CommandExecutionError(
            "No transforms are registered in this session. Import a transform or MRI volume first."
        )

    chain_spec = kwargs.get("chain")
    transform_names = None
    if chain_spec:
        transform_names = chain_spec if isinstance(chain_spec, list) else [chain_spec]

    if len(args) in {2, 3}:
        point_name = args[0]
        if len(args) == 2:
            target_frame = args[1]
            try:
                point = session.get_point(point_name)
            except KeyError as exc:
                raise CommandExecutionError(f"Unknown point: {point_name}") from exc
            source_frame = point.frame.name
        else:
            source_frame, target_frame = args[1], args[2]
            try:
                point = session.get_point(point_name)
            except KeyError as exc:
                raise CommandExecutionError(f"Unknown point: {point_name}") from exc
            if point.frame.name != source_frame:
                raise CommandExecutionError(
                    f"Point '{point_name}' is in frame '{point.frame.name}', not '{source_frame}'"
                )

        if source_frame == target_frame:
            updated_point = point
            steps: List[Dict[str, object]] = []
            explain_lines = [
                f"Resolving transform from '{source_frame}' to '{target_frame}':",
                "",
                "- Source and target are the same frame; no transform needed",
            ]
            payload: Dict[str, object] = {
                "point": {
                    "name": point_name,
                    "frame": updated_point.frame.name,
                    "coords": updated_point.coords.tolist(),
                },
                "transform": {"input": None, "output": None, "chain": []},
            }
            text_lines = [
                f"Point '{point_name}' is already in frame '{target_frame}'",
                f"Output point ({updated_point.frame.name}): {updated_point.coords.tolist()} {updated_point.frame.units}",
            ]
            if kwargs.get("show_chain"):
                text_lines.append("Transform chain:")
                text_lines.append("  <none>")
            if kwargs.get("explain"):
                payload["transform"]["explain"] = explain_lines
                text_lines = explain_lines + [""] + text_lines
            if kwargs.get("show_matrix"):
                payload["transform"]["composed_matrix"] = np.eye(4).tolist()
            session.add_point(point_name, updated_point)
            return CommandResult(
                message="\n".join(text_lines),
                data=payload,
                output_format="json" if kwargs.get("json") else "text",
            )

        if not transform_registry.list_transforms():
            raise CommandExecutionError(
                "No transforms are registered in this session. Import a transform or MRI volume first."
            )

        try:
            chain, steps, explain_lines = _resolve_transform_chain(
                transform_registry,
                source_frame,
                target_frame,
                transform_names,
            )
        except TransformResolutionError as exc:
            lines = []
            if kwargs.get("explain") and exc.explain_lines:
                lines.extend(exc.explain_lines)
            lines.append(str(exc))
            return CommandResult(
                message="\n".join(lines),
                data={"error": str(exc)},
                output_format="json" if kwargs.get("json") else "text"
            )

        result = chain.apply(point)
        if result.frame.name != target_frame:
            raise CommandExecutionError(
                f"Transform chain result frame '{result.frame.name}' does not match "
                f"requested target frame '{target_frame}'."
            )

        session.add_point(point_name, result)

        payload = {
            "point": {
                "name": point_name,
                "frame": result.frame.name,
                "coords": result.coords.tolist(),
            },
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

        text_lines = [
            f"Transformed point '{point_name}' from '{source_frame}' to '{target_frame}'",
            f"Input point ({point.frame.name}): {point.coords.tolist()} {point.frame.units}",
            f"Output point ({result.frame.name}): {result.coords.tolist()} {result.frame.units}",
        ]

        if kwargs.get("show_chain"):
            text_lines.append("Transform chain:")
            for step in steps:
                name = f"{step['name']} (inverse)" if step.get("inverted") else str(step["name"])
                text_lines.append(f"  {step['source']} -> {step['target']}   (affine: {name})")

        if kwargs.get("show_matrix"):
            composed = chain.compose()
            payload["composed_matrix"] = composed.matrix.tolist()
            text_lines.append("Composed affine:")
            text_lines.append(str(composed.matrix))

        if kwargs.get("explain"):
            payload["explain"] = explain_lines
            text_lines = explain_lines + [""] + text_lines

        return CommandResult(
            message="\n".join(text_lines),
            data=payload,
            output_format="json" if kwargs.get("json") else "text",
        )

    if len(args) != 5:
        raise CommandExecutionError(f"transform expects 2, 3, or 5 arguments, got {len(args)}")

    source_frame, target_frame, x_str, y_str, z_str = args

    frame_registry = session.frames
    frames = {name: frame_registry.get_frame(name) for name in frame_registry.list_frames()}

    if source_frame not in frames or target_frame not in frames:
        raise CommandExecutionError(f"Unknown frame: source='{source_frame}' or target='{target_frame}'")

    try:
        coords = [float(x_str), float(y_str), float(z_str)]
    except ValueError as exc:
        raise CommandExecutionError(f"Invalid coordinates: {exc}") from exc

    point = Point(np.asarray(coords, dtype=float), frames[source_frame])

    try:
        chain, steps, explain_lines = _resolve_transform_chain(
            transform_registry,
            source_frame,
            target_frame,
            transform_names,
        )
    except TransformResolutionError as exc:
        lines = []
        if kwargs.get("explain") and exc.explain_lines:
            lines.extend(exc.explain_lines)
        lines.append(str(exc))
        return CommandResult(
            message="\n".join(lines),
            data={"error": str(exc)},
            output_format="json" if kwargs.get("json") else "text"
        )

    result = chain.apply(point)

    if result.frame.name != target_frame:
        raise CommandExecutionError(
            f"Transform chain result frame '{result.frame.name}' does not match "
            f"requested target frame '{target_frame}'."
        )

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

    if kwargs.get("show_chain"):
        text_lines.append("Transform chain:")
        for step in steps:
            name = f"{step['name']} (inverse)" if step.get("inverted") else str(step["name"])
            text_lines.append(
                f"  {step['source']} -> {step['target']}   (affine: {name})"
            )

    if kwargs.get("show_matrix"):
        composed = chain.compose()
        payload["composed_matrix"] = composed.matrix.tolist()
        text_lines.append("Composed affine:")
        text_lines.append(str(composed.matrix))

    if kwargs.get("explain"):
        payload["explain"] = explain_lines
        text_lines = explain_lines + [""] + text_lines

    message = "\n".join(text_lines)

    return CommandResult(
        message=message,
        data=payload,
        output_format="json" if kwargs.get("json") else "text"
    )


def _cmd_volume_load(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    """Load a NIfTI volume into the session using MRI import routing.

    This now delegates to the import service so MRI volumes get subject-specific
    frames and affine transforms instead of relying on shared default frames.
    """
    if len(args) != 1:
        raise CommandExecutionError(f"volume load expects 1 argument, got {len(args)}")
    
    file_path = args[0]
    try:
        image, info_msg = import_mri_image(session, file_path)
    except Exception as exc:
        raise CommandExecutionError(f"Failed to import MRI image: {exc}") from exc

    first_line = info_msg.splitlines()[0] if info_msg else "Imported MRI image"
    image_name = first_line.split(":", 1)[-1].strip() if ":" in first_line else Path(file_path).stem

    payload: Dict[str, object] = {
        "image": {
            "name": image_name,
            "path": file_path,
            "shape": tuple(int(value) for value in image.shape),
            "frame": image.frame.name,
            "affine_shape": image.affine.shape,
        }
    }

    message = info_msg

    return CommandResult(
        message=message,
        data=payload,
        output_format="json" if kwargs.get("json") else "text"
    )


def _cmd_volume_import(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
    return _cmd_volume_load(session, args, kwargs)


def register_default_commands(registry: CommandRegistry) -> None:
    registry.register("point.add", _cmd_point_add, "point add <name> <x> <y> <z> <frame>")
    registry.register("point.list", _cmd_point_list, "point list")
    registry.register("frame.list", _cmd_frame_list, "frame list")
    registry.register("transform", _cmd_transform, "transform <point> <target> [--chain name1 name2] [--show-chain] [--show-matrix] [--explain] [--json]")
    registry.register("transform.list", _cmd_transform_list, "transform list")
    registry.register("volume.load", _cmd_volume_load, "volume load <path> [--json]")
    registry.register("volume.import", _cmd_volume_import, "volume import <path> [--json]")
    registry.register("surface.import", _cmd_surface_import, "surface import <path> [--name name] [--frame frame] [--json]")
    registry.register("surface.list", _cmd_surface_list, "surface list")
    registry.register("view.surface", _cmd_view_surface, "view surface <name> [--json]")
    registry.register("view.volume", _cmd_view_volume, "view volume <name> [--json]")
    registry.register("register", _cmd_register, "register <moving> <reference> [--quality fast|standard|accurate] [--name name] [--report] [--json]")
    registry.register("session.summary", _cmd_session_summary, "session summary")

    def _help_wrapper(session: Session, args: Sequence[str], kwargs: Dict[str, Any]) -> CommandResult:
        return _cmd_help(session, args, kwargs, registry)

    registry.register("help", _help_wrapper, "help")
