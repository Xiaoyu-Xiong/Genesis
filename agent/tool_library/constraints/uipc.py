from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from ...configs import CONFIGS
from ...ir_schema.program import RigidIR


def validate_fem_ipc_uipc_sanity(program: RigidIR) -> list[str]:
    if CONFIGS.deformable.simulation_backend != "fem_ipc":
        return []
    if not any(body.is_deformable for body in program.bodies):
        return []
    repo_root = Path(__file__).resolve().parents[3]
    payload = program.model_dump(mode="json")
    probe = """
import json
import sys

from agent.ir_schema.program import parse_ir_payload, normalize_ir
from agent.runtime.setup import configure_headless_if_needed, ensure_genesis_initialized, create_runtime_context
import genesis as gs

def _runtime_object_name_map(program, runtime):
    mapping = {}
    env_count = int(getattr(runtime.scene.sim, "_B", 1))

    deformable_body_names = [body.name for body in program.bodies if body.is_deformable]

    fem_solver = getattr(runtime.scene, "fem_solver", None)
    if fem_solver is not None:
        for i_e, entity in enumerate(getattr(fem_solver, "entities", [])):
            solver_type = "cloth" if getattr(entity, "dim", None) == 2 else "fem"
            body_name = deformable_body_names[i_e] if i_e < len(deformable_body_names) else f"<unknown_deformable_{i_e}>"
            for env_idx in range(env_count):
                mapping[f"{solver_type}_{i_e}_{env_idx}"] = {
                    "body_name": body_name,
                    "entity_type": "deformable",
                }

    rigid_solver = getattr(runtime.scene, "rigid_solver", None)
    if rigid_solver is not None:
        for i_e, entity in enumerate(getattr(rigid_solver, "entities", [])):
            entity_name = (
                getattr(entity, "name", None)
                or getattr(entity, "_name", None)
                or f"<unknown_rigid_entity_{i_e}>"
            )
            for link in getattr(entity, "links", []):
                link_name = getattr(link, "name", None) or f"<unnamed_link_{link.idx}>"
                for env_idx in range(env_count):
                    mapping[f"rigid_link_{link.idx}_{env_idx}"] = {
                        "body_name": entity_name,
                        "link_name": link_name,
                        "entity_type": "rigid",
                    }
                for geom in getattr(link, "geoms", []):
                    for env_idx in range(env_count):
                        mapping[f"rigid_plane_{geom.idx}_{env_idx}"] = {
                            "body_name": entity_name,
                            "link_name": link_name,
                            "geom_idx": int(geom.idx),
                            "entity_type": "rigid_plane",
                        }
    return mapping

payload = json.loads(sys.stdin.read())
program = normalize_ir(parse_ir_payload(payload))
program = program.model_copy(deep=True)
program.scene.show_viewer = False
program.scene.render = None
configure_headless_if_needed(program)
runtime = None
try:
    ensure_genesis_initialized(gs, program)
    runtime = create_runtime_context(gs, program)
    print("UIPC_SANITY_RUNTIME_MAP:" + json.dumps(_runtime_object_name_map(program, runtime), ensure_ascii=False))
    runtime.scene.build()
    print("UIPC_SANITY_OK")
except Exception as exc:
    print(f"UIPC_SANITY_BUILD_ERROR:{type(exc).__name__}:{exc}")
    raise
finally:
    if runtime is not None:
        try:
            runtime.scene.destroy()
        except Exception:
            pass
    try:
        gs.destroy()
    except Exception:
        pass
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        input=_json_dumps(payload),
        text=True,
        capture_output=True,
        cwd=repo_root,
        timeout=120.0,
    )
    if result.returncode == 0:
        return []

    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    runtime_object_map: dict[str, dict[str, Any]] = {}
    filtered_lines: list[str] = []
    for line in lines:
        if line.startswith("UIPC_SANITY_RUNTIME_MAP:"):
            try:
                runtime_object_map = json.loads(line.split(":", 1)[1])
            except json.JSONDecodeError:
                runtime_object_map = {}
            continue
        filtered_lines.append(line)
    lines = filtered_lines

    exact_errors = [
        line
        for line in lines
        if any(
            token in line
            for token in (
                "SimplicialSurfaceIntersectionCheck",
                "HalfPlaneVertexDistanceCheck",
                "World is not valid",
                "UIPC_SANITY_BUILD_ERROR",
                "too close (distance <= 0)",
                "intersects with Geometry(",
                "Intersected mesh has",
                "Intersected mesh is saved at",
                "Create mesh [intersected_mesh",
                "intersected_mesh: <SimplicialComplex>",
            )
        )
    ]
    if not exact_errors:
        exact_errors = lines[-8:]

    has_specific_intersection = any(
        token in line for line in exact_errors for token in ("SimplicialSurfaceIntersectionCheck", "intersects with Geometry(")
    )
    has_specific_clearance = any(
        token in line for line in exact_errors for token in ("HalfPlaneVertexDistanceCheck", "too close (distance <= 0)")
    )
    if has_specific_intersection or has_specific_clearance:
        exact_errors = [
            line
            for line in exact_errors
            if "UIPC_SANITY_BUILD_ERROR:GenesisException:IPC rigid state accessor feature is unavailable" not in line
        ]

    exact_errors = [_annotate_ipc_runtime_objects(line, runtime_object_map) for line in exact_errors]
    intersection_summary = _summarize_ipc_intersections(lines, runtime_object_map)
    if intersection_summary is not None:
        exact_errors.append(intersection_summary)

    hint = _build_uipc_sanity_hint(exact_errors)
    return [
        "Initial FEM+IPC libuipc sanity check failed: "
        + " | ".join(_strip_ansi(line) for line in exact_errors)
        + f" {hint}"
    ]


def _build_uipc_sanity_hint(exact_errors: list[str]) -> str:
    joined = " | ".join(_strip_ansi(line) for line in exact_errors)
    if any(token in joined for token in ("SimplicialSurfaceIntersectionCheck", "intersects with Geometry(")):
        return (
            "This IR fails libuipc's own sanity/build check because bodies start in geometric intersection. "
            "Revise only the overlapping bodies' `initial_pose.pos` (and, if truly necessary, orientation) to create "
            "clear positive separation. Do NOT change shapes, scales, materials, densities, stiffness values, actions, "
            "or unrelated bodies."
        )
    if any(token in joined for token in ("too close (distance <= 0)", "HalfPlaneVertexDistanceCheck")):
        return (
            "This IR fails libuipc's own sanity/build check. Revise only `bodies[*].initial_pose.pos` to increase "
            "clearance between bodies and from the ground. DO NOT change shapes, sizes, scales, materials, densities, "
            "stiffness values, actions, or any other fields."
        )
    if any(token in joined for token in ("Rigid link has no collision geometry", "external_articulation")):
        return (
            "This IR uses an articulated helper structure that FEM+IPC cannot couple in its current form. Prefer "
            "non-articulated rigid primitives or rigid mesh movers with scripted motion unless the task explicitly "
            "requires a robot or articulated mechanism. If an articulated body is truly required, every link/body "
            "must include at least one collision-enabled primitive geom."
        )
    return (
        "This IR fails libuipc's own sanity/build check. Revise the generated structure so it stays within current "
        "FEM+IPC runtime support; do not assume that changing only positions will fix non-penetration-independent errors."
    )


_IPC_OBJECT_RE = re.compile(r"Object\[(?P<name>[^()\]]+)\((?P<object_id>\d+)\)\]")
_IPC_INTERSECTION_RE = re.compile(
    r"Object\[(?P<lhs>[^()\]]+)\((?P<lhs_id>\d+)\)\]\s+intersects with Geometry\(\d+\)\s+in\s+Object\[(?P<rhs>[^()\]]+)\((?P<rhs_id>\d+)\)\]"
)


def _annotate_ipc_runtime_objects(text: str, runtime_object_map: dict[str, dict[str, Any]]) -> str:
    def _replacement(match: re.Match[str]) -> str:
        object_name = match.group("name")
        object_id = match.group("object_id")
        metadata = runtime_object_map.get(object_name)
        if not metadata:
            return match.group(0)
        labels: list[str] = []
        body_name = metadata.get("body_name")
        link_name = metadata.get("link_name")
        if isinstance(body_name, str) and body_name:
            labels.append(f"body={body_name}")
        if isinstance(link_name, str) and link_name:
            labels.append(f"link={link_name}")
        if not labels:
            return match.group(0)
        return f"Object[{object_name}({object_id}) -> {', '.join(labels)}]"

    return _IPC_OBJECT_RE.sub(_replacement, text)


def _summarize_ipc_intersections(
    lines: list[str],
    runtime_object_map: dict[str, dict[str, Any]],
) -> str | None:
    pairs: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    for line in lines:
        match = _IPC_INTERSECTION_RE.search(line)
        if match is None:
            continue
        lhs_name = match.group("lhs")
        rhs_name = match.group("rhs")
        lhs_meta = runtime_object_map.get(lhs_name, {})
        rhs_meta = runtime_object_map.get(rhs_name, {})
        lhs_body = lhs_meta.get("body_name", lhs_name)
        rhs_body = rhs_meta.get("body_name", rhs_name)
        if not isinstance(lhs_body, str) or not isinstance(rhs_body, str):
            continue
        pair = tuple(sorted((lhs_body, rhs_body)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        pairs.append(f"{pair[0]} <-> {pair[1]}")
    if not pairs:
        return None
    return "Mapped intersecting body pairs: " + "; ".join(pairs)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
