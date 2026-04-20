from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import xml.etree.ElementTree as ET

from ..configs import CONFIGS
from ..ir_schema import (
    ApplyExternalWrenchActionIR,
    ObserveActionIR,
    SetDofsPositionActionIR,
    SetDofsVelocityActionIR,
    SetPoseActionIR,
    SetTargetPosActionIR,
    SetTorqueActionIR,
    RigidIR,
    StepActionIR,
)


def validate_program_constraints(
    program: RigidIR,
    *,
    required_shape_kind: str | None = None,
    required_shape_file: str | None = None,
    allowed_shape_kinds: tuple[str, ...] | None = None,
    allowed_articulated_joint_names_by_body: dict[str, tuple[str, ...]] | None = None,
    enforce_articulated_actuator_control: bool = False,
    xml_generation_enabled: bool = False,
    generated_xml_shape_files_by_body: dict[str, str] | None = None,
    mesh_generation_enabled: bool = False,
    generated_mesh_shape_files_by_body: dict[str, str] | None = None,
    failed_generated_mesh_shape_files_by_body: dict[str, str] | None = None,
    target_sim_duration_sec: float | None = None,
    sim_duration_tolerance_sec: float = 0.75,
) -> list[str]:
    errors: list[str] = []

    bodies = program.bodies
    articulated_bodies = [body for body in bodies if body.shape.kind in {"mjcf", "urdf"}]
    mesh_bodies = [body for body in bodies if body.shape.kind == "mesh"]
    deformable_bodies = [body for body in bodies if body.is_deformable]

    if required_shape_kind is not None and not any(body.shape.kind == required_shape_kind for body in bodies):
        actual_kinds = [body.shape.kind for body in bodies]
        errors.append(f"Required at least one body.shape.kind=`{required_shape_kind}`, but got {actual_kinds}.")

    if allowed_shape_kinds is not None:
        allowed_shape_kind_set = set(allowed_shape_kinds)
        invalid_bodies = [body.name for body in bodies if body.shape.kind not in allowed_shape_kind_set]
        if invalid_bodies:
            errors.append(
                f"Allowed body.shape.kind={list(allowed_shape_kinds)}, but invalid bodies are {invalid_bodies}."
            )

    if required_shape_file is not None:
        matched_bodies = [body.name for body in bodies if getattr(body.shape, "file", None) == required_shape_file]
        if not matched_bodies:
            actual_files = [getattr(body.shape, "file", None) for body in bodies]
            errors.append(f"Required at least one body.shape.file=`{required_shape_file}`, but got {actual_files}.")

    if enforce_articulated_actuator_control and articulated_bodies:
        errors.extend(_validate_articulated_actuator_control(program, allowed_articulated_joint_names_by_body))

    if deformable_bodies:
        errors.extend(_validate_deformable_constraints(program))
        errors.extend(_validate_fem_ipc_uipc_sanity(program))

    mjcf_bodies = [body for body in articulated_bodies if body.shape.kind == "mjcf"]
    if xml_generation_enabled and mjcf_bodies:
        for body in mjcf_bodies:
            actual_file = getattr(body.shape, "file", None)
            if not isinstance(actual_file, str) or not actual_file.strip():
                errors.append(f"Articulated body `{body.name}` is missing `shape.file`.")
                continue
            if not Path(actual_file).exists():
                errors.append(
                    f"Articulated body `{body.name}` references missing XML asset `{actual_file}`."
                )
                continue

            if generated_xml_shape_files_by_body:
                expected_file = generated_xml_shape_files_by_body.get(body.name)
                if expected_file is not None and actual_file != expected_file:
                    errors.append(
                        f"Generated XML asset for body `{body.name}` was not attached correctly. "
                        f"Expected body.shape.file=`{expected_file}`, got `{actual_file}`."
                    )

    for body in mesh_bodies:
        actual_file = getattr(body.shape, "file", None)
        if not isinstance(actual_file, str) or not actual_file.strip():
            errors.append(f"Mesh body `{body.name}` is missing `shape.file`.")
            continue
        if not Path(actual_file).exists():
            errors.append(f"Mesh body `{body.name}` references missing mesh asset `{actual_file}`.")
            continue
        if mesh_generation_enabled and generated_mesh_shape_files_by_body:
            expected_file = generated_mesh_shape_files_by_body.get(body.name)
            if expected_file is not None and actual_file != expected_file:
                errors.append(
                    f"Generated mesh asset for body `{body.name}` was not attached correctly. "
                    f"Expected body.shape.file=`{expected_file}`, got `{actual_file}`."
                )
        if mesh_generation_enabled and failed_generated_mesh_shape_files_by_body:
            failed_file = failed_generated_mesh_shape_files_by_body.get(body.name)
            if failed_file is not None and actual_file == failed_file:
                errors.append(
                    f"Generated mesh asset for body `{body.name}` failed manifold validation and cannot be used: "
                    f"`{actual_file}`."
                )

    if target_sim_duration_sec is not None:
        final_step = sum(action.steps for action in program.actions if isinstance(action, StepActionIR))
        sim_duration = final_step * float(program.scene.sim.dt)
        delta = abs(sim_duration - target_sim_duration_sec)
        if delta > sim_duration_tolerance_sec:
            suggested_steps = max(1, int(round(target_sim_duration_sec / float(program.scene.sim.dt))))
            errors.append(
                "Simulation duration mismatch: "
                f"target={target_sim_duration_sec:.3f}s, actual={sim_duration:.3f}s "
                f"(dt={program.scene.sim.dt}, final_step={final_step}). "
                f"Suggested final_step around {suggested_steps}."
            )

    return errors


def _selected_entities(entity: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if entity is None:
        return ()
    if isinstance(entity, str):
        return (entity,)
    return entity


def _validate_articulated_actuator_control(
    program: RigidIR,
    allowed_articulated_joint_names_by_body: dict[str, tuple[str, ...]] | None,
) -> list[str]:
    errors: list[str] = []
    articulated_body_names = {body.name for body in program.bodies if body.shape.kind in {"mjcf", "urdf"}}
    direct_state_action_types = (SetPoseActionIR, SetDofsPositionActionIR, SetDofsVelocityActionIR)
    for idx, action in enumerate(program.actions):
        is_direct_state_action = isinstance(action, direct_state_action_types)
        is_articulated_body = any(entity in articulated_body_names for entity in _selected_entities(getattr(action, "entity", None)))
        if is_direct_state_action and is_articulated_body:
            errors.append(
                f"Action[{idx}] `{action.op}` is forbidden in articulated actuator-control mode. "
                "Use actuator control actions instead."
            )

    articulated_bodies = [body for body in program.bodies if body.name in articulated_body_names]
    articulated_actuators = [actuator for body in articulated_bodies for actuator in body.actuators]
    if not articulated_actuators:
        errors.append(
            "Articulated actuator-control mode requires non-empty actuators on the articulated body in `bodies[].actuators`."
        )

    if not any(
        isinstance(action, (SetTargetPosActionIR, SetTorqueActionIR))
        and getattr(action, "entity", None) in articulated_body_names
        for action in program.actions
    ):
        errors.append(
            "Articulated actuator-control mode requires at least one actuator control action "
            "(`set_target_pos` or `set_torque`)."
        )

    if allowed_articulated_joint_names_by_body is None:
        allowed_articulated_joint_names_by_body = {}

    for body in articulated_bodies:
        allowed_joint_names = allowed_articulated_joint_names_by_body.get(body.name)
        if allowed_joint_names is None and body.shape.kind == "mjcf":
            file_path = getattr(body.shape, "file", None)
            if isinstance(file_path, str) and Path(file_path).exists():
                from ..llm_generator.agents.xml_agent import list_named_joint_names

                allowed_joint_names = list_named_joint_names(file_path)
        if allowed_joint_names is None:
            continue
        allowed_joint_set = set(allowed_joint_names)
        for actuator in body.actuators:
            if actuator.joint_names is None:
                continue
            unknown_joint_names = [name for name in actuator.joint_names if name not in allowed_joint_set]
            if unknown_joint_names:
                errors.append(
                    f"Actuator `{actuator.name}` on body `{body.name}` references unknown joint_names={unknown_joint_names}. "
                    f"Available joint names: {sorted(allowed_joint_set)}."
                )
    return errors


def _validate_deformable_constraints(program: RigidIR) -> list[str]:
    errors: list[str] = []
    deformable_body_names = {body.name for body in program.bodies if body.is_deformable}
    deformable_observe_fields = {"pos", "vel", "bbox_min", "bbox_max", "bbox_size", "vertex_disp_mean", "vertex_disp_max"}
    forbidden_action_types = (
        SetPoseActionIR,
        SetDofsPositionActionIR,
        SetDofsVelocityActionIR,
        ApplyExternalWrenchActionIR,
        SetTargetPosActionIR,
        SetTorqueActionIR,
    )

    for index, action in enumerate(program.actions):
        selected_entities = _selected_entities(getattr(action, "entity", None))
        if not any(entity in deformable_body_names for entity in selected_entities):
            continue
        if isinstance(action, forbidden_action_types):
            errors.append(f"Action[{index}] `{action.op}` is not supported for deformable bodies in v1.")
        if isinstance(action, ObserveActionIR):
            invalid_fields = [field for field in action.fields if field not in deformable_observe_fields]
            if invalid_fields:
                errors.append(
                    f"Action[{index}] observe on deformable bodies cannot use fields {invalid_fields}. "
                    f"Allowed fields: {sorted(deformable_observe_fields)}."
                )
    if CONFIGS.deformable.simulation_backend == "fem_ipc":
        errors.extend(_validate_deformable_articulated_collision_geometry(program))
    return errors


def _validate_deformable_articulated_collision_geometry(program: RigidIR) -> list[str]:
    errors: list[str] = []
    if not any(body.is_deformable for body in program.bodies):
        return errors

    for body in program.bodies:
        if body.shape.kind != "mjcf":
            continue
        xml_file = getattr(body.shape, "file", None)
        if not isinstance(xml_file, str) or not xml_file.strip():
            continue
        path = Path(xml_file)
        if not path.exists():
            continue
        try:
            missing_links = _find_mjcf_bodies_without_collision_geometry(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"Could not inspect articulated collision geometry for `{body.name}` at `{xml_file}`: {type(exc).__name__}: {exc}"
            )
            continue
        if missing_links:
            errors.append(
                "In deformable FEM+IPC scenes, articulated helper bodies are only supported when every MJCF link/body "
                f"has at least one collision-enabled primitive geom. Body `{body.name}` uses `{xml_file}`, but these "
                f"links have no collision geometry: {missing_links}. Prefer rigid primitive/mesh moving boundaries "
                "with scripted motion unless the task explicitly requires an articulated mechanism."
            )
    return errors


def _find_mjcf_bodies_without_collision_geometry(xml_file: Path) -> list[str]:
    root = ET.fromstring(xml_file.read_text(encoding="utf-8"))
    worldbody = root.find("worldbody")
    if worldbody is None:
        return []

    direct_bodies = [child for child in list(worldbody) if child.tag == "body"]
    if len(direct_bodies) != 1:
        return []

    missing: list[str] = []
    for body_elem in direct_bodies[0].iter("body"):
        name = body_elem.attrib.get("name", "<unnamed_body>")
        if not any(_geom_is_collision_enabled(geom) for geom in body_elem.findall("geom")):
            missing.append(name)
    return missing


def _geom_is_collision_enabled(geom_elem: ET.Element) -> bool:
    contype = geom_elem.attrib.get("contype")
    conaffinity = geom_elem.attrib.get("conaffinity")
    if contype is None and conaffinity is None:
        return True
    try:
        contype_value = int(contype) if contype is not None else 1
        conaffinity_value = int(conaffinity) if conaffinity is not None else 1
    except ValueError:
        return True
    return not (contype_value == 0 and conaffinity_value == 0)


def _validate_fem_ipc_uipc_sanity(program: RigidIR) -> list[str]:
    if CONFIGS.deformable.simulation_backend != "fem_ipc":
        return []
    if not any(body.is_deformable for body in program.bodies):
        return []
    repo_root = Path(__file__).resolve().parents[2]
    payload = program.model_dump(mode="json")
    probe = """
import json
import sys

from agent.ir_schema import parse_ir_payload, normalize_ir
from agent.runtime.setup import configure_headless_if_needed, ensure_genesis_initialized, create_runtime_context
import genesis as gs

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
    exact_errors = [
        line
        for line in lines
        if any(
            token in line
            for token in (
                "HalfPlaneVertexDistanceCheck",
                "World is not valid",
                "UIPC_SANITY_BUILD_ERROR",
                "too close (distance <= 0)",
            )
        )
    ]
    if not exact_errors:
        exact_errors = lines[-8:]

    hint = _build_uipc_sanity_hint(exact_errors)
    return [
        "Initial FEM+IPC libuipc sanity check failed: "
        + " | ".join(_strip_ansi(line) for line in exact_errors)
        + f" {hint}"
    ]


def _build_uipc_sanity_hint(exact_errors: list[str]) -> str:
    joined = " | ".join(_strip_ansi(line) for line in exact_errors)
    if any(token in joined for token in ("too close (distance <= 0)", "HalfPlaneVertexDistanceCheck", "World is not valid")):
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


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)
