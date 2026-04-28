from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from ...ir_schema.actions import (
    SetDofsPositionActionIR,
    SetDofsVelocityActionIR,
    SetPoseActionIR,
    SetTargetPosActionIR,
    SetTorqueActionIR,
)
from ...ir_schema.program import RigidIR


def selected_entities(entity: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if entity is None:
        return ()
    if isinstance(entity, str):
        return (entity,)
    return entity


def validate_articulated_actuator_control(
    program: RigidIR,
    allowed_articulated_joint_names_by_body: dict[str, tuple[str, ...]] | None,
) -> list[str]:
    errors: list[str] = []
    articulated_body_names = {body.name for body in program.bodies if body.shape.kind in {"mjcf", "urdf"}}
    direct_state_action_types = (SetPoseActionIR, SetDofsPositionActionIR, SetDofsVelocityActionIR)
    for idx, action in enumerate(program.actions):
        is_direct_state_action = isinstance(action, direct_state_action_types)
        is_articulated_body = any(entity in articulated_body_names for entity in selected_entities(getattr(action, "entity", None)))
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
                from ...llm_generator.agents.xml_agent import list_named_joint_names

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


def validate_deformable_articulated_collision_geometry(program: RigidIR) -> list[str]:
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
