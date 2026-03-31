from __future__ import annotations

from pathlib import Path
from typing import Any

from ..ir_schema import (
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
    target_sim_duration_sec: float | None = None,
    sim_duration_tolerance_sec: float = 0.75,
) -> list[str]:
    errors: list[str] = []

    bodies = program.bodies
    articulated_bodies = [body for body in bodies if body.shape.kind in {"mjcf", "urdf"}]

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
