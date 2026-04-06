from __future__ import annotations

from typing import Any

from ..ir_schema import IR_VERSION, RigidIR
from ..llm_generator.constraints.general_constraints import ALLOWED_OBSERVE_FIELDS, default_render_config
from .overrides import GeneratorParameterOverrides
from .tool_spec_rules import (
    ARTICULATED_BODY_MESH_POLICY,
    ARTICULATED_BODY_XML_POLICY,
    ARTICULATED_DECISION_POLICY,
    BODY_COUNT_POLICY,
    BODY_NAMING_POLICY,
    FIXED_BODY_NOTE,
    IR_CONCISENESS_POLICY,
    MESH_BODY_POLICY,
    MESH_DECISION_POLICY,
    MESH_LOCAL_FRAME_POLICY,
    MESH_REUSE_POLICY,
    ROOT_STRUCTURE_NOTE,
)


def build_generation_guide_payload(
    *,
    required_shape_kind: str | None,
    required_shape_file: str | None,
    allowed_shape_kinds: tuple[str, ...] | None,
    allowed_articulated_joint_names_by_body: dict[str, tuple[str, ...]] | None,
    enforce_articulated_actuator_control: bool,
    target_sim_duration_sec: float | None,
    duration_tolerance_sec: float,
    xml_generation_enabled: bool,
    generated_xml_paths_by_body: dict[str, str] | None,
    mesh_generation_enabled: bool = False,
    generated_mesh_paths_by_body: dict[str, str] | None = None,
    parameter_overrides: GeneratorParameterOverrides | None = None,
) -> dict[str, Any]:
    effective_dt = 0.01 if parameter_overrides is None or parameter_overrides.sim_dt is None else parameter_overrides.sim_dt
    effective_render = default_render_config()
    if parameter_overrides is not None:
        if parameter_overrides.render_every_n_steps is not None:
            effective_render["render_every_n_steps"] = parameter_overrides.render_every_n_steps
        if parameter_overrides.render_res is not None:
            effective_render["res"] = list(parameter_overrides.render_res)
        if parameter_overrides.sim_dt is not None and parameter_overrides.render_every_n_steps is not None:
            effective_render["fps"] = max(1, int(round(1.0 / (effective_dt * effective_render["render_every_n_steps"]))))

    constraints = _build_constraints(
        required_shape_kind=required_shape_kind,
        required_shape_file=required_shape_file,
        allowed_shape_kinds=allowed_shape_kinds,
        allowed_articulated_joint_names_by_body=allowed_articulated_joint_names_by_body,
        enforce_articulated_actuator_control=enforce_articulated_actuator_control,
        target_sim_duration_sec=target_sim_duration_sec,
        duration_tolerance_sec=duration_tolerance_sec,
        xml_generation_enabled=xml_generation_enabled,
        generated_xml_paths_by_body=generated_xml_paths_by_body,
        mesh_generation_enabled=mesh_generation_enabled,
        generated_mesh_paths_by_body=generated_mesh_paths_by_body,
        parameter_overrides=parameter_overrides,
    )
    templates = _build_templates(effective_dt=effective_dt, effective_render=effective_render)
    return {"ok": True, "mode": "general_rigid_scene", "constraints": constraints, "templates": templates}


def build_observation_field_guide_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "fields": {
            "pos": {"meaning": "world-frame position of body origin", "dimension": 3},
            "quat": {"meaning": "world-frame orientation quaternion (w,x,y,z)", "dimension": 4},
            "vel": {"meaning": "world-frame linear velocity", "dimension": 3},
            "ang": {"meaning": "world-frame angular velocity", "dimension": 3},
            "qpos": {"meaning": "generalized joint position state (articulated only)", "dimension": "N"},
            "dofs_position": {"meaning": "local dof positions (articulated only)", "dimension": "N"},
            "dofs_velocity": {"meaning": "local dof velocities (articulated only)", "dimension": "N"},
        },
    }


def build_schema_payload() -> dict[str, Any]:
    return {"ok": True, "schema": RigidIR.model_json_schema()}


def build_generation_bootstrap_payload(
    *,
    required_shape_kind: str | None,
    required_shape_file: str | None,
    allowed_shape_kinds: tuple[str, ...] | None,
    allowed_articulated_joint_names_by_body: dict[str, tuple[str, ...]] | None,
    enforce_articulated_actuator_control: bool,
    target_sim_duration_sec: float | None,
    duration_tolerance_sec: float,
    xml_generation_enabled: bool,
    generated_xml_paths_by_body: dict[str, str] | None,
    mesh_generation_enabled: bool = False,
    generated_mesh_paths_by_body: dict[str, str] | None = None,
    parameter_overrides: GeneratorParameterOverrides | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "generation_guide": build_generation_guide_payload(
            required_shape_kind=required_shape_kind,
            required_shape_file=required_shape_file,
            allowed_shape_kinds=allowed_shape_kinds,
            allowed_articulated_joint_names_by_body=allowed_articulated_joint_names_by_body,
            enforce_articulated_actuator_control=enforce_articulated_actuator_control,
            target_sim_duration_sec=target_sim_duration_sec,
            duration_tolerance_sec=duration_tolerance_sec,
            xml_generation_enabled=xml_generation_enabled,
            generated_xml_paths_by_body=generated_xml_paths_by_body,
            mesh_generation_enabled=mesh_generation_enabled,
            generated_mesh_paths_by_body=generated_mesh_paths_by_body,
            parameter_overrides=parameter_overrides,
        ),
        "observation_field_guide": build_observation_field_guide_payload(),
        "schema": build_schema_payload()["schema"],
    }


def _build_constraints(
    *,
    required_shape_kind: str | None,
    required_shape_file: str | None,
    allowed_shape_kinds: tuple[str, ...] | None,
    allowed_articulated_joint_names_by_body: dict[str, tuple[str, ...]] | None,
    enforce_articulated_actuator_control: bool,
    target_sim_duration_sec: float | None,
    duration_tolerance_sec: float,
    xml_generation_enabled: bool,
    generated_xml_paths_by_body: dict[str, str] | None,
    mesh_generation_enabled: bool,
    generated_mesh_paths_by_body: dict[str, str] | None,
    parameter_overrides: GeneratorParameterOverrides | None,
) -> dict[str, Any]:
    constraints: dict[str, Any] = {
        "ir_version": IR_VERSION,
        "allowed_shape_kinds": ["sphere", "box", "cylinder", "mesh", "mjcf", "urdf"],
        "multi_body_supported": True,
        "multi_articulated_supported": True,
        "root_structure_note": ROOT_STRUCTURE_NOTE,
        "body_count_policy": BODY_COUNT_POLICY,
        "body_naming_policy": BODY_NAMING_POLICY,
        "articulated_body_xml_policy": ARTICULATED_BODY_XML_POLICY,
        "articulated_decision_policy": ARTICULATED_DECISION_POLICY,
        "mesh_body_policy": MESH_BODY_POLICY,
        "mesh_decision_policy": MESH_DECISION_POLICY,
        "mesh_reuse_policy": MESH_REUSE_POLICY,
        "mesh_local_frame_policy": MESH_LOCAL_FRAME_POLICY,
        "articulated_body_mesh_policy": ARTICULATED_BODY_MESH_POLICY,
        "ir_conciseness_policy": IR_CONCISENESS_POLICY,
        "fixed_body_support": True,
        "fixed_body_note": FIXED_BODY_NOTE,
        "multi_entity_action_support": {
            "observe": (
                "Supports `entity` as a single body name or a list of body names; emits one observation event per body. "
                "Use the list form when fields/tag/timing are identical."
            ),
            "set_pose": (
                "Supports `entity` as a single body name or a list of body names; applies the same pose update to each "
                "selected body. Use the list form when the same pose update should be broadcast."
            ),
            "apply_external_wrench": (
                "Supports `entity` as a single body name or a list of body names; applies the same external wrench update "
                "to each selected body. Use the list form when the same disturbance should be broadcast."
            ),
        },
        "allowed_observe_fields": sorted(ALLOWED_OBSERVE_FIELDS),
        "backend_default": "cpu",
        "show_viewer_default": False,
        "render_required": True,
        "default_render_enabled": True,
        "default_render_output_video": default_render_config()["output_video"],
        "render_follow_entity_supported": True,
        "recommended_articulated_action_ops": ["set_target_pos", "set_torque"],
        "default_observation_policy": {
            "final_step<=0": "require >=1 observe",
            "0<final_step<100": "require >=2 observes, one pre-final and one at final step",
            "final_step>=100": "require >=3 observes, with one intermediate and one at final step",
        },
    }
    if required_shape_kind is not None:
        constraints["required_shape_kind"] = required_shape_kind
    if required_shape_file is not None:
        constraints["required_shape_file"] = required_shape_file
    if allowed_shape_kinds is not None:
        constraints["enforced_allowed_shape_kinds"] = list(allowed_shape_kinds)
    if allowed_articulated_joint_names_by_body is not None:
        constraints["allowed_articulated_joint_names_by_body"] = {
            body_name: list(joint_names)
            for body_name, joint_names in sorted(allowed_articulated_joint_names_by_body.items())
        }
    if enforce_articulated_actuator_control:
        constraints["articulated_motion_policy"] = {
            "forbid_actions": ["set_pose", "set_dofs_position", "set_dofs_velocity"],
            "require_articulated_body_actuators": True,
            "require_at_least_one_actuator_control_action": True,
            "actuator_type_rules": {
                "position_actuator": ["set_target_pos"],
                "motor_actuator": ["set_torque"],
            },
        }
    constraints["pre_sim_only_actions"] = ["set_pose", "set_dofs_position", "set_dofs_velocity"]
    if parameter_overrides is not None:
        constraints["fixed_parameter_overrides"] = parameter_overrides.as_dict()
        constraints["fixed_parameter_override_policy"] = (
            "These parameters are fixed by the system. Treat them as hard constraints and do not try to "
            "change or work around them."
        )
    constraints["parameter_notes"] = _build_parameter_notes()
    constraints["parameter_relationship_notes"] = _build_parameter_relationship_notes()
    if target_sim_duration_sec is not None:
        constraints["target_sim_duration_sec"] = target_sim_duration_sec
        constraints["sim_duration_tolerance_sec"] = duration_tolerance_sec
        constraints["sim_duration_definition"] = "sim_duration_sec = final_step * scene.sim.dt"
    constraints["xml_generation_is_available"] = xml_generation_enabled
    constraints["mesh_generation_is_available"] = mesh_generation_enabled
    if generated_xml_paths_by_body is not None:
        constraints["generated_xml_paths_by_body"] = dict(sorted(generated_xml_paths_by_body.items()))
    if generated_mesh_paths_by_body is not None:
        constraints["generated_mesh_paths_by_body"] = dict(sorted(generated_mesh_paths_by_body.items()))
    return constraints


def _build_parameter_notes() -> dict[str, str]:
    return {
        "scene.render.follow_entity.smoothing": (
            "Follow-camera smoothing factor. Higher values make camera motion smoother but increase lag."
        ),
        "scene.render.follow_entity.fixed_axis": (
            "Per-axis lock for follow-camera target position. Use null on an axis to follow the entity on that axis, "
            "or set a number to keep that axis fixed."
        ),
        "bodies[].collision.coup_restitution": (
            "Impact bounciness. Higher values create more rebound and usually make contact behavior less stable."
        ),
        "scene.ground_collision.friction": (
            "Contact friction coefficient. Higher values resist sliding more strongly, but do not guarantee perfectly non-slipping contact."
        ),
        "bodies[].collision.friction": (
            "Contact friction coefficient. Higher values resist sliding more strongly, but do not guarantee perfectly non-slipping contact."
        ),
        "bodies[].rho": (
            "Material density. Higher rho makes the body heavier and increases inertia, but does not change geometric size."
        ),
        "bodies[].shape.default_armature": (
            "Additional articulated-joint armature used mainly for stability and numerical conditioning, not for "
            "task-level motion design."
        ),
        "bodies[].fixed": (
            "Whether a body is fixed in the world. Use this for primitive, mesh, or URDF obstacles, tables, and props "
            "that should not fall under gravity. For MJCF, express a fixed base in the XML itself."
        ),
        "bodies[].actuators[].kp": (
            "Position-control stiffness. Increasing kp makes tracking more aggressive, but if it is too large the "
            "joint can oscillate or destabilize."
        ),
        "bodies[].actuators[].kv": (
            "Position-control damping. kv suppresses oscillation and overshoot; too little damping can be shaky, "
            "too much can make motion sluggish."
        ),
        "bodies[].actuators[].force_range": (
            "Actuator output limit. This caps the maximum available force/torque; if it is too small, the joint may "
            "still be weak even when kp is large."
        ),
        "SetTargetPosActionIR.values": (
            "Target positions for position actuators. These are desired setpoints, not direct joint-state writes."
        ),
        "SetTorqueActionIR.values": (
            "Direct force/torque commands for motor actuators. These do not provide position tracking on their own."
        ),
        "ApplyExternalWrenchActionIR.force": (
            "External force disturbance applied to a body or selected links. It is not an actuator command. Its "
            "effect persists across subsequent step actions until another wrench update changes it. If the effect is "
            "too weak or too strong, prefer adjusting force magnitude first before changing application duration."
        ),
        "ApplyExternalWrenchActionIR.torque": (
            "External torque disturbance applied to a body or selected links. It is not an actuator command. Its "
            "effect persists across subsequent step actions until another wrench update changes it."
        ),
        "ApplyExternalWrenchActionIR.ref": (
            "Reference point used for the external wrench (`link_origin`, `link_com`, or `root_com`). This changes "
            "how the same force produces translation versus rotation."
        ),
        "ApplyExternalWrenchActionIR.local": (
            "Whether the force/torque vector is interpreted in the world frame (`false`) or the target link's local "
            "frame (`true`)."
        ),
        "ObserveActionIR.entity": (
            "May be a single body name or a list of body names. Use the list form when observing several bodies with "
            "the same fields and tag at the same timestep."
        ),
        "SetPoseActionIR.entity": (
            "May be a single body name or a list of body names. Use the list form when broadcasting the same pose "
            "change to several bodies."
        ),
        "ApplyExternalWrenchActionIR.entity": (
            "May be a single body name or a list of body names. Use the list form when broadcasting the same "
            "external disturbance to several bodies."
        ),
    }


def _build_parameter_relationship_notes() -> dict[str, str]:
    return {
        "position_actuator_tuning": (
            "For position actuators, kp sets how hard the controller tries to reach the target, kv damps motion, "
            "and force_range caps the actual output. If motion is too weak or the target is not reached, the cause "
            "may be insufficient kp, insufficient force_range, or both. If motion is too oscillatory, kp may be too "
            "high, kv may be too low, or force_range may be large enough to expose that instability. Critiques and "
            "fixes should distinguish between insufficient stiffness, insufficient damping, and insufficient output limit."
        ),
        "external_wrench_usage": (
            "`apply_external_wrench` is best understood as writing an external disturbance state into the solver, not "
            "as a one-step impulse helper. A common pattern is: set nonzero force/torque, step for some duration, then "
            "set the wrench back to zero. Critiques and fixes should distinguish between too-small wrench magnitude, "
            "too-short application duration, wrong reference point (`ref`), and wrong frame interpretation (`local`). "
            "When tuning the effect, prefer changing force/torque magnitude first and only then changing how long the "
            "wrench stays applied."
        ),
    }


def _build_templates(*, effective_dt: float, effective_render: dict[str, Any]) -> dict[str, Any]:
    return {
        "minimal_scene": {
            "backend": "cpu",
            "show_viewer": False,
            "add_ground": True,
            "sim": {"dt": effective_dt, "gravity": [0.0, 0.0, -9.81]},
            "render": effective_render,
        },
        "minimal_bodies": [
            {
                "name": "robot",
                "shape": {"kind": "mjcf", "file": "path/to/model.xml", "scale": 1.0},
            },
            {
                "name": "target_box",
                "fixed": True,
                "shape": {"kind": "box", "size": [0.3, 0.3, 0.3]},
                "initial_pose": {"pos": [1.0, 0.0, 0.15], "quat": [1.0, 0.0, 0.0, 0.0]},
            },
        ],
        "mesh_body_example": {
            "name": "workpiece_tray",
            "fixed": True,
            "shape": {"kind": "mesh", "file": "path/to/tray.obj", "scale": 1.0},
            "initial_pose": {"pos": [0.8, 0.0, 0.12], "quat": [1.0, 0.0, 0.0, 0.0]},
        },
        "mesh_reuse_example": [
            {
                "name": "crate_a",
                "shape": {"kind": "mesh", "file": "path/to/shared_crate.obj", "scale": 1.0},
                "initial_pose": {"pos": [0.0, 0.0, 0.2], "quat": [1.0, 0.0, 0.0, 0.0]},
            },
            {
                "name": "crate_b",
                "shape": {"kind": "mesh", "file": "path/to/shared_crate.obj", "scale": 1.0},
                "initial_pose": {"pos": [0.8, 0.0, 0.2], "quat": [1.0, 0.0, 0.0, 0.0]},
            },
        ],
        "multiple_articulated_bodies_example": [
            {
                "name": "robot_a",
                "shape": {"kind": "mjcf", "file": "path/to/robot_a.xml", "scale": 1.0},
                "actuators": [
                    {
                        "kind": "position",
                        "name": "joint0_pos",
                        "joint_names": ["joint0"],
                        "force_range": {"lower": -120.0, "upper": 120.0},
                    }
                ],
            },
            {
                "name": "robot_b",
                "shape": {"kind": "mjcf", "file": "path/to/robot_b.xml", "scale": 1.0},
                "actuators": [
                    {
                        "kind": "position",
                        "name": "joint1_pos",
                        "joint_names": ["joint1"],
                        "force_range": {"lower": -120.0, "upper": 120.0},
                    }
                ],
            },
        ],
        "render_follow_entity_example": {
            "render": {
                **effective_render,
                "follow_entity": {
                    "entity": "robot",
                    "fixed_axis": [None, None, 0.8],
                    "smoothing": 0.9,
                    "fix_orientation": False,
                },
            }
        },
        "multi_entity_observe_example": {
            "op": "observe",
            "entity": ["striker", "domino_01", "domino_02"],
            "fields": ["pos", "quat", "vel", "ang"],
            "tag": "early_state_multi",
        },
        "shapes": {
            "sphere": {"kind": "sphere", "radius": 0.2},
            "box": {"kind": "box", "size": [0.4, 0.3, 0.2]},
            "cylinder": {"kind": "cylinder", "radius": 0.1, "height": 0.4},
            "mesh": {"kind": "mesh", "file": "path/to/prop.obj", "scale": 1.0},
            "mjcf": {"kind": "mjcf", "file": "path/to/model.xml", "scale": 1.0},
            "urdf": {"kind": "urdf", "file": "path/to/model.urdf", "scale": 1.0},
        },
        "articulated_actuator_setup": {
            "body_actuators_example": [
                {"kind": "position", "name": "joint0_pos", "dofs_idx_local": [0], "kp": 80.0, "kv": 6.0},
                {
                    "kind": "motor",
                    "name": "joint1_motor",
                    "dofs_idx_local": [1],
                    "force_range": {"lower": -200.0, "upper": 200.0},
                },
            ],
            "actions_example": [
                {"op": "set_target_pos", "entity": "robot", "actuator": "joint0_pos", "values": [0.8]},
                {"op": "set_torque", "entity": "robot", "actuator": "joint1_motor", "values": [30.0]},
                {"op": "step", "steps": 120},
                {
                    "op": "observe",
                    "entity": "robot",
                    "fields": ["qpos", "dofs_position", "dofs_velocity"],
                    "tag": "after_control",
                },
            ],
        },
    }
