from __future__ import annotations

from typing import Any

from .overrides import GeneratorParameterOverrides
from .tool_specs import (
    COMPACT_HARD_RULE_KEYS,
    build_generation_guide_payload,
    build_observation_field_guide_payload,
    build_schema_payload,
    build_tool_specs,
)


def build_generator_tool_context(
    *,
    xml_generation_enabled: bool = True,
    mesh_generation_enabled: bool = True,
    parameter_overrides: GeneratorParameterOverrides | None = None,
) -> dict[str, Any]:
    guide = build_generation_guide_payload(
        required_shape_kind=None,
        required_shape_file=None,
        allowed_shape_kinds=None,
        allowed_articulated_joint_names_by_body=None,
        enforce_articulated_actuator_control=True,
        target_sim_duration_sec=None,
        duration_tolerance_sec=0.75,
        xml_generation_enabled=xml_generation_enabled,
        generated_xml_paths_by_body=None,
        mesh_generation_enabled=mesh_generation_enabled,
        generated_mesh_paths_by_body=None,
        parameter_overrides=parameter_overrides,
    )
    constraints = dict(guide["constraints"])
    constraints["direct_state_actions_pre_step_only"] = [
        "set_pose",
        "set_dofs_position",
        "set_dofs_velocity",
    ]
    constraints["implementable_fix_rule"] = (
        "Only recommend changes expressible through the current generator tool library, IR fields, "
        "MJCF generation path, and supported action ops."
    )
    return {
        "tool_specs": build_tool_specs(
            xml_generation_enabled=xml_generation_enabled,
            mesh_generation_enabled=mesh_generation_enabled,
        ),
        "generation_guide": {
            "ok": guide["ok"],
            "mode": guide["mode"],
            "constraints": constraints,
            "templates": guide["templates"],
        },
        "observation_field_guide": build_observation_field_guide_payload(),
        "schema": build_schema_payload()["schema"],
    }


def build_compact_generator_tool_context(
    *,
    xml_generation_enabled: bool = True,
    mesh_generation_enabled: bool = True,
    parameter_overrides: GeneratorParameterOverrides | None = None,
) -> dict[str, Any]:
    guide = build_generation_guide_payload(
        required_shape_kind=None,
        required_shape_file=None,
        allowed_shape_kinds=None,
        allowed_articulated_joint_names_by_body=None,
        enforce_articulated_actuator_control=True,
        target_sim_duration_sec=None,
        duration_tolerance_sec=0.75,
        xml_generation_enabled=xml_generation_enabled,
        generated_xml_paths_by_body=None,
        mesh_generation_enabled=mesh_generation_enabled,
        generated_mesh_paths_by_body=None,
        parameter_overrides=parameter_overrides,
    )
    constraints = guide["constraints"]
    parameter_notes = constraints.get("parameter_notes", {})
    relationship_notes = constraints.get("parameter_relationship_notes", {})

    compact_parameter_keys = [
        "scene.ground_collision.friction",
        "bodies[].collision.friction",
        "bodies[].collision.coup_restitution",
        "bodies[].rho",
        "bodies[].shape.default_armature",
        "bodies[].fixed",
        "bodies[].actuators[].kp",
        "bodies[].actuators[].kv",
        "bodies[].actuators[].force_range",
        "SetTargetPosActionIR.values",
        "SetTorqueActionIR.values",
        "ApplyExternalWrenchActionIR.force",
        "ApplyExternalWrenchActionIR.torque",
        "ApplyExternalWrenchActionIR.ref",
        "ApplyExternalWrenchActionIR.local",
        "ObserveActionIR.entity",
        "SetPoseActionIR.entity",
        "ApplyExternalWrenchActionIR.entity",
    ]

    return {
        "capabilities": {
            "allowed_shape_kinds": constraints.get("allowed_shape_kinds"),
            "multiple_bodies": constraints.get("multi_body_supported"),
            "multiple_articulated_bodies": constraints.get("multi_articulated_supported"),
            "multi_entity_action_support": constraints.get("multi_entity_action_support"),
            "recommended_articulated_action_ops": constraints.get("recommended_articulated_action_ops"),
            "render_follow_entity_supported": constraints.get("render_follow_entity_supported"),
            "xml_generation_is_available": constraints.get("xml_generation_is_available"),
            "mesh_generation_is_available": constraints.get("mesh_generation_is_available"),
        },
        "hard_rules": {
            **{
                key: constraints.get(key)
                for key in COMPACT_HARD_RULE_KEYS
            },
            "implementable_fix_rule": (
                "Only recommend changes expressible through the current generator tool library, IR fields, "
                "MJCF generation path, and supported action ops."
            ),
        },
        "fixed_parameter_overrides": constraints.get("fixed_parameter_overrides"),
        "parameter_notes": {
            key: parameter_notes[key]
            for key in compact_parameter_keys
            if key in parameter_notes
        },
        "parameter_relationship_notes": {
            key: relationship_notes[key]
            for key in ("position_actuator_tuning", "external_wrench_usage")
            if key in relationship_notes
        },
    }
