from __future__ import annotations

from typing import Any

from ...configs import CONFIGS
from ...ir_schema.common import IR_VERSION
from ...ir_schema.program import RigidIR
from ...llm_generator.constraints.general_constraints import ALLOWED_OBSERVE_FIELDS, default_render_config
from .notes import build_parameter_notes, build_parameter_relationship_notes
from .templates import build_templates
from ..constraints.rules import (
    ARTICULATED_BODY_MESH_POLICY,
    ARTICULATED_BODY_XML_POLICY,
    ARTICULATED_DECISION_POLICY,
    BODY_COUNT_POLICY,
    BODY_NAMING_POLICY,
    DEFORMABLE_ACTION_POLICY,
    DEFORMABLE_BODY_POLICY,
    DEFORMABLE_GEOMETRY_POLICY,
    DEFORMABLE_MATERIAL_POLICY,
    DEFORMABLE_MESH_ASSET_POLICY,
    DEFORMABLE_OBSERVE_POLICY,
    DEFORMABLE_SCENE_POLICY,
    DYNAMIC_SCENE_POLICY,
    FIXED_BODY_NOTE,
    IR_CONCISENESS_POLICY,
    MESH_BODY_POLICY,
    MESH_BBOX_POLICY,
    MESH_DECISION_POLICY,
    MESH_LOCAL_FRAME_POLICY,
    MESH_REUSE_POLICY,
    MESH_SCALE_POLICY,
    ROOT_STRUCTURE_NOTE,
)


def _default_scene_backend() -> str:
    return "cpu" if CONFIGS.deformable.simulation_backend == "fem_ipc" else CONFIGS.optimization.backend


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
    generated_mesh_summaries_by_body: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    effective_dt = CONFIGS.runtime.sim_dt
    effective_render = default_render_config()
    effective_render["render_every_n_steps"] = CONFIGS.runtime.render_every_n_steps
    effective_render["res"] = list(CONFIGS.runtime.render_res)
    effective_render["fps"] = max(1, min(240, int(round(1.0 / (effective_dt * CONFIGS.runtime.render_every_n_steps)))))

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
        generated_mesh_summaries_by_body=generated_mesh_summaries_by_body,
    )
    templates = build_templates(
        effective_dt=effective_dt,
        effective_render=effective_render,
        default_scene_backend=_default_scene_backend(),
    )
    return {"ok": True, "mode": "general_rigid_scene", "constraints": constraints, "templates": templates}


def build_observation_field_guide_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "fields": {
            "pos": {
                "meaning": "world-frame position of the body origin for rigid bodies, or center-of-mass position for deformable bodies",
                "dimension": 3,
            },
            "quat": {"meaning": "world-frame orientation quaternion (w,x,y,z)", "dimension": 4},
            "vel": {
                "meaning": "world-frame linear velocity for rigid bodies, or center-of-mass average velocity for deformable bodies",
                "dimension": 3,
            },
            "ang": {"meaning": "world-frame angular velocity", "dimension": 3},
            "qpos": {"meaning": "generalized joint position state (articulated only)", "dimension": "N"},
            "dofs_position": {"meaning": "local dof positions (articulated only)", "dimension": "N"},
            "dofs_velocity": {"meaning": "local dof velocities (articulated only)", "dimension": "N"},
            "bbox_min": {"meaning": "axis-aligned bounding box minimum corner for the current body state", "dimension": 3},
            "bbox_max": {"meaning": "axis-aligned bounding box maximum corner for the current body state", "dimension": 3},
            "bbox_size": {"meaning": "axis-aligned bounding box size for the current body state", "dimension": 3},
            "vertex_disp_mean": {
                "meaning": "mean vertex displacement magnitude relative to the deformable body's initial configuration",
                "dimension": 1,
            },
            "vertex_disp_max": {
                "meaning": "maximum vertex displacement magnitude relative to the deformable body's initial configuration",
                "dimension": 1,
            },
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
    generated_mesh_summaries_by_body: dict[str, dict[str, Any]] | None = None,
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
            generated_mesh_summaries_by_body=generated_mesh_summaries_by_body,
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
    generated_mesh_summaries_by_body: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    constraints: dict[str, Any] = {
        "ir_version": IR_VERSION,
        "allowed_shape_kinds": ["sphere", "box", "cylinder", "mesh", "mjcf", "urdf"],
        "supported_simulation_kinds": ["rigid", "deformable"],
        "active_deformable_backend": CONFIGS.deformable.simulation_backend,
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
        "mesh_reuse_preference": (
            "If repeating one mesh will not cause a clearly noticeable visual loss, prefer reusing that mesh across "
            "multiple bodies instead of generating near-duplicate mesh assets."
        ),
        "mesh_local_frame_policy": MESH_LOCAL_FRAME_POLICY,
        "mesh_runtime_path_policy": (
            "For generated mesh assets, `bodies[].shape.file` must use the canonical runtime mesh path from "
            "`generate_mesh_asset` (`mesh_path`, usually under `processed/repaired*.obj`). Do not point the main IR "
            "to `textured_mesh_path`."
        ),
        "mesh_texture_branch_policy": (
            "If texture generation is enabled, the textured OBJ branch (`textured/model.obj`, MTL, base color, and "
            "related texture images) is an auxiliary texture asset branch only. It is not the runtime geometry input "
            "for the main simulation IR."
        ),
        "mesh_scale_policy": MESH_SCALE_POLICY,
        "mesh_bbox_policy": MESH_BBOX_POLICY,
        "articulated_body_mesh_policy": ARTICULATED_BODY_MESH_POLICY,
        "deformable_body_policy": DEFORMABLE_BODY_POLICY,
        "deformable_geometry_policy": DEFORMABLE_GEOMETRY_POLICY,
        "deformable_material_policy": DEFORMABLE_MATERIAL_POLICY,
        "deformable_action_policy": DEFORMABLE_ACTION_POLICY,
        "deformable_observe_policy": DEFORMABLE_OBSERVE_POLICY,
        "deformable_mesh_asset_policy": DEFORMABLE_MESH_ASSET_POLICY,
        "deformable_scene_policy": DEFORMABLE_SCENE_POLICY,
        "ir_conciseness_policy": IR_CONCISENESS_POLICY,
        "dynamic_scene_policy": DYNAMIC_SCENE_POLICY,
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
        "backend_default": _default_scene_backend(),
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
    constraints["parameter_notes"] = build_parameter_notes()
    constraints["parameter_relationship_notes"] = build_parameter_relationship_notes()
    if target_sim_duration_sec is not None:
        constraints["target_sim_duration_sec"] = target_sim_duration_sec
        constraints["sim_duration_tolerance_sec"] = duration_tolerance_sec
        constraints["sim_duration_definition"] = (
            "sim_duration_sec is determined by final_step under the system simulation timestep."
        )
    constraints["xml_generation_is_available"] = xml_generation_enabled
    constraints["mesh_generation_is_available"] = mesh_generation_enabled
    if generated_xml_paths_by_body is not None:
        constraints["generated_xml_paths_by_body"] = dict(sorted(generated_xml_paths_by_body.items()))
    if generated_mesh_paths_by_body is not None:
        constraints["generated_mesh_paths_by_body"] = dict(sorted(generated_mesh_paths_by_body.items()))
    if generated_mesh_summaries_by_body is not None:
        constraints["generated_mesh_summaries_by_body"] = dict(sorted(generated_mesh_summaries_by_body.items()))
    return constraints
