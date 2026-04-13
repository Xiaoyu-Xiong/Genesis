from __future__ import annotations

from typing import Any

from ..defaults import DEFAULTS
from ..ir_schema import IR_VERSION, RigidIR
from ..llm_generator.constraints.general_constraints import ALLOWED_OBSERVE_FIELDS, default_render_config
from .tool_spec_rules import (
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
    return "cpu" if DEFAULTS.deformable.simulation_backend == "fem_ipc" else DEFAULTS.optimization.backend


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
    effective_dt = DEFAULTS.runtime.sim_dt
    effective_render = default_render_config()
    effective_render["render_every_n_steps"] = DEFAULTS.runtime.render_every_n_steps
    effective_render["res"] = list(DEFAULTS.runtime.render_res)
    effective_render["fps"] = max(1, min(240, int(round(1.0 / (effective_dt * DEFAULTS.runtime.render_every_n_steps)))))

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
    templates = _build_templates(effective_dt=effective_dt, effective_render=effective_render)
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
        "active_deformable_backend": DEFAULTS.deformable.simulation_backend,
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
    constraints["parameter_notes"] = _build_parameter_notes()
    constraints["parameter_relationship_notes"] = _build_parameter_relationship_notes()
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


def _build_parameter_notes() -> dict[str, str]:
    notes = {
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
            "Contact friction coefficient. Higher values resist sliding more strongly, but do not guarantee perfectly non-slipping contact. Use 0.8 as a reasonable default."
        ),
        "bodies[].collision.friction": (
            "Contact friction coefficient. Higher values resist sliding more strongly, but do not guarantee perfectly non-slipping contact. Use 0.8 as a reasonable default."
        ),
        "bodies[].rho": (
            "Material density. Higher rho makes the body heavier and increases inertia, but does not change geometric "
            "size. Keep density in the range 300 to 3000 kg/m^3."
        ),
        "bodies[].shape.default_armature": (
            "Additional articulated-joint armature used mainly for stability and numerical conditioning, not for "
            "task-level motion design."
        ),
        "bodies[].shape.scale": (
            "Uniform mesh scale factor. Use this when a mesh body's overall size is wrong for the scene: increase it "
            "to make the whole mesh larger, decrease it to make the whole mesh smaller. For deformable mesh bodies, "
            "this also changes the physical tetrahedralization size because the geometry itself is rescaled before "
            "remeshing and TetGen. When mesh bounding-box metadata is available, use that `bbox_size` evidence to "
            "estimate `shape.scale` instead of guessing."
        ),
        "bodies[].fixed": (
            "Whether a rigid body is fixed in the world. Use this for rigid primitive, rigid mesh, or URDF obstacles, "
            "tables, and props that should not fall under gravity. For MJCF, express a fixed base in the XML itself."
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
        "bodies[].simulation_kind": (
            "Choose `deformable` when soft-body deformation visually makes the task closer to the prompt. Otherwise prefer `rigid`."
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
    if DEFAULTS.deformable.simulation_backend == "pbd":
        notes["bodies[].deformable_material.stretch_compliance"] = (
            "PBD stretch compliance. Lower values make the soft body stiffer in edge-length preservation; higher "
            "values make it stretch and sag more easily. Very stiff elastic solids are often around 1e-8 to 1e-6, "
            "softer jelly-like solids around 1e-6 to 1e-4, and very floppy bodies can go higher. If the body is too "
            "floppy, decrease this value by about 3x to 10x; if it is too rigid, increase it by about 3x to 10x. "
            "Use **3e-5** as default for a moderately soft material with clearly visible deformation."
        )
        notes["bodies[].deformable_material.volume_compliance"] = (
            "PBD volume compliance. Lower values preserve volume more strongly; higher values allow more compression. "
            "If the body collapses or squashes too much, decrease this value; if it stays too incompressible, increase it. "
            "Use **3e-6** as default for visibly compressible but not completely mushy behavior."
        )
        notes["bodies[].deformable_material.rho"] = (
            "Density for PBD elastic bodies. Larger values make the soft body heavier without changing its geometry. "
            "Keep deformable density in the range 300 to 3000 kg/m^3."
        )
    else:
        notes["bodies[].deformable_material.E"] = (
            "Young's modulus for FEM elastic bodies, measured in Pascals, controls the body's resistance to stretching "
            "and compression. Higher `E` makes the body stiffer, lower `E` makes it softer. As a rough guide: very "
            "soft jelly-like solids are often around `1e4` to `5e4`, moderately soft rubbery solids around `5e4` to "
            "`5e5`, and firmer but still visibly deformable solids around `5e5` to `5e6`. If the body visibly collapses "
            "too much or cannot support load, increase `E` by about 3x to 10x; if it hardly deforms, decrease `E` by "
            "about 3x to 10x. Use about **1e5** as a good default initial guess for a medium-elastic soft solid."
        )
        notes["bodies[].deformable_material.nu"] = (
            "Poisson ratio for FEM elastic bodies. Higher values make the material less compressible. "
            "Use about **0.35** as a good default initial guess for a moderately compressible soft solid."
        )
        notes["bodies[].deformable_material.rho"] = (
            "Density for FEM elastic bodies. Larger values make the deformable body heavier without changing its "
            "geometry. Keep deformable density in the range 300 to 3000 kg/m^3."
        )
        notes["bodies[].initial_pose.pos"] = (
            "For FEM+IPC scenes, initial placements must avoid penetration and interpenetration. Leave a small positive "
            "clearance between deformable bodies, rigid bodies, and support surfaces instead of starting in overlap."
        )
    return notes


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
    if DEFAULTS.deformable.simulation_backend == "pbd":
        deformable_material_example = {
            "kind": "elastic",
            "rho": 1100.0,
            "stretch_compliance": 1e-4,
            "volume_compliance": 1e-5,
        }
    else:
        deformable_material_example = {
            "kind": "elastic",
            "rho": 1000.0,
            "E": 5e5,
            "nu": 0.35,
        }
    return {
        "minimal_scene": {
            "backend": _default_scene_backend(),
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
        "deformable_body_example": {
            "name": "soft_cylinder",
            "simulation_kind": "deformable",
            "shape": {"kind": "cylinder", "radius": 0.08, "height": 0.24},
            "deformable_material": deformable_material_example,
            "initial_pose": {"pos": [0.0, 0.0, 0.4], "quat": [1.0, 0.0, 0.0, 0.0]},
        },
        "deformable_mesh_body_example": {
            "name": "soft_mesh_blob",
            "simulation_kind": "deformable",
            "shape": {"kind": "mesh", "file": "path/to/generated_or_existing_blob.obj", "scale": 1.0},
            "deformable_material": deformable_material_example,
            "initial_pose": {"pos": [0.0, 0.0, 0.5], "quat": [1.0, 0.0, 0.0, 0.0]},
        },
        "deformable_scene_example": {
            "backend": _default_scene_backend(),
            "show_viewer": False,
            "add_ground": True,
            "sim": {"dt": effective_dt, "gravity": [0.0, 0.0, -9.81]},
            "render": effective_render,
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
        "deformable_observe_example": {
            "op": "observe",
            "entity": "soft_cylinder",
            "fields": ["pos", "vel", "bbox_size", "vertex_disp_max"],
            "tag": "soft_response",
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
