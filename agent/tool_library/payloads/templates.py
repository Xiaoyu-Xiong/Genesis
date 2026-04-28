from __future__ import annotations

from ...configs import CONFIGS


def build_templates(
    *,
    effective_dt: float,
    effective_render: dict[str, object],
    default_scene_backend: str,
) -> dict[str, object]:
    if CONFIGS.deformable.simulation_backend == "pbd":
        deformable_material_example: dict[str, object] = {
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
            "backend": default_scene_backend,
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
            "backend": default_scene_backend,
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
