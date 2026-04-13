from __future__ import annotations

from ..defaults import DEFAULTS

ROOT_STRUCTURE_NOTE = "Use top-level `bodies` list."
BODY_COUNT_POLICY = "Multiple bodies are allowed, including multiple articulated bodies."
BODY_NAMING_POLICY = "Each body.name must be unique. Actions refer to bodies through the `entity` field."
FIXED_BODY_NOTE = (
    "Use `bodies[].fixed=true` for fixed rigid primitive bodies, rigid mesh bodies, or URDF objects such as obstacles, "
    "tables, and targets. For MJCF bodies, encode a fixed base inside the XML itself."
)
ARTICULATED_BODY_XML_POLICY = (
    "Each articulated body should have its own XML asset. When XML generation is used, call "
    "`generate_articulated_xml` separately for each articulated body and pass `body_name` explicitly."
)
ARTICULATED_DECISION_POLICY = (
    "Use articulated bodies only when the object truly requires joints or linked rigid parts and cannot be "
    "adequately represented as a single mesh body or a small primitive assembly. If an object could plausibly "
    "be modeled either as an articulated body or as a non-articulated mesh body, prefer the mesh body."
)
MESH_BODY_POLICY = (
    "Non-articulated bodies may use `shape.kind='mesh'` with a generated or preexisting mesh file. "
    "Use mesh bodies for props whose silhouette cannot be expressed cleanly with sphere/box/cylinder."
)
MESH_DECISION_POLICY = (
    "Prefer mesh for non-articulated props with visually important cutouts, handles, hollow interiors, tapered shells, "
    "tray lips, arch openings, multiple thin legs, feet, ribbing, or similar silhouette details. Prefer simple primitives "
    "for bodies that are already well represented by spheres, boxes, cylinders, walls, platforms, plain blocks, or basic pillars."
)
MESH_REUSE_POLICY = (
    "If several non-articulated bodies intentionally share the same geometry, generate the mesh asset once "
    "and reuse the same `shape.file` across those bodies. If reusing one mesh would not cause a clearly noticeable "
    "visual loss, prefer reusing the mesh instead of generating near-duplicate assets."
)
MESH_LOCAL_FRAME_POLICY = (
    "Generated mesh assets are post-processed so their geometric centroid is at the local origin. "
    "Use the returned axis-aligned bounding box to choose sensible world placement and scale."
)
MESH_SCALE_POLICY = (
    "For mesh bodies, if the imported or generated mesh is globally too large or too small for the scene, adjust "
    "`bodies[].shape.scale` to resize the whole mesh uniformly. Prefer changing `shape.scale` for overall mesh size "
    "corrections instead of changing the mesh file, density, stiffness, or unrelated parameters."
)
MESH_BBOX_POLICY = (
    "When mesh bounding-box metadata is provided, use `bbox_size` in mesh-local pre-scale units to estimate an "
    "appropriate `bodies[].shape.scale`. Estimate the final world-size first, then choose `shape.scale` from that "
    "evidence instead of guessing."
)
ARTICULATED_BODY_MESH_POLICY = (
    "Articulated bodies should be represented via MJCF/URDF and use only simple primitive geoms inside XML. "
    "Do not use mesh assets inside articulated-body XML in the current pipeline."
)
IR_CONCISENESS_POLICY = (
    "Prefer concise IR. If the same `observe`, `set_pose`, or `apply_external_wrench` should apply to multiple "
    "bodies at the same moment with identical parameters, use one action with `entity` as a body-name list "
    "instead of duplicating many near-identical actions."
)
DYNAMIC_SCENE_POLICY = (
    "When the task allows open-ended scene design, prefer scenes with clear motion, visible state changes, and "
    "meaningful contact-rich interactions instead of mostly static tableaux. Favor behaviors that make the scene "
    "evolve noticeably over the requested duration."
)
DEFORMABLE_BODY_POLICY = (
    "Use `simulation_kind='deformable'` when soft-body deformation will visually make the scene closer to the prompts; "
    "otherwise prefer rigid bodies."
)
DEFORMABLE_GEOMETRY_POLICY = (
    "In deformable v1, deformable bodies may only use `sphere`, `box`, `cylinder`, or `mesh` shapes."
)
if DEFAULTS.deformable.simulation_backend == "pbd":
    DEFORMABLE_MATERIAL_POLICY = (
        "In deformable v1, the active backend is PBD elastic. When specifying a deformable material, only set "
        "`rho`, `stretch_compliance`, and `volume_compliance`; particle size and solver iteration hyperparameters are "
        "fixed by the system. Lower compliance makes the body stiffer; higher compliance makes it softer or more compressible. "
        "A good default initial guess for clearly visible but still controlled softness is `stretch_compliance=1e-4` and "
        "`volume_compliance=1e-5`. Keep all density values in the range `300` to `3000` kg/m^3."
    )
    DEFORMABLE_SCENE_POLICY = (
        "In deformable PBD scenes, standard `scene.add_ground` semantics remain available. When `scene.add_ground=true`, "
        "the runtime configures the ground so rigid bodies collide with it while deformable PBD bodies continue to use "
        "PBD boundary handling instead of rigid-geometry coupling."
    )
else:
    DEFORMABLE_MATERIAL_POLICY = (
        "In deformable v1, the active backend is FEM+IPC. When specifying a deformable material, only set "
        "`rho`, `E`, and `nu`; IPC/FEM solver and contact hyperparameters are fixed by the system. "
        "Higher `E` makes the body stiffer, lower `E` makes it softer. Higher `nu` makes it less compressible. "
        "As a rough guide, very soft jelly-like solids are often around `E=1e4` to `5e4`, moderately soft rubbery "
        "solids around `E=5e4` to `5e5`, and firmer but still visibly deformable solids around `E=5e5` to `5e6`. "
        "A good default initial guess for a medium-elastic soft solid is `E=1e5`, `nu=0.35`, and `rho=1000`. "
        "Keep all density values in the range `300` to `3000` kg/m^3."
    )
    DEFORMABLE_SCENE_POLICY = (
        "In deformable FEM+IPC scenes, standard `scene.add_ground` semantics remain available. When `scene.add_ground=true`, "
        "the runtime keeps pure rigid-ground and pure rigid-rigid contact on Genesis' rigid solver, while FEM-involving "
        "contact uses IPC. The ground is represented both as a normal Genesis rigid ground for rigid bodies and as a "
        "hidden IPC-only plane for FEM-ground contact. When using FEM+IPC, do not generate bodies with initial "
        "penetration or interpenetration. Leave a small positive clearance between all bodies and support surfaces."
    )
DEFORMABLE_ACTION_POLICY = (
    "In deformable v1, deformable bodies are passive soft bodies. Do not use actuators, `set_pose`, `set_dofs_position`, "
    "`set_dofs_velocity`, `set_target_pos`, `set_torque`, or `apply_external_wrench` on them."
)
DEFORMABLE_OBSERVE_POLICY = (
    "For deformable bodies, observe deformable-friendly fields such as `pos`, `vel`, `bbox_min`, `bbox_max`, "
    "`bbox_size`, `vertex_disp_mean`, and `vertex_disp_max` instead of rigid-only pose/joint fields. Do not mix "
    "deformable and rigid bodies in the same multi-entity `observe` action, and do not use `include_contacts=true` "
    "on deformable bodies in v1."
)
DEFORMABLE_MESH_ASSET_POLICY = (
    "Deformable mesh bodies may use `generate_mesh_asset` outputs. When generating "
    "deformable mesh geometry, the mesh must be manifold-ready and suitable for FEM/PBD preprocessing; keep the shape "
    "simple, watertight, and thick enough for stable tetrahedralization, and leave positive clearance from the ground "
    "and nearby bodies at initialization."
)

COMPACT_HARD_RULE_KEYS = (
    "root_structure_note",
    "body_naming_policy",
    "articulated_body_xml_policy",
    "articulated_decision_policy",
    "mesh_body_policy",
    "mesh_decision_policy",
    "mesh_reuse_policy",
    "mesh_local_frame_policy",
    "mesh_scale_policy",
    "mesh_bbox_policy",
    "articulated_body_mesh_policy",
    "deformable_body_policy",
    "deformable_geometry_policy",
    "deformable_material_policy",
    "deformable_action_policy",
    "deformable_observe_policy",
    "deformable_mesh_asset_policy",
    "deformable_scene_policy",
    "fixed_body_note",
    "pre_sim_only_actions",
    "articulated_motion_policy",
    "ir_conciseness_policy",
    "dynamic_scene_policy",
)


def build_ir_agent_process_requirements(*, mesh_generation_available: bool) -> list[str]:
    lines = [
        f"- {ROOT_STRUCTURE_NOTE}",
        f"- {FIXED_BODY_NOTE}",
        "- Use `shape.kind='mesh'` only for non-articulated bodies. Mesh bodies should reference a mesh asset file and may be fixed or movable.",
        f"- {ARTICULATED_DECISION_POLICY}",
        f"- {MESH_DECISION_POLICY}",
        f"- {MESH_SCALE_POLICY}",
        f"- {MESH_BBOX_POLICY}",
        f"- {DEFORMABLE_BODY_POLICY}",
        f"- {DEFORMABLE_GEOMETRY_POLICY}",
        f"- {DEFORMABLE_MATERIAL_POLICY}",
        f"- {DEFORMABLE_ACTION_POLICY}",
        f"- {DEFORMABLE_OBSERVE_POLICY}",
        f"- {DEFORMABLE_MESH_ASSET_POLICY}",
        f"- {DEFORMABLE_SCENE_POLICY}",
        f"- {DYNAMIC_SCENE_POLICY}",
        "- `observe`, `set_pose`, and `apply_external_wrench` may target a single body or a list of body names via the `entity` field.",
        f"- {IR_CONCISENESS_POLICY}",
        "- Render is mandatory for generated IR; ensure scene.render is present.",
        f"- {BODY_COUNT_POLICY}",
        "- If articulated structure is needed and tool is available, call generate_articulated_xml once per articulated body.",
        "- Every generate_articulated_xml tool call must include the target `body_name`.",
    ]
    if mesh_generation_available:
        lines.extend(
            [
                "- If a non-articulated rigid or deformable body needs a generated mesh asset and the tool is available, call generate_mesh_asset once for that body.",
                "- Every generate_mesh_asset tool call must include the target `body_name`.",
                f"- {MESH_LOCAL_FRAME_POLICY}",
                "- If several non-articulated bodies intentionally share one geometry, use the same `reuse_key` when calling generate_mesh_asset so one generated asset can be reused.",
                "- If multiple mesh bodies need generated assets, batch those generate_mesh_asset tool calls in one response when possible.",
                "- Reuse the same generated mesh path across multiple bodies when they intentionally share the same object geometry (for example several identical crates or barriers). Do not regenerate duplicate mesh assets in that case.",
                "- If repeating the same mesh will not noticeably hurt the visual result, prefer reusing one generated mesh across multiple bodies instead of generating many near-duplicate meshes.",
            ]
        )
    else:
        lines.append("- Do not assume mesh generation is available unless the bootstrap says so.")
    lines.extend(
        [
            "- If multiple articulated bodies need XML generation, batch those generate_articulated_xml tool calls in one response when possible.",
            "- Each articulated body should use its own XML asset; do not reuse one XML path across unrelated articulated bodies.",
            "- If an articulated body's XML does not need to change in this revision, keep its existing xml_path instead of regenerating it.",
            f"- {ARTICULATED_BODY_MESH_POLICY}",
            "- Do not define actuators inside XML; define actuators only on the articulated body in `bodies[].actuators`.",
            "- Use `set_target_pos` only with position actuators and `set_torque` only with motor actuators.",
            "- If task specifies target simulation duration, pass it to validate_ir as target_sim_duration_sec.",
            "- Use validate_ir when it is helpful to check a draft before finalizing, but it is not mandatory before final output.",
            "- Return only final valid IR JSON.",
        ]
    )
    return lines
