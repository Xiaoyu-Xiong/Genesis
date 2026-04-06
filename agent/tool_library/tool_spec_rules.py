from __future__ import annotations

ROOT_STRUCTURE_NOTE = "Use top-level `bodies` list."
BODY_COUNT_POLICY = "Multiple bodies are allowed, including multiple articulated bodies."
BODY_NAMING_POLICY = "Each body.name must be unique. Actions refer to bodies through the `entity` field."
FIXED_BODY_NOTE = (
    "Use `bodies[].fixed=true` for fixed primitive, mesh, or URDF objects such as obstacles, tables, and targets. "
    "For MJCF bodies, encode a fixed base inside the XML itself."
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
    "and reuse the same `shape.file` across those bodies."
)
MESH_LOCAL_FRAME_POLICY = (
    "Generated mesh assets are post-processed so their geometric centroid is at the local origin. "
    "Use the returned axis-aligned bounding box to choose sensible world placement and scale."
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

COMPACT_HARD_RULE_KEYS = (
    "root_structure_note",
    "body_naming_policy",
    "articulated_body_xml_policy",
    "articulated_decision_policy",
    "mesh_body_policy",
    "mesh_decision_policy",
    "mesh_reuse_policy",
    "mesh_local_frame_policy",
    "articulated_body_mesh_policy",
    "fixed_body_note",
    "pre_sim_only_actions",
    "articulated_motion_policy",
    "ir_conciseness_policy",
    "fixed_parameter_override_policy",
)


def build_ir_agent_process_requirements(*, mesh_generation_available: bool) -> list[str]:
    lines = [
        f"- {ROOT_STRUCTURE_NOTE}",
        f"- {FIXED_BODY_NOTE}",
        "- Use `shape.kind='mesh'` only for non-articulated bodies. Mesh bodies should reference a mesh asset file and may be fixed or movable.",
        f"- {ARTICULATED_DECISION_POLICY}",
        f"- {MESH_DECISION_POLICY}",
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
                "- If a non-articulated body needs a generated mesh asset and tool is available, call generate_mesh_asset once for that body.",
                "- Every generate_mesh_asset tool call must include the target `body_name`.",
                f"- {MESH_LOCAL_FRAME_POLICY}",
                "- If several non-articulated bodies intentionally share one geometry, use the same `reuse_key` when calling generate_mesh_asset so one generated asset can be reused.",
                "- If multiple mesh bodies need generated assets, batch those generate_mesh_asset tool calls in one response when possible.",
                "- Reuse the same generated mesh path across multiple bodies when they intentionally share the same object geometry (for example several identical crates or barriers). Do not regenerate duplicate mesh assets in that case.",
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
