from __future__ import annotations

from .generation import (
    build_generation_bootstrap_payload,
    build_generation_guide_payload,
    build_observation_field_guide_payload,
    build_schema_payload,
)
from ..constraints.rules import (
    ARTICULATED_BODY_MESH_POLICY,
    ARTICULATED_BODY_XML_POLICY,
    ARTICULATED_DECISION_POLICY,
    BODY_COUNT_POLICY,
    BODY_NAMING_POLICY,
    COMPACT_HARD_RULE_KEYS,
    DYNAMIC_SCENE_POLICY,
    FIXED_BODY_NOTE,
    IR_CONCISENESS_POLICY,
    MESH_BODY_POLICY,
    MESH_DECISION_POLICY,
    MESH_LOCAL_FRAME_POLICY,
    MESH_REUSE_POLICY,
    ROOT_STRUCTURE_NOTE,
    build_ir_agent_process_requirements,
)


def build_tool_specs(*, xml_generation_enabled: bool, mesh_generation_enabled: bool = False) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {
                "name": "get_generation_bootstrap",
                "description": (
                    "Return the main rigid-scene generation bootstrap context: generation guide, observation field guide, "
                    "and full JSON schema."
                ),
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "validate_ir",
                "description": "Validate candidate IR and return normalized IR if valid.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidate_ir": {
                            "type": "object",
                            "description": "Candidate RigidIR JSON object.",
                        },
                        "normalize": {
                            "type": "boolean",
                            "description": "Whether to quaternion-normalize before returning.",
                            "default": True,
                        },
                        "target_sim_duration_sec": {
                            "type": "number",
                            "description": (
                                "Optional target simulation duration in seconds "
                                "(compared against the resulting program duration under the system simulation timestep)."
                            ),
                        },
                        "sim_duration_tolerance_sec": {
                            "type": "number",
                            "description": "Optional allowed absolute simulation-duration error in seconds.",
                        },
                    },
                    "required": ["candidate_ir"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    if xml_generation_enabled:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": "generate_articulated_xml",
                    "description": "Generate MJCF XML for one articulated body and return the asset path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "body_name": {
                                "type": "string",
                                "description": "Name of the articulated body this XML belongs to.",
                            },
                            "xml_task": {
                                "type": "string",
                                "description": "Optional specialized XML generation instruction.",
                            },
                            "file_stem": {
                                "type": "string",
                                "description": "Optional output filename stem (without .xml).",
                            },
                        },
                        "required": ["body_name"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    if mesh_generation_enabled:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": "generate_mesh_asset",
                    "description": "Generate a non-articulated mesh asset for one body and return the mesh file path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "body_name": {
                                "type": "string",
                                "description": "Name of the non-articulated mesh body this asset belongs to.",
                            },
                            "mesh_task": {
                                "type": "string",
                                "description": "Optional specialized mesh-generation instruction.",
                            },
                            "file_stem": {
                                "type": "string",
                                "description": "Optional output filename stem (without extension).",
                            },
                            "reuse_key": {
                                "type": "string",
                                "description": (
                                    "Optional shared-geometry key. If multiple bodies intentionally share the same mesh "
                                    "geometry, call the tool with the same reuse_key so the asset can be reused."
                                ),
                            },
                        },
                        "required": ["body_name"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    return specs


__all__ = [
    "ROOT_STRUCTURE_NOTE",
    "BODY_COUNT_POLICY",
    "BODY_NAMING_POLICY",
    "DYNAMIC_SCENE_POLICY",
    "FIXED_BODY_NOTE",
    "ARTICULATED_BODY_XML_POLICY",
    "ARTICULATED_DECISION_POLICY",
    "MESH_BODY_POLICY",
    "MESH_DECISION_POLICY",
    "MESH_REUSE_POLICY",
    "MESH_LOCAL_FRAME_POLICY",
    "ARTICULATED_BODY_MESH_POLICY",
    "IR_CONCISENESS_POLICY",
    "COMPACT_HARD_RULE_KEYS",
    "build_ir_agent_process_requirements",
    "build_tool_specs",
    "build_generation_guide_payload",
    "build_observation_field_guide_payload",
    "build_schema_payload",
    "build_generation_bootstrap_payload",
]
