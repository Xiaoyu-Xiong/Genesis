from __future__ import annotations

from typing import Any

from code_agent.assets.mesh.request_adapter import positive_vector3, request_size


XML_ASSET_TYPES = {
    "mjcf",
    "generated_xml",
}


def select_xml_requests(
    planner_output: dict[str, Any],
    asset_names: list[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_requests = planner_output.get("asset_requests")
    requests = [item for item in raw_requests if isinstance(item, dict)] if isinstance(raw_requests, list) else []
    requested_names = {name for name in asset_names or [] if name}
    selected: list[dict[str, Any]] = []
    found_names: set[str] = set()
    for request in requests:
        name = str(request.get("name", ""))
        if requested_names and name not in requested_names:
            continue
        found_names.add(name)
        if is_xml_asset_request(request):
            selected.append(request)
    selected_names = {str(item.get("name", "")) for item in selected}
    skipped_names = sorted(requested_names - selected_names - {""})
    skipped_names.extend(
        sorted(
            name
            for name in found_names
            if name in requested_names and all(str(item.get("name", "")) != name for item in selected)
        )
    )
    return selected, sorted(set(skipped_names))


def is_xml_asset_request(request: dict[str, Any]) -> bool:
    asset_type = str(request.get("asset_type", "")).strip().lower()
    return asset_type in XML_ASSET_TYPES


def xml_prompt_from_request(request: dict[str, Any], task: str) -> str:
    name = str(request.get("name", "xml_asset")).replace("_", " ")
    purpose = _clean_prompt_field(request.get("purpose"))
    simulation_role = _clean_prompt_field(request.get("simulation_role"))
    texture_needs = _clean_prompt_field(request.get("texture_needs"))
    parts = [
        f"Episode task: {task}",
        f"Generate one articulated MJCF XML asset: {name}.",
        f"Purpose: {purpose}.",
        f"Simulation role: {simulation_role}.",
    ]
    size = request_size(request)
    if size is not None:
        parts.append(
            "Approximate positive XYZ dimensions in meters to bake into primitive MJCF geometry: "
            f"{size}. Do not rely on the scene writer to scale the XML later."
        )
    if texture_needs:
        parts.append(
            "The Planner attached these visual/material needs for context only: "
            f"{texture_needs}. Use primitive MJCF materials/colors if helpful, but do not reference image textures, "
            "mesh assets, or hfields in the XML."
        )
    parts.append(
        "Expose named joints and named actuators with clear command ranges so the Action Worker can control the asset."
    )
    parts.append(
        "Keep the XML asset self-contained: one articulated body tree only, no scene ground, cameras, lights, arenas, "
        "projectiles, or global simulation options."
    )
    return _clean_prompt_field("\n".join(part for part in parts if part))


def xml_requested_bbox(request: dict[str, Any]) -> list[float] | None:
    return positive_vector3(request.get("bbox")) or positive_vector3(request.get("scale"))


def _clean_prompt_field(value: object) -> str:
    return " ".join(str(value or "").split())
