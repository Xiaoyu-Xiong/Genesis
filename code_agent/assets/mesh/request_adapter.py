from __future__ import annotations

from pathlib import Path
from typing import Any

from code_agent.assets.mesh.models import MeshRepairConfig, MeshyGenerationConfig, MeshyTextureConfig
from code_agent.configs import CONFIGS


MESH_ASSET_TYPES = {"generated_mesh"}


def select_mesh_requests(
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
        if is_mesh_asset_request(request):
            selected.append(request)
    skipped_names = sorted(requested_names - {str(item.get("name", "")) for item in selected} - {""})
    skipped_names.extend(
        sorted(
            name
            for name in found_names
            if name in requested_names and all(str(item.get("name", "")) != name for item in selected)
        )
    )
    return selected, sorted(set(skipped_names))


def is_mesh_asset_request(request: dict[str, Any]) -> bool:
    asset_type = str(request.get("asset_type", "")).strip().lower()
    return asset_type in MESH_ASSET_TYPES


def meshy_generation_config(prompt: str, output_dir: Path) -> MeshyGenerationConfig:
    defaults = CONFIGS.meshy_request
    return MeshyGenerationConfig(
        prompt=prompt,
        output_dir=output_dir,
        prompt_max_chars=defaults.prompt_max_chars,
        mesh_format=defaults.mesh_format,
        ai_model=defaults.ai_model,
        art_style=defaults.art_style,
        should_remesh=defaults.should_remesh,
        topology=defaults.topology,
        target_polycount=defaults.target_polycount,
        symmetry_mode=defaults.symmetry_mode,
        moderation=defaults.moderation,
        negative_prompt=defaults.negative_prompt,
        auto_size=defaults.auto_size,
        origin_at=defaults.origin_at,
        poll_interval_sec=defaults.poll_interval_sec,
        max_wait_sec=defaults.max_wait_sec,
    )


def meshy_texture_config(request: dict[str, Any]) -> MeshyTextureConfig | None:
    defaults = CONFIGS.meshy_request
    texture_needs = request.get("texture_needs")
    if not defaults.texture_enabled or texture_needs is None:
        return None
    texture_prompt = str(texture_needs).strip()
    return MeshyTextureConfig(
        enabled=bool(texture_prompt),
        texture_prompt=texture_prompt or None,
        ai_model=defaults.texture_ai_model,
        enable_pbr=defaults.texture_enable_pbr,
        remove_lighting=defaults.texture_remove_lighting,
    )


def mesh_repair_config() -> MeshRepairConfig:
    defaults = CONFIGS.mesh_repair
    return MeshRepairConfig(
        component_count_face_cap=defaults.component_count_face_cap,
        min_component_faces=defaults.min_component_faces,
        max_repair_attempts=defaults.max_repair_attempts,
        merge_vertices=defaults.merge_vertices,
        merge_digits_vertex=defaults.merge_digits_vertex,
        fix_normals=defaults.fix_normals,
        process_validate=defaults.process_validate,
        keep_largest_component=defaults.keep_largest_component,
        ftetwild_edge_length_fac=defaults.ftetwild_edge_length_fac,
        ftetwild_edge_length_abs=defaults.ftetwild_edge_length_abs,
        ftetwild_optimize=defaults.ftetwild_optimize,
        ftetwild_simplify=defaults.ftetwild_simplify,
        ftetwild_epsilon=defaults.ftetwild_epsilon,
        ftetwild_stop_energy=defaults.ftetwild_stop_energy,
        ftetwild_coarsen=defaults.ftetwild_coarsen,
        ftetwild_num_threads=defaults.ftetwild_num_threads,
        ftetwild_num_opt_iter=defaults.ftetwild_num_opt_iter,
        ftetwild_quiet=defaults.ftetwild_quiet,
        ftetwild_disable_filtering=defaults.ftetwild_disable_filtering,
    )


def mesh_prompt_from_request(request: dict[str, Any], task: str) -> str:
    _ = task
    name = str(request.get("name", "mesh_asset")).replace("_", " ")
    purpose = _clean_prompt_field(request.get("purpose"))
    simulation_role = _clean_prompt_field(request.get("simulation_role"))
    texture_needs = _clean_prompt_field(request.get("texture_needs"))
    parts = [
        f"Create one simulation-ready 3D mesh: {name}.",
        purpose,
        f"Role: {simulation_role}.",
    ]
    size = request_size(request)
    if size is not None:
        parts.append(f"Approximate positive dimensions in meters: {size}.")
    if texture_needs:
        parts.append(f"Material: {texture_needs}.")
    parts.append("Keep one coherent object with clear silhouette and robust closed surfaces for physics.")
    return _clean_prompt_field(" ".join(part for part in parts if part))


def request_size(request: dict[str, Any]) -> list[float] | None:
    return positive_vector3(request.get("scale")) or positive_vector3(request.get("bbox"))


def vector3(value: object) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    output: list[float] = []
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool):
            return None
        output.append(float(item))
    return output


def positive_vector3(value: object) -> list[float] | None:
    vector = vector3(value)
    if vector is None:
        return None
    if any(item <= 0.0 for item in vector):
        return None
    return vector


def _clean_prompt_field(value: object) -> str:
    return " ".join(str(value or "").split())
