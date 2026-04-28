from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from ...configs import CONFIGS
from ...mesh.models import (
    MeshRepairConfig,
    MeshyApiConfig,
    MeshyGenerationConfig,
    MeshyTextureConfig,
    TextToMeshBundle,
)
from ...mesh.pipeline import generate_meshy_mesh_from_text


@dataclass(slots=True)
class MeshGenerationAttemptLog:
    attempt: int
    output_dir: str
    raw_manifold_ok: bool
    repaired_manifold_ok: bool
    profile_sec: dict[str, float]
    error: str | None


@dataclass(slots=True)
class MeshGenerationResult:
    provider: str
    attempts: int
    mesh_path: str
    raw_manifold_ok: bool
    repaired_manifold_ok: bool
    texture_requested: bool
    texture_succeeded: bool
    textured_mesh_path: str | None
    textured_mtl_path: str | None
    base_color_path: str | None
    centroid_before_translation: tuple[float, float, float] | None
    bbox_min: tuple[float, float, float] | None
    bbox_max: tuple[float, float, float] | None
    bbox_size: tuple[float, float, float] | None
    centroid_at_origin: bool
    logs: list[MeshGenerationAttemptLog]


def generate_mesh_asset_with_meshy(
    *,
    task: str,
    output_dir: str | Path,
    file_stem: str,
    mesh_format: str | None = None,
    texture_enabled: bool | None = None,
    timeout_sec: float | None = None,
    api_key_env: str | None = None,
    base_url_env: str | None = None,
) -> MeshGenerationResult:
    request_defaults = CONFIGS.meshy_request
    repair_defaults = CONFIGS.mesh_repair
    output_dir = Path(output_dir) / file_stem
    api_config = MeshyApiConfig.from_env(
        api_key_env=api_key_env or "MESHY_API_KEY",
        base_url_env=base_url_env or "MESHY_API_BASE_URL",
        timeout_sec=request_defaults.timeout_sec if timeout_sec is None else timeout_sec,
    )
    generation_config = MeshyGenerationConfig(
        prompt=task,
        output_dir=output_dir,
        mesh_format=request_defaults.mesh_format if mesh_format is None else mesh_format,
        ai_model=request_defaults.ai_model,
        art_style=request_defaults.art_style,
        should_remesh=request_defaults.should_remesh,
        topology=request_defaults.topology,
        target_polycount=request_defaults.target_polycount,
        symmetry_mode=request_defaults.symmetry_mode,
        moderation=request_defaults.moderation,
        negative_prompt=request_defaults.negative_prompt,
        auto_size=request_defaults.auto_size,
        origin_at=request_defaults.origin_at,
        poll_interval_sec=request_defaults.poll_interval_sec,
        max_wait_sec=request_defaults.max_wait_sec,
    )
    texture_config = MeshyTextureConfig(
        enabled=request_defaults.texture_enabled if texture_enabled is None else texture_enabled,
        ai_model=request_defaults.texture_ai_model,
        enable_pbr=request_defaults.texture_enable_pbr,
        remove_lighting=request_defaults.texture_remove_lighting,
    )
    repair_config = MeshRepairConfig(
        component_count_face_cap=repair_defaults.component_count_face_cap,
        min_component_faces=repair_defaults.min_component_faces,
        max_repair_attempts=repair_defaults.max_repair_attempts,
        merge_vertices=repair_defaults.merge_vertices,
        merge_digits_vertex=repair_defaults.merge_digits_vertex,
        fix_normals=repair_defaults.fix_normals,
        process_validate=repair_defaults.process_validate,
        keep_largest_component=repair_defaults.keep_largest_component,
        ftetwild_edge_length_fac=repair_defaults.ftetwild_edge_length_fac,
        ftetwild_edge_length_abs=repair_defaults.ftetwild_edge_length_abs,
        ftetwild_optimize=repair_defaults.ftetwild_optimize,
        ftetwild_simplify=repair_defaults.ftetwild_simplify,
        ftetwild_epsilon=repair_defaults.ftetwild_epsilon,
        ftetwild_stop_energy=repair_defaults.ftetwild_stop_energy,
        ftetwild_coarsen=repair_defaults.ftetwild_coarsen,
        ftetwild_num_threads=repair_defaults.ftetwild_num_threads,
        ftetwild_num_opt_iter=repair_defaults.ftetwild_num_opt_iter,
        ftetwild_quiet=repair_defaults.ftetwild_quiet,
        ftetwild_disable_filtering=repair_defaults.ftetwild_disable_filtering,
    )

    bundle = generate_meshy_mesh_from_text(
        prompt=task,
        api_config=api_config,
        generation_config=generation_config,
        texture_config=texture_config,
        repair_config=repair_config,
    )
    return _bundle_to_result(bundle)


def load_existing_mesh_generation_result(mesh_path: str | Path) -> MeshGenerationResult | None:
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        return None
    asset_root = _asset_root_from_mesh_path(mesh_path)
    metadata = _load_json_dict(asset_root / "metadata.json")
    if metadata is None:
        return None
    raw_payload = _load_json_dict(asset_root / "raw_manifold_check.json") or {}
    manifold_payload = _load_json_dict(asset_root / "manifold_check.json") or {}
    repair_payload = metadata.get("repair") if isinstance(metadata.get("repair"), dict) else {}
    texture_payload = metadata.get("texture") if isinstance(metadata.get("texture"), dict) else {}

    provider = metadata.get("provider")
    if not isinstance(provider, str):
        provider = "meshy"

    raw_ok = bool(raw_payload.get("ok"))
    repaired_ok = bool(manifold_payload.get("ok"))
    return MeshGenerationResult(
        provider=provider,
        attempts=0,
        mesh_path=str(mesh_path),
        raw_manifold_ok=raw_ok,
        repaired_manifold_ok=repaired_ok,
        texture_requested=bool(texture_payload.get("requested", False)),
        texture_succeeded=bool(texture_payload.get("ok", False)),
        textured_mesh_path=_str_or_none(texture_payload.get("textured_mesh_path")),
        textured_mtl_path=_str_or_none(texture_payload.get("textured_mtl_path")),
        base_color_path=_str_or_none((texture_payload.get("texture_paths") or {}).get("base_color"))
        if isinstance(texture_payload.get("texture_paths"), dict)
        else None,
        centroid_before_translation=_tuple3(repair_payload.get("centroid_before_translation")),
        bbox_min=_tuple3(repair_payload.get("bbox_min")),
        bbox_max=_tuple3(repair_payload.get("bbox_max")),
        bbox_size=_tuple3(repair_payload.get("bbox_size")),
        centroid_at_origin=bool(repair_payload.get("centroid_at_origin", False)),
        logs=[
            MeshGenerationAttemptLog(
                attempt=0,
                output_dir=asset_root.as_posix(),
                raw_manifold_ok=raw_ok,
                repaired_manifold_ok=repaired_ok,
                profile_sec={},
                error=None if repaired_ok else "reused asset did not pass manifold check",
            )
        ],
    )


def _bundle_to_result(bundle: TextToMeshBundle) -> MeshGenerationResult:
    raw_ok = bool(bundle.raw_manifold and bundle.raw_manifold.ok)
    repaired_ok = bool(bundle.manifold and bundle.manifold.ok)
    mesh_path = bundle.generation.mesh_path
    if bundle.repair is not None and bundle.repair.ok:
        mesh_path = str(bundle.repair.output_mesh_path)
    logs = _build_attempt_logs(bundle)
    return MeshGenerationResult(
        provider=bundle.generation.provider,
        attempts=max(1, len(logs)),
        mesh_path=mesh_path,
        raw_manifold_ok=raw_ok,
        repaired_manifold_ok=repaired_ok,
        texture_requested=bundle.texture is not None and bundle.texture.requested,
        texture_succeeded=bool(bundle.texture and bundle.texture.ok),
        textured_mesh_path=(
            None
            if bundle.texture is None or bundle.texture.textured_mesh_path is None
            else str(bundle.texture.textured_mesh_path)
        ),
        textured_mtl_path=(
            None
            if bundle.texture is None or bundle.texture.textured_mtl_path is None
            else str(bundle.texture.textured_mtl_path)
        ),
        base_color_path=(
            None
            if bundle.texture is None or "base_color" not in bundle.texture.texture_paths
            else str(bundle.texture.texture_paths["base_color"])
        ),
        centroid_before_translation=None if bundle.repair is None else bundle.repair.centroid_before_translation,
        bbox_min=None if bundle.repair is None else bundle.repair.bbox_min,
        bbox_max=None if bundle.repair is None else bundle.repair.bbox_max,
        bbox_size=None if bundle.repair is None else bundle.repair.bbox_size,
        centroid_at_origin=bool(bundle.repair and bundle.repair.centroid_at_origin),
        logs=logs,
    )


def _asset_root_from_mesh_path(mesh_path: Path) -> Path:
    if mesh_path.parent.name in {"downloads", "processed", "textured"}:
        return mesh_path.parent.parent
    return mesh_path.parent


def _build_attempt_logs(bundle: TextToMeshBundle) -> list[MeshGenerationAttemptLog]:
    if not bundle.repair_attempts:
        return [
            MeshGenerationAttemptLog(
                attempt=1,
                output_dir=bundle.generation.output_dir.as_posix(),
                raw_manifold_ok=bool(bundle.raw_manifold and bundle.raw_manifold.ok),
                repaired_manifold_ok=bool(bundle.manifold and bundle.manifold.ok),
                profile_sec=bundle.profile_sec or {},
                error=None if bundle.manifold is None else bundle.manifold.error,
            )
        ]

    successful_attempt = (
        bundle.repair.attempt_index
        if bundle.repair is not None and bundle.manifold is not None and bundle.manifold.ok
        else None
    )
    logs: list[MeshGenerationAttemptLog] = []
    for attempt in bundle.repair_attempts:
        logs.append(
            MeshGenerationAttemptLog(
                attempt=attempt.attempt_index,
                output_dir=attempt.output_mesh_path.parent.parent.as_posix(),
                raw_manifold_ok=bool(bundle.raw_manifold and bundle.raw_manifold.ok),
                repaired_manifold_ok=attempt.ok and attempt.attempt_index == successful_attempt,
                profile_sec=attempt.stage_durations_sec,
                error=attempt.error,
            )
        )
    if bundle.manifold is not None and not bundle.manifold.ok and logs:
        logs[-1].error = bundle.manifold.error
    return logs


def _load_json_dict(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _tuple3(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    converted: list[float] = []
    for component in value:
        if not isinstance(component, int | float) or isinstance(component, bool):
            return None
        converted.append(float(component))
    return tuple(converted)


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
