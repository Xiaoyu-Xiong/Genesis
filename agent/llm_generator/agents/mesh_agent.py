from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from ...mesh import MeshRepairConfig, MeshyApiConfig, MeshyGenerationConfig, TextToMeshBundle, generate_meshy_mesh_from_text


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
    mesh_format: str = "obj",
    timeout_sec: float = 1000.0,
    api_key_env: str = "MESHY_API_KEY",
    base_url_env: str = "MESHY_API_BASE_URL",
) -> MeshGenerationResult:
    output_dir = Path(output_dir) / file_stem
    api_config = MeshyApiConfig.from_env(
        api_key_env=api_key_env,
        base_url_env=base_url_env,
        timeout_sec=timeout_sec,
    )
    generation_config = MeshyGenerationConfig(
        prompt=task,
        output_dir=output_dir,
        mesh_format=mesh_format,
        should_remesh=True,
        target_polycount=5000,
    )
    repair_config = MeshRepairConfig()

    bundle = generate_meshy_mesh_from_text(
        prompt=task,
        api_config=api_config,
        generation_config=generation_config,
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
        centroid_before_translation=None if bundle.repair is None else bundle.repair.centroid_before_translation,
        bbox_min=None if bundle.repair is None else bundle.repair.bbox_min,
        bbox_max=None if bundle.repair is None else bundle.repair.bbox_max,
        bbox_size=None if bundle.repair is None else bundle.repair.bbox_size,
        centroid_at_origin=bool(bundle.repair and bundle.repair.centroid_at_origin),
        logs=logs,
    )


def _asset_root_from_mesh_path(mesh_path: Path) -> Path:
    if mesh_path.parent.name in {"downloads", "processed"}:
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
