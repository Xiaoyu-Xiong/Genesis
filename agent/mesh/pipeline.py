from __future__ import annotations

from dataclasses import replace
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..io_utils import dump_json
from .meshy_client import MeshyClient
from .models import (
    MeshManifoldCheckResult,
    MeshRepairConfig,
    MeshRepairResult,
    MeshyApiConfig,
    MeshyGenerationConfig,
    MeshyGenerationResult,
    TextToMeshBundle,
)
from .postprocess import build_ftetwild_retry_configs, repair_mesh_for_simulation
from .sanity import run_mesh_manifold_check


def default_mesh_output_dir(prompt: str, *, root: Path = Path("agent/generated_meshes")) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(prompt)
    return root / timestamp / slug


def generate_meshy_mesh_from_text(
    *,
    prompt: str,
    api_config: MeshyApiConfig,
    generation_config: MeshyGenerationConfig,
    repair_config: MeshRepairConfig | None = None,
) -> TextToMeshBundle:
    output_dir = generation_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_sec: dict[str, float] = {}
    total_start = time.monotonic()

    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(prompt.strip() + "\n", encoding="utf-8")

    client = MeshyClient(api_config)
    submit_response = _time_stage(profile_sec, "submit_request", lambda: client.submit_text_to_mesh(generation_config))
    submit_response_path = output_dir / "meshy_submit_response.json"
    dump_json(submit_response, submit_response_path)

    preview_task_id = _extract_preview_task_id(submit_response)
    final_response = _time_stage(
        profile_sec,
        "wait_preview",
        lambda: client.wait_for_preview_completion(
            preview_task_id=preview_task_id,
            poll_interval_sec=generation_config.poll_interval_sec,
            max_wait_sec=generation_config.max_wait_sec,
        ),
    )
    final_response_path = output_dir / "meshy_final_response.json"
    dump_json(final_response, final_response_path)

    mesh_path = _time_stage(
        profile_sec,
        "download_mesh",
        lambda: client.download_mesh(
            task_response=final_response,
            output_dir=output_dir,
            mesh_format=generation_config.mesh_format,
        ),
    )

    generation_result = MeshyGenerationResult(
        provider="meshy",
        prompt=prompt,
        output_dir=output_dir,
        mesh_path=mesh_path,
        prompt_path=prompt_path,
        submit_response_path=submit_response_path,
        final_response_path=final_response_path,
        metadata_path=output_dir / "metadata.json",
        preview_task_id=preview_task_id,
        final_status=str(final_response.get("status", "")),
        stage_durations_sec={
            "submit_request": profile_sec["submit_request"],
            "wait_preview": profile_sec["wait_preview"],
            "download_mesh": profile_sec["download_mesh"],
        },
        submit_response=submit_response,
        final_response=final_response,
    )

    raw_manifold_result = _time_stage(profile_sec, "raw_manifold_check", lambda: run_mesh_manifold_check(mesh_path))
    dump_json(raw_manifold_result.to_dict(), output_dir / "raw_manifold_check.json")

    repair_result, repair_attempts, final_manifold_result = _run_repair_pipeline(
        mesh_path=mesh_path,
        output_dir=output_dir,
        repair_config=repair_config,
        raw_manifold_result=raw_manifold_result,
        profile_sec=profile_sec,
    )

    dump_json(final_manifold_result.to_dict(), output_dir / "manifold_check.json")
    profile_sec["total"] = time.monotonic() - total_start
    dump_json(profile_sec, output_dir / "profile.json")

    metadata = {
        "provider": "meshy",
        "api_config": api_config.to_dict(),
        "generation_config": generation_config.to_dict(),
        "generation": generation_result.to_dict(),
        "raw_manifold": raw_manifold_result.to_dict(),
        "manifold": final_manifold_result.to_dict(),
        "profile_sec": profile_sec,
    }
    if repair_result is not None:
        metadata["repair"] = repair_result.to_dict()
    if repair_attempts:
        metadata["repair_attempts"] = [attempt.to_dict() for attempt in repair_attempts]
    dump_json(metadata, generation_result.metadata_path)

    return TextToMeshBundle(
        generation=generation_result,
        repair=repair_result,
        repair_attempts=tuple(repair_attempts),
        raw_manifold=raw_manifold_result,
        manifold=final_manifold_result,
        profile_sec=profile_sec,
    )


def parse_extra_payload(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    candidate = value.strip()
    if not candidate:
        return {}
    if candidate.startswith("{"):
        parsed = json.loads(candidate)
    else:
        path = Path(candidate)
        if path.exists():
            parsed = json.loads(path.read_text(encoding="utf-8"))
        else:
            parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Extra payload must be a JSON object.")
    return parsed


def _run_repair_pipeline(
    *,
    mesh_path: Path,
    output_dir: Path,
    repair_config: MeshRepairConfig | None,
    raw_manifold_result: MeshManifoldCheckResult,
    profile_sec: dict[str, float],
) -> tuple[MeshRepairResult | None, list[MeshRepairResult], MeshManifoldCheckResult]:
    if repair_config is None:
        return None, [], raw_manifold_result

    repair_attempts: list[MeshRepairResult] = []
    final_manifold_result = raw_manifold_result
    repair_result: MeshRepairResult | None = None
    repair_total_start = time.monotonic()
    final_manifold_elapsed = 0.0

    for attempt_index, (strategy_name, attempt_config) in enumerate(build_ftetwild_retry_configs(repair_config), start=1):
        attempt_result = _time_stage(
            profile_sec,
            f"repair_attempt_{attempt_index:02d}",
            lambda cfg=attempt_config, idx=attempt_index, name=strategy_name: repair_mesh_for_simulation(
                mesh_path,
                output_dir,
                cfg,
                attempt_index=idx,
                strategy_name=name,
            ),
        )
        repair_attempts.append(attempt_result)
        repair_result = attempt_result

        if not attempt_result.ok:
            final_manifold_result = MeshManifoldCheckResult(
                ok=False,
                mesh_path=attempt_result.output_mesh_path,
                vertex_count=0,
                face_count=0,
                component_count=0,
                is_watertight=False,
                is_winding_consistent=False,
                volume=None,
                error=f"Repair failed: {attempt_result.error}",
            )
            continue

        manifold_key = f"final_manifold_check_attempt_{attempt_index:02d}"
        attempt_manifold_result = _time_stage(
            profile_sec,
            manifold_key,
            lambda path=attempt_result.output_mesh_path: run_mesh_manifold_check(path),
        )
        final_manifold_elapsed += profile_sec[manifold_key]
        final_manifold_result = attempt_manifold_result
        if attempt_manifold_result.ok:
            repair_result, final_manifold_result = _canonicalize_successful_repair(
                mesh_path=mesh_path,
                output_dir=output_dir,
                repair_result=attempt_result,
                manifold_result=attempt_manifold_result,
            )
            repair_attempts[-1] = repair_result
            break

    profile_sec["repair_total"] = time.monotonic() - repair_total_start
    profile_sec["final_manifold_check"] = final_manifold_elapsed
    if repair_result is not None:
        dump_json(repair_result.to_dict(), output_dir / "repair.json")
    dump_json([attempt.to_dict() for attempt in repair_attempts], output_dir / "repair_attempts.json")
    return repair_result, repair_attempts, final_manifold_result


def _canonicalize_successful_repair(
    *,
    mesh_path: Path,
    output_dir: Path,
    repair_result: MeshRepairResult,
    manifold_result: MeshManifoldCheckResult,
) -> tuple[MeshRepairResult, MeshManifoldCheckResult]:
    canonical_repaired_path = output_dir / "processed" / f"repaired{mesh_path.suffix.lower()}"
    if repair_result.output_mesh_path == canonical_repaired_path:
        return repair_result, manifold_result
    shutil.copyfile(repair_result.output_mesh_path, canonical_repaired_path)
    return (
        replace(repair_result, output_mesh_path=canonical_repaired_path),
        replace(manifold_result, mesh_path=canonical_repaired_path),
    )


def _time_stage(profile_sec: dict[str, float], key: str, fn):
    stage_start = time.monotonic()
    result = fn()
    profile_sec[key] = time.monotonic() - stage_start
    return result


def _extract_preview_task_id(payload: dict[str, object]) -> str:
    result = payload.get("result")
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Meshy submit response did not contain a valid `result` task id.")
    return result.strip()


def _slugify(prompt: str) -> str:
    lowered = prompt.strip().lower()
    collapsed = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if not collapsed:
        return "mesh_prompt"
    return collapsed[:80]
