from __future__ import annotations

from dataclasses import replace
import re
import shutil
import time
from pathlib import Path

from ..io_utils import dump_json
from .meshy_client import MeshyClient
from .models import (
    MeshManifoldCheckResult,
    MeshRepairConfig,
    MeshRepairResult,
    MeshyGenerationConfig,
    MeshyTextureConfig,
    MeshyTextureResult,
)
from .postprocess import build_ftetwild_retry_configs, repair_mesh_for_simulation
from .sanity import run_mesh_manifold_check


def run_repair_pipeline(
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

    for attempt_index, (strategy_name, attempt_config) in enumerate(
        build_ftetwild_retry_configs(repair_config),
        start=1,
    ):
        attempt_result = time_stage(
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
        attempt_manifold_result = time_stage(
            profile_sec,
            manifold_key,
            lambda path=attempt_result.output_mesh_path: run_mesh_manifold_check(path),
        )
        final_manifold_elapsed += profile_sec[manifold_key]
        final_manifold_result = attempt_manifold_result
        if attempt_manifold_result.ok:
            repair_result, final_manifold_result = canonicalize_successful_repair(
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


def canonicalize_successful_repair(
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


def run_texture_pipeline(
    *,
    client: MeshyClient,
    output_dir: Path,
    preview_task_id: str,
    prompt: str,
    generation_config: MeshyGenerationConfig,
    texture_config: MeshyTextureConfig | None,
    profile_sec: dict[str, float],
) -> MeshyTextureResult | None:
    if texture_config is None or not texture_config.enabled:
        return None

    texture_prompt = texture_config.texture_prompt.strip() if texture_config.texture_prompt else prompt.strip()
    ai_model = texture_config.ai_model or generation_config.ai_model
    submit_response_path = output_dir / "meshy_texture_submit_response.json"
    final_response_path = output_dir / "meshy_texture_final_response.json"
    stage_durations_sec: dict[str, float] = {}

    try:
        submit_response = time_stage(
            profile_sec,
            "submit_texture_refine",
            lambda: client.submit_text_to_texture_refine(
                preview_task_id=preview_task_id,
                texture_prompt=texture_prompt,
                ai_model=ai_model,
                enable_pbr=texture_config.enable_pbr,
                moderation=generation_config.moderation,
                remove_lighting=texture_config.remove_lighting,
            ),
        )
        stage_durations_sec["submit_texture_refine"] = profile_sec["submit_texture_refine"]
        dump_json(submit_response, submit_response_path)

        refine_task_id = extract_task_id(submit_response)
        final_response = time_stage(
            profile_sec,
            "wait_texture_refine",
            lambda: client.wait_for_text_to_3d_completion(
                task_id=refine_task_id,
                poll_interval_sec=generation_config.poll_interval_sec,
                max_wait_sec=generation_config.max_wait_sec,
                stage_label="refine",
            ),
        )
        stage_durations_sec["wait_texture_refine"] = profile_sec["wait_texture_refine"]
        dump_json(final_response, final_response_path)

        textured_mesh_path = time_stage(
            profile_sec,
            "download_textured_mesh",
            lambda: client.download_mesh(
                task_response=final_response,
                output_dir=output_dir,
                mesh_format="obj",
                subdir="textured",
            ),
        )
        stage_durations_sec["download_textured_mesh"] = profile_sec["download_textured_mesh"]

        texture_paths = time_stage(
            profile_sec,
            "download_texture_maps",
            lambda: client.download_texture_maps(
                task_response=final_response,
                output_dir=output_dir,
                subdir="textured",
            ),
        )
        stage_durations_sec["download_texture_maps"] = profile_sec["download_texture_maps"]

        textured_dir = output_dir / "textured"
        textured_mtl_path = textured_dir / "model.mtl"
        base_color_path = texture_paths.get("base_color")
        if base_color_path is None:
            raise RuntimeError("Texture refine succeeded but no base_color texture was returned.")
        rewrite_obj_mtllib(textured_mesh_path, mtl_name="model.mtl")
        rewrite_mtl_base_color(textured_mtl_path, texture_name=base_color_path.name)

        return MeshyTextureResult(
            requested=True,
            ok=True,
            prompt=texture_prompt,
            output_dir=textured_dir,
            preview_task_id=preview_task_id,
            refine_task_id=refine_task_id,
            submit_response_path=submit_response_path,
            final_response_path=final_response_path,
            textured_mesh_path=textured_mesh_path,
            textured_mtl_path=textured_mtl_path if textured_mtl_path.exists() else None,
            texture_paths=texture_paths,
            ai_model=ai_model,
            enable_pbr=texture_config.enable_pbr,
            remove_lighting=texture_config.remove_lighting,
            final_status=str(final_response.get("status", "")),
            stage_durations_sec=stage_durations_sec,
            submit_response=submit_response,
            final_response=final_response,
        )
    except Exception as exc:  # noqa: BLE001
        return MeshyTextureResult(
            requested=True,
            ok=False,
            prompt=texture_prompt,
            output_dir=output_dir / "textured",
            preview_task_id=preview_task_id,
            refine_task_id=None,
            submit_response_path=submit_response_path if submit_response_path.exists() else None,
            final_response_path=final_response_path if final_response_path.exists() else None,
            textured_mesh_path=None,
            textured_mtl_path=None,
            texture_paths={},
            ai_model=ai_model,
            enable_pbr=texture_config.enable_pbr,
            remove_lighting=texture_config.remove_lighting,
            final_status=None,
            stage_durations_sec=stage_durations_sec,
            error=f"{type(exc).__name__}: {exc}",
        )


def select_pipeline_source_mesh(
    *,
    raw_mesh_path: Path,
    texture_config: MeshyTextureConfig | None,
    texture_result: MeshyTextureResult | None,
) -> tuple[Path, str]:
    if texture_config is not None and texture_config.enabled:
        if texture_result is not None and texture_result.ok and texture_result.textured_mesh_path is not None:
            return texture_result.textured_mesh_path, "textured_raw_obj"
        return raw_mesh_path, "preview_raw_fallback_after_texture_failure"
    return raw_mesh_path, "preview_raw_obj"


def time_stage(profile_sec: dict[str, float], key: str, fn):
    stage_start = time.monotonic()
    result = fn()
    profile_sec[key] = time.monotonic() - stage_start
    return result


def extract_preview_task_id(payload: dict[str, object]) -> str:
    return extract_task_id(payload)


def extract_task_id(payload: dict[str, object]) -> str:
    result = payload.get("result")
    if not isinstance(result, str) or not result.strip():
        raise ValueError("Meshy submit response did not contain a valid `result` task id.")
    return result.strip()


def rewrite_obj_mtllib(obj_path: Path, *, mtl_name: str) -> None:
    text = obj_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    replaced = False
    rewritten_lines: list[str] = []
    for line in lines:
        if line.startswith("mtllib "):
            rewritten_lines.append(f"mtllib {mtl_name}")
            replaced = True
        else:
            rewritten_lines.append(line)
    if not replaced:
        rewritten_lines.insert(0, f"mtllib {mtl_name}")
    obj_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")


def rewrite_mtl_base_color(mtl_path: Path, *, texture_name: str) -> None:
    if not mtl_path.exists():
        raise FileNotFoundError(f"Expected MTL file not found: {mtl_path}")

    lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rewritten_lines: list[str] = []
    saw_map_kd = False
    saw_newmtl = False
    inserted_in_current_block = False

    for line in lines:
        if line.startswith("newmtl "):
            if saw_newmtl and not inserted_in_current_block:
                rewritten_lines.append(f"map_Kd {texture_name}")
            rewritten_lines.append(line)
            saw_newmtl = True
            inserted_in_current_block = False
            continue
        if line.startswith("map_Kd "):
            rewritten_lines.append(f"map_Kd {texture_name}")
            saw_map_kd = True
            inserted_in_current_block = True
            continue
        rewritten_lines.append(line)

    if saw_newmtl and not inserted_in_current_block:
        rewritten_lines.append(f"map_Kd {texture_name}")
        saw_map_kd = True
    if not saw_newmtl and not saw_map_kd:
        rewritten_lines.extend(["newmtl material_0", f"map_Kd {texture_name}"])

    mtl_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")


def augment_meshy_geometry_prompt(prompt: str) -> str:
    base_prompt = prompt.strip()
    manifold_requirement = (
        "The mesh should be a single closed watertight manifold surface with no open boundaries, no missing patches, "
        "no self-intersections, no non-manifold edges or vertices, and no large unrecoverable holes. Avoid thin open "
        "shells, floating fragments, and underresolved gaps; prioritize simulation-ready geometric integrity over fine detail."
    )
    lowered = base_prompt.lower()
    manifold_markers = (
        "watertight",
        "manifold",
        "no self-intersections",
        "non-manifold",
        "open boundaries",
        "holes",
    )
    if any(marker in lowered for marker in manifold_markers):
        return base_prompt
    return f"{base_prompt}\n\n{manifold_requirement}"


def slugify_prompt(prompt: str) -> str:
    lowered = prompt.strip().lower()
    collapsed = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    if not collapsed:
        return "mesh_prompt"
    return collapsed[:80]
