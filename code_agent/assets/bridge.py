from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from code_agent.assets.mesh.models import MeshRepairConfig, MeshyApiConfig, MeshyGenerationConfig, MeshyTextureConfig
from code_agent.assets.mesh.pipeline import DownloadedMeshyAsset, download_meshy_mesh_from_text, process_downloaded_meshy_mesh
from code_agent.assets.mesh.workflow.steps import slugify_prompt
from code_agent.assets.mesh.workflow.summary import load_mesh_asset_summary
from code_agent.configs import CONFIGS
from code_agent.io_utils import dump_json


MESH_ASSET_TYPES = {"mesh", "generated_mesh", "text_to_mesh", "text-to-mesh", "meshy"}


def generate_mesh_assets_for_episode(
    *,
    case_dir: Path,
    task: str,
    planner_output: dict[str, Any],
    asset_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run Planner-requested text-to-mesh assets and write the episode asset manifest."""

    assets_dir = case_dir / "assets"
    manifest_path = assets_dir / "asset_manifest.json"
    report_path = case_dir / "reports" / "asset_generation_report.json"
    selected_requests, skipped_names = _select_mesh_requests(planner_output, asset_names)

    if not selected_requests:
        manifest = {
            "assets": [],
            "assumptions": ["No generated mesh asset requests were selected for this episode."],
            "unresolved_risks": [
                f"Requested asset name was not found or was not a mesh request: {name}" for name in skipped_names
            ],
        }
        report = {
            "ok": not skipped_names,
            "status": "no_mesh_requests",
            "asset_manifest_path": str(manifest_path),
            "asset_generation_report_path": str(report_path),
            "selected_asset_names": [],
            "skipped_asset_names": skipped_names,
            "assets": [],
        }
        dump_json(manifest, manifest_path)
        dump_json(report, report_path)
        return report

    try:
        api_config = MeshyApiConfig.from_env(timeout_sec=CONFIGS.meshy_request.timeout_sec)
    except Exception as exc:  # noqa: BLE001 - convert provider setup failures into manifest entries.
        entries = [_failed_manifest_entry(request, f"{type(exc).__name__}: {exc}") for request in selected_requests]
        manifest = _build_manifest(entries, skipped_names=skipped_names)
        report = {
            "ok": False,
            "status": "provider_unavailable",
            "asset_manifest_path": str(manifest_path),
            "asset_generation_report_path": str(report_path),
            "selected_asset_names": [str(request.get("name", "")) for request in selected_requests],
            "skipped_asset_names": skipped_names,
            "assets": [
                {"ok": False, "request": request, "manifest_entry": entry, "error": entry["notes"][0]}
                for request, entry in zip(selected_requests, entries, strict=False)
            ],
        }
        dump_json(manifest, manifest_path)
        dump_json(report, report_path)
        return report

    output_root = assets_dir / "mesh"
    api_max_workers = _meshy_api_max_workers(len(selected_requests))
    local_max_workers = max(1, min(len(selected_requests), CONFIGS.meshy_request.max_parallel_local_processing))
    selected_names = [str(request.get("name", "")) for request in selected_requests]
    previous_report = _load_json_if_exists(report_path)
    reusable_api_results = _reusable_api_results(previous_report, selected_requests)
    reusable_processed_results = _reusable_processed_results(previous_report)
    progress_report = {
        "ok": False,
        "status": "mesh_asset_generation_running",
        "asset_manifest_path": str(manifest_path),
        "asset_generation_report_path": str(report_path),
        "num_assets": len(selected_requests),
        "api_parallel": api_max_workers > 1,
        "max_parallel_api_requests": api_max_workers,
        "local_parallel": local_max_workers > 1,
        "max_parallel_local_processing": local_max_workers,
        "selected_asset_names": selected_names,
        "skipped_asset_names": skipped_names,
        "api_assets": [],
        "assets": [],
    }
    dump_json(progress_report, report_path)

    api_results: list[dict[str, Any] | None] = [None] * len(selected_requests)
    if reusable_api_results is not None:
        api_results = reusable_api_results
        progress_report["status"] = "meshy_api_requests_reused"
        progress_report["api_assets"] = [_api_progress_entry(result) for result in api_results if result is not None]
        progress_report["resume_source_report_path"] = str(report_path)
        dump_json(progress_report, report_path)
    elif api_max_workers == 1:
        for index, request in enumerate(selected_requests):
            progress_report["status"] = "meshy_api_requests_running"
            progress_report["current_api_asset"] = str(request.get("name", ""))
            progress_report["current_api_asset_index"] = index
            dump_json(progress_report, report_path)
            api_results[index] = _download_one_mesh_asset(
                index=index,
                request=request,
                task=task,
                output_root=output_root,
                api_config=api_config,
            )
            progress_report["api_assets"] = [
                _api_progress_entry(result) for result in api_results if result is not None
            ]
            dump_json(progress_report, report_path)
    else:
        progress_report["status"] = "meshy_api_requests_running"
        dump_json(progress_report, report_path)
        with ThreadPoolExecutor(max_workers=api_max_workers, thread_name_prefix="meshy_api") as executor:
            futures = {
                executor.submit(
                    _download_one_mesh_asset,
                    index=index,
                    request=request,
                    task=task,
                    output_root=output_root,
                    api_config=api_config,
                ): index
                for index, request in enumerate(selected_requests)
            }
            for future in as_completed(futures):
                index = futures[future]
                api_results[index] = future.result()
                progress_report["api_assets"] = [
                    _api_progress_entry(result) for result in api_results if result is not None
                ]
                dump_json(progress_report, report_path)

    results: list[dict[str, Any]] = []
    for index, api_result in enumerate(api_results):
        if api_result is None:
            request = selected_requests[index]
            api_result = {
                "ok": False,
                "request": request,
                "mesh_prompt": _mesh_prompt_from_request(request, task),
                "manifest_entry": _failed_manifest_entry(request, "InternalError: Meshy API phase did not return."),
                "error": "InternalError: Meshy API phase did not return.",
            }
        progress_report["status"] = "mesh_local_processing_running"
        progress_report["current_local_asset"] = str(api_result.get("request", {}).get("name", ""))
        progress_report["current_local_asset_index"] = index
        dump_json(progress_report, report_path)
        local_name = str(api_result.get("request", {}).get("name", ""))
        if local_name in reusable_processed_results:
            result = reusable_processed_results[local_name]
        else:
            result = _process_one_mesh_asset(api_result)
        results.append(result)
        progress_report["assets"] = results
        partial_manifest = _build_manifest([item["manifest_entry"] for item in results], skipped_names=skipped_names)
        dump_json(partial_manifest, manifest_path)
        dump_json(progress_report, report_path)

    manifest = _build_manifest([result["manifest_entry"] for result in results], skipped_names=skipped_names)
    ok = all(bool(result.get("ok")) for result in results) and not skipped_names
    report = {
        "ok": ok,
        "status": "mesh_assets_generated" if ok else "mesh_asset_generation_failed",
        "asset_manifest_path": str(manifest_path),
        "asset_generation_report_path": str(report_path),
        "num_assets": len(results),
        "api_parallel": api_max_workers > 1,
        "max_parallel_api_requests": api_max_workers,
        "local_parallel": local_max_workers > 1,
        "max_parallel_local_processing": local_max_workers,
        "selected_asset_names": selected_names,
        "skipped_asset_names": skipped_names,
        "api_assets": [_api_progress_entry(result) for result in api_results if result is not None],
        "assets": results,
    }
    dump_json(manifest, manifest_path)
    dump_json(report, report_path)
    return report


def _select_mesh_requests(
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
        if _is_mesh_asset_request(request):
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


def _is_mesh_asset_request(request: dict[str, Any]) -> bool:
    asset_type = str(request.get("asset_type", "")).strip().lower().replace(" ", "_")
    return asset_type in MESH_ASSET_TYPES


def _meshy_api_max_workers(num_requests: int) -> int:
    configured = CONFIGS.meshy_request.max_parallel_api_requests
    if configured is None or configured <= 0:
        return max(1, num_requests)
    return max(1, min(num_requests, configured))


def _download_one_mesh_asset(
    *,
    index: int,
    request: dict[str, Any],
    task: str,
    output_root: Path,
    api_config: MeshyApiConfig,
) -> dict[str, Any]:
    try:
        output_dir = output_root / f"{index:02d}_{slugify_prompt(str(request.get('name', 'mesh_asset')))}"
        mesh_prompt = _mesh_prompt_from_request(request, task)
        downloaded = download_meshy_mesh_from_text(
            prompt=mesh_prompt,
            api_config=api_config,
            generation_config=_generation_config(mesh_prompt, output_dir),
            texture_config=_texture_config(request),
        )
        return {
            "ok": True,
            "request": request,
            "mesh_prompt": mesh_prompt,
            "downloaded": downloaded,
            "downloaded_asset": downloaded.to_dict(),
        }
    except Exception as exc:  # noqa: BLE001 - one failed mesh should not hide the full asset report.
        manifest_entry = _failed_manifest_entry(request, f"{type(exc).__name__}: {exc}")
        return {
            "ok": False,
            "request": request,
            "mesh_prompt": _mesh_prompt_from_request(request, task),
            "manifest_entry": manifest_entry,
            "error": manifest_entry["notes"][0],
        }


def _process_one_mesh_asset(api_result: dict[str, Any]) -> dict[str, Any]:
    request = api_result.get("request")
    if not isinstance(request, dict):
        request = {}
    if not api_result.get("ok"):
        return {
            "ok": False,
            "request": request,
            "mesh_prompt": api_result.get("mesh_prompt", ""),
            "manifest_entry": api_result["manifest_entry"],
            "error": api_result.get("error", "Meshy API phase failed."),
            "api_phase": _api_progress_entry(api_result),
        }
    try:
        bundle = process_downloaded_meshy_mesh(
            downloaded=api_result["downloaded"],
            repair_config=_repair_config(),
        )
        manifest_entry = _manifest_entry_from_bundle(request, bundle)
        return {
            "ok": manifest_entry["status"] == "ready",
            "request": request,
            "mesh_prompt": api_result.get("mesh_prompt", ""),
            "manifest_entry": manifest_entry,
            "api_phase": _api_progress_entry(api_result),
            "bundle": bundle.to_dict(),
        }
    except Exception as exc:  # noqa: BLE001
        manifest_entry = _failed_manifest_entry(request, f"{type(exc).__name__}: {exc}")
        return {
            "ok": False,
            "request": request,
            "mesh_prompt": api_result.get("mesh_prompt", ""),
            "manifest_entry": manifest_entry,
            "api_phase": _api_progress_entry(api_result),
            "error": manifest_entry["notes"][0],
        }


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _reusable_api_results(
    previous_report: dict[str, Any] | None,
    selected_requests: list[dict[str, Any]],
) -> list[dict[str, Any] | None] | None:
    if previous_report is None:
        return None
    raw_entries = previous_report.get("api_assets")
    if not isinstance(raw_entries, list):
        raw_assets = previous_report.get("assets")
        raw_entries = [
            item["api_phase"]
            for item in raw_assets
            if isinstance(item, dict) and isinstance(item.get("api_phase"), dict)
        ] if isinstance(raw_assets, list) else None
    if not isinstance(raw_entries, list):
        return None
    entries_by_name = {
        str(entry.get("request", {}).get("name", "")): entry for entry in raw_entries if isinstance(entry, dict)
    }
    results: list[dict[str, Any] | None] = []
    for request in selected_requests:
        name = str(request.get("name", ""))
        entry = entries_by_name.get(name)
        if not entry or not entry.get("ok") or not isinstance(entry.get("downloaded_asset"), dict):
            return None
        downloaded = DownloadedMeshyAsset.from_dict(entry["downloaded_asset"])
        results.append(
            {
                "ok": True,
                "request": request,
                "mesh_prompt": entry.get("mesh_prompt", ""),
                "downloaded": downloaded,
                "downloaded_asset": downloaded.to_dict(),
            }
        )
    return results


def _reusable_processed_results(previous_report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if previous_report is None:
        return {}
    raw_assets = previous_report.get("assets")
    if not isinstance(raw_assets, list):
        return {}
    reusable: dict[str, dict[str, Any]] = {}
    for item in raw_assets:
        if not isinstance(item, dict) or "manifest_entry" not in item:
            continue
        manifest_entry = item.get("manifest_entry")
        if not isinstance(manifest_entry, dict):
            continue
        if manifest_entry.get("status") != "ready":
            continue
        runtime_path = manifest_entry.get("runtime_path")
        visual_path = manifest_entry.get("visual_path")
        texture_path = manifest_entry.get("texture_path")
        required_paths = [runtime_path, visual_path, texture_path]
        if any(isinstance(path, str) and path != "unavailable" and not Path(path).is_file() for path in required_paths):
            continue
        name = str(item.get("request", {}).get("name", ""))
        if name:
            reusable[name] = item
    return reusable


def _api_progress_entry(api_result: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "ok": bool(api_result.get("ok")),
        "request": api_result.get("request"),
        "mesh_prompt": api_result.get("mesh_prompt", ""),
    }
    if api_result.get("ok"):
        entry["downloaded_asset"] = api_result.get("downloaded_asset")
    else:
        entry["manifest_entry"] = api_result.get("manifest_entry")
        entry["error"] = api_result.get("error")
    return entry


def _generation_config(prompt: str, output_dir: Path) -> MeshyGenerationConfig:
    defaults = CONFIGS.meshy_request
    return MeshyGenerationConfig(
        prompt=prompt,
        output_dir=output_dir,
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


def _texture_config(request: dict[str, Any]) -> MeshyTextureConfig | None:
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


def _repair_config() -> MeshRepairConfig:
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


def _mesh_prompt_from_request(request: dict[str, Any], task: str) -> str:
    _ = task
    name = str(request.get("name", "mesh_asset")).replace("_", " ")
    purpose = _clean_prompt_field(request.get("purpose"), max_chars=260)
    simulation_role = _clean_prompt_field(request.get("simulation_role"), max_chars=120)
    texture_needs = _clean_prompt_field(request.get("texture_needs"), max_chars=160)
    parts = [
        f"Create one simulation-ready 3D mesh: {name}.",
        purpose,
        f"Role: {simulation_role}.",
    ]
    request_size = _request_size(request)
    if request_size is not None:
        parts.append(f"Approximate positive dimensions in meters: {request_size}.")
    if texture_needs:
        parts.append(f"Material: {texture_needs}.")
    parts.append("Keep one coherent object with clear silhouette and robust closed surfaces for physics.")
    return _clean_prompt_field(" ".join(part for part in parts if part), max_chars=560)


def _manifest_entry_from_bundle(request: dict[str, Any], bundle: Any) -> dict[str, Any]:
    repair = bundle.repair
    manifold = bundle.manifold
    runtime_path = repair.output_mesh_path if repair is not None and repair.ok else bundle.generation.mesh_path
    summary = load_mesh_asset_summary(runtime_path)
    request_size = _request_size(request)
    mesh_bbox = _vector3(summary.get("bbox_size"))
    file_meshes_are_zup = False
    genesis_bbox = _bbox_after_yup_to_zup(mesh_bbox)
    scale = _scale_to_bbox(genesis_bbox, request_size) or [1.0, 1.0, 1.0]
    visual_path = _visual_mesh_path(bundle) or runtime_path
    texture_path = _texture_path(bundle)
    notes = [
        "Generated by the Planner-callable mesh asset action.",
        "Use runtime_path as the strict-manifold Genesis geometry for simulation/collision.",
        "visual_path is the seam-aware textured render artifact for the same logical asset, not a separate object to "
        "instantiate as an independent simulation body.",
        "Pass file_meshes_are_zup and scale exactly as listed; generated OBJ assets are provider Y-up unless stated "
        "otherwise.",
        "If texture transfer succeeded, visual_path binds the rebaked base color through its neighboring MTL file while "
        "runtime_path remains the repaired manifold geometry.",
        "Use texture_path as evidence metadata for the transferred base-color image and for texture preview checks.",
    ]
    if repair is None or not repair.ok:
        notes.append(
            "Mesh repair did not produce a ready repaired mesh; runtime_path falls back to the downloaded mesh."
        )
    if request_size is not None and genesis_bbox is not None:
        notes.append("Scale was estimated from requested size and mesh bbox after Y-up to Genesis Z-up conversion.")
    status = "ready" if repair is not None and repair.ok and manifold is not None and manifold.ok else "failed"
    return {
        "logical_name": str(request.get("name", "mesh_asset")),
        "source_type": "generated_mesh",
        "runtime_path": str(runtime_path.resolve()),
        "visual_path": str(visual_path.resolve()) if visual_path is not None else None,
        "scale": scale,
        "bbox": request_size or genesis_bbox,
        "file_meshes_are_zup": file_meshes_are_zup,
        "texture_path": str(texture_path.resolve()) if texture_path is not None else None,
        "simulation_role": str(request.get("simulation_role", "scene prop")),
        "status": status,
        "notes": notes,
    }


def _visual_mesh_path(bundle: Any) -> Path | None:
    transfer = bundle.texture_transfer
    if transfer is not None and transfer.ok and transfer.output_mesh_path is not None:
        return transfer.output_mesh_path
    texture = bundle.texture
    if texture is not None and texture.ok and texture.textured_mesh_path is not None:
        return texture.textured_mesh_path
    return None


def _texture_path(bundle: Any) -> Path | None:
    transfer = bundle.texture_transfer
    if transfer is not None and transfer.ok and transfer.output_texture_path is not None:
        return transfer.output_texture_path
    texture = bundle.texture
    if texture is not None and texture.ok:
        return texture.texture_paths.get("base_color")
    return None


def _failed_manifest_entry(request: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "logical_name": str(request.get("name", "mesh_asset")),
        "source_type": "generated_mesh",
        "runtime_path": "unavailable",
        "visual_path": None,
        "scale": _positive_vector3(request.get("scale")),
        "bbox": _request_size(request),
        "file_meshes_are_zup": False,
        "texture_path": None,
        "simulation_role": str(request.get("simulation_role", "scene prop")),
        "status": "failed",
        "notes": [error],
    }


def _build_manifest(entries: list[dict[str, Any]], *, skipped_names: list[str]) -> dict[str, Any]:
    unresolved = [entry["logical_name"] for entry in entries if entry.get("status") != "ready"]
    unresolved.extend(f"Skipped asset request: {name}" for name in skipped_names)
    return {
        "assets": entries,
        "assumptions": [
            "Generated mesh runtime paths are written before code writers run, so workers should read this manifest.",
        ],
        "unresolved_risks": unresolved,
    }


def _vector3(value: object) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    output: list[float] = []
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool):
            return None
        output.append(float(item))
    return output


def _positive_vector3(value: object) -> list[float] | None:
    vector = _vector3(value)
    if vector is None:
        return None
    if any(item <= 0.0 for item in vector):
        return None
    return vector


def _request_size(request: dict[str, Any]) -> list[float] | None:
    return _positive_vector3(request.get("scale")) or _positive_vector3(request.get("bbox"))


def _scale_to_bbox(mesh_bbox: list[float] | None, request_bbox: list[float] | None) -> list[float] | None:
    if mesh_bbox is None or request_bbox is None:
        return None
    scale: list[float] = []
    for mesh_size, requested_size in zip(mesh_bbox, request_bbox, strict=True):
        if mesh_size <= 0 or requested_size <= 0:
            return None
        scale.append(float(requested_size) / float(mesh_size))
    return scale


def _bbox_after_yup_to_zup(mesh_bbox: list[float] | None) -> list[float] | None:
    if mesh_bbox is None:
        return None
    return [float(mesh_bbox[0]), float(mesh_bbox[2]), float(mesh_bbox[1])]


def _clean_prompt_field(value: object, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip(" ,.;:") + "."
