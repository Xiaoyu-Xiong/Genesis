from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from code_agent.assets.mesh.manifest import build_manifest, failed_manifest_entry, manifest_entry_from_bundle
from code_agent.assets.mesh.models import MeshyApiConfig
from code_agent.assets.mesh.pipeline import DownloadedMeshyAsset, download_meshy_mesh_from_text, process_downloaded_meshy_mesh
from code_agent.assets.mesh.request_adapter import (
    mesh_prompt_from_request,
    mesh_repair_config,
    meshy_generation_config,
    meshy_texture_config,
    select_mesh_requests,
)
from code_agent.assets.mesh.workflow.steps import slugify_prompt
from code_agent.configs import CONFIGS
from code_agent.io_utils import dump_json


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
    selected_requests, skipped_names = select_mesh_requests(planner_output, asset_names)

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
        entries = [failed_manifest_entry(request, f"{type(exc).__name__}: {exc}") for request in selected_requests]
        manifest = build_manifest(entries, skipped_names=skipped_names)
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
                "mesh_prompt": mesh_prompt_from_request(request, task),
                "manifest_entry": failed_manifest_entry(request, "InternalError: Meshy API phase did not return."),
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
        partial_manifest = build_manifest([item["manifest_entry"] for item in results], skipped_names=skipped_names)
        dump_json(partial_manifest, manifest_path)
        dump_json(progress_report, report_path)

    manifest = build_manifest([result["manifest_entry"] for result in results], skipped_names=skipped_names)
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
        mesh_prompt = mesh_prompt_from_request(request, task)
        downloaded = download_meshy_mesh_from_text(
            prompt=mesh_prompt,
            api_config=api_config,
            generation_config=meshy_generation_config(mesh_prompt, output_dir),
            texture_config=meshy_texture_config(request),
        )
        return {
            "ok": True,
            "request": request,
            "mesh_prompt": mesh_prompt,
            "downloaded": downloaded,
            "downloaded_asset": downloaded.to_dict(),
        }
    except Exception as exc:  # noqa: BLE001 - one failed mesh should not hide the full asset report.
        manifest_entry = failed_manifest_entry(request, f"{type(exc).__name__}: {exc}")
        return {
            "ok": False,
            "request": request,
            "mesh_prompt": mesh_prompt_from_request(request, task),
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
            repair_config=mesh_repair_config(),
        )
        manifest_entry = manifest_entry_from_bundle(request, bundle)
        return {
            "ok": manifest_entry["status"] == "ready",
            "request": request,
            "mesh_prompt": api_result.get("mesh_prompt", ""),
            "manifest_entry": manifest_entry,
            "api_phase": _api_progress_entry(api_result),
            "bundle": bundle.to_dict(),
        }
    except Exception as exc:  # noqa: BLE001
        manifest_entry = failed_manifest_entry(request, f"{type(exc).__name__}: {exc}")
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
