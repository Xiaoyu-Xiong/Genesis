from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

from code_agent.assets.mesh.manifest import build_manifest, failed_manifest_entry, manifest_entry_from_bundle
from code_agent.assets.mesh.models import MeshyApiConfig, MeshyPromptLengthError
from code_agent.assets.mesh.pipeline import download_meshy_mesh_from_text, process_downloaded_meshy_mesh
from code_agent.assets.mesh.request_adapter import (
    mesh_prompt_from_request,
    mesh_repair_config,
    meshy_generation_config,
    meshy_texture_config,
    select_mesh_requests,
)
from code_agent.assets.mesh.validation import run_genesis_fem_import_validation
from code_agent.assets.mesh.workflow.steps import slugify_prompt
from code_agent.configs import CONFIGS
from code_agent.io_utils import dump_json


MESH_PROMPT_LENGTH_FAILURE_CLASS = "mesh.prompt_length_exceeded"


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
    selected_names = [str(request.get("name", "")) for request in selected_requests]
    mentioned_names = {name for name in selected_names if name} | {name for name in asset_names or [] if name}
    preserved_entries = _preserved_ready_manifest_entries(manifest_path, mentioned_names)
    preserved_results = [_preserved_result_entry(entry) for entry in preserved_entries]
    preserved_names = [str(entry.get("logical_name", "")) for entry in preserved_entries if entry.get("logical_name")]

    if not selected_requests:
        manifest = build_manifest(preserved_entries, skipped_names=skipped_names)
        manifest["assumptions"].append(
            "No generated mesh asset requests were selected; ready unselected mesh entries were preserved."
        )
        report = {
            "ok": not skipped_names,
            "status": "mesh_assets_ready" if preserved_entries and not skipped_names else "no_mesh_requests",
            "message": None,
            "asset_manifest_path": str(manifest_path),
            "asset_generation_report_path": str(report_path),
            "num_assets": len(preserved_entries),
            "selected_asset_names": [],
            "preserved_asset_names": preserved_names,
            "skipped_asset_names": skipped_names,
            "failure_classes": [],
            "assets": preserved_results,
        }
        dump_json(manifest, manifest_path)
        dump_json(report, report_path)
        return report

    try:
        api_config = MeshyApiConfig.from_env(timeout_sec=CONFIGS.meshy_request.timeout_sec)
    except Exception as exc:  # noqa: BLE001 - convert provider setup failures into manifest entries.
        generated_entries = [
            failed_manifest_entry(request, f"{type(exc).__name__}: {exc}") for request in selected_requests
        ]
        entries = _merge_manifest_entries(preserved_entries, generated_entries)
        manifest = build_manifest(entries, skipped_names=skipped_names)
        generated_results = [
            {"ok": False, "request": request, "manifest_entry": entry, "error": entry["notes"][0]}
            for request, entry in zip(selected_requests, generated_entries, strict=False)
        ]
        report = {
            "ok": False,
            "status": "provider_unavailable",
            "message": "Mesh asset provider is unavailable.",
            "asset_manifest_path": str(manifest_path),
            "asset_generation_report_path": str(report_path),
            "num_assets": len(entries),
            "selected_asset_names": selected_names,
            "generated_asset_names": selected_names,
            "preserved_asset_names": preserved_names,
            "skipped_asset_names": skipped_names,
            "failure_classes": ["mesh.provider_unavailable"],
            "assets": preserved_results + generated_results,
        }
        dump_json(manifest, manifest_path)
        dump_json(report, report_path)
        return report

    output_root = assets_dir / "mesh"
    api_max_workers = _meshy_api_max_workers(len(selected_requests))
    local_max_workers = max(1, min(len(selected_requests), CONFIGS.meshy_request.max_parallel_local_processing))
    progress_report = {
        "ok": False,
        "status": "mesh_asset_generation_running",
        "asset_manifest_path": str(manifest_path),
        "asset_generation_report_path": str(report_path),
        "num_assets": len(preserved_entries) + len(selected_requests),
        "api_parallel": api_max_workers > 1,
        "max_parallel_api_requests": api_max_workers,
        "local_parallel": local_max_workers > 1,
        "max_parallel_local_processing": local_max_workers,
        "selected_asset_names": selected_names,
        "generated_asset_names": selected_names,
        "preserved_asset_names": preserved_names,
        "skipped_asset_names": skipped_names,
        "failure_classes": [],
        "api_assets": [],
        "assets": preserved_results,
    }
    dump_json(progress_report, report_path)

    api_results: list[dict[str, Any] | None] = [None] * len(selected_requests)
    if api_max_workers == 1:
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
        result = _process_one_mesh_asset(api_result)
        results.append(result)
        progress_report["assets"] = preserved_results + results
        partial_entries = _merge_manifest_entries(preserved_entries, [item["manifest_entry"] for item in results])
        partial_manifest = build_manifest(partial_entries, skipped_names=skipped_names)
        dump_json(partial_manifest, manifest_path)
        dump_json(progress_report, report_path)

    entries = _merge_manifest_entries(preserved_entries, [result["manifest_entry"] for result in results])
    manifest = build_manifest(entries, skipped_names=skipped_names)
    ok = all(bool(result.get("ok")) for result in results) and not skipped_names
    failure_classes = _failure_classes_from_results(results)
    prompt_length_assets = _asset_names_for_failure(results, MESH_PROMPT_LENGTH_FAILURE_CLASS)
    prompt_length_message = _prompt_length_report_message(prompt_length_assets) if prompt_length_assets else None
    report = {
        "ok": ok,
        "status": (
            "mesh_assets_generated"
            if ok
            else "mesh_prompt_length_exceeded"
            if prompt_length_assets
            else "mesh_asset_generation_failed"
        ),
        "message": prompt_length_message,
        "recommended_owner": "planner" if prompt_length_assets else None,
        "repair_summary": prompt_length_message,
        "asset_manifest_path": str(manifest_path),
        "asset_generation_report_path": str(report_path),
        "num_assets": len(entries),
        "api_parallel": api_max_workers > 1,
        "max_parallel_api_requests": api_max_workers,
        "local_parallel": local_max_workers > 1,
        "max_parallel_local_processing": local_max_workers,
        "selected_asset_names": selected_names,
        "generated_asset_names": selected_names,
        "preserved_asset_names": preserved_names,
        "skipped_asset_names": skipped_names,
        "failure_classes": failure_classes,
        "api_assets": [_api_progress_entry(result) for result in api_results if result is not None],
        "assets": preserved_results + results,
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
    except MeshyPromptLengthError as exc:
        mesh_prompt = mesh_prompt_from_request(request, task)
        message = _prompt_length_asset_message(request, mesh_prompt, exc)
        manifest_entry = failed_manifest_entry(request, message)
        return {
            "ok": False,
            "request": request,
            "mesh_prompt": mesh_prompt,
            "manifest_entry": manifest_entry,
            "error": message,
            "failure_class": MESH_PROMPT_LENGTH_FAILURE_CLASS,
            "recommended_owner": "planner",
            "repair_summary": message,
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
            "failure_class": api_result.get("failure_class"),
            "recommended_owner": api_result.get("recommended_owner"),
            "repair_summary": api_result.get("repair_summary"),
            "api_phase": _api_progress_entry(api_result),
        }
    try:
        bundle = process_downloaded_meshy_mesh(
            downloaded=api_result["downloaded"],
            repair_config=mesh_repair_config(),
        )
        pre_validation_entry = manifest_entry_from_bundle(request, bundle)
        genesis_validation = run_genesis_fem_import_validation(pre_validation_entry)
        bundle.genesis_fem_import = genesis_validation
        _write_genesis_validation_artifacts(bundle)
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


def _preserved_ready_manifest_entries(manifest_path: Path, selected_names: set[str]) -> list[dict[str, Any]]:
    manifest = _load_json_dict(manifest_path)
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list):
        return []
    preserved: list[dict[str, Any]] = []
    for raw_entry in raw_assets:
        if not isinstance(raw_entry, dict):
            continue
        entry = dict(raw_entry)
        name = str(entry.get("logical_name", ""))
        if name in selected_names:
            continue
        if _is_ready_generated_mesh_entry(entry):
            preserved.append(entry)
    return preserved


def _is_ready_generated_mesh_entry(entry: dict[str, Any]) -> bool:
    if entry.get("source_type") != "generated_mesh" or entry.get("status") != "ready":
        return False
    runtime_path = entry.get("runtime_path")
    if not _manifest_file_available(runtime_path, required=True):
        return False
    if not _manifest_file_available(entry.get("visual_path"), required=False):
        return False
    return _manifest_file_available(entry.get("texture_path"), required=False)


def _manifest_file_available(value: Any, *, required: bool) -> bool:
    if value is None:
        return not required
    if not isinstance(value, str) or not value or value == "unavailable":
        return False
    return Path(value).is_file()


def _preserved_result_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "preserved": True,
        "request": {"name": entry.get("logical_name", "")},
        "manifest_entry": entry,
    }


def _merge_manifest_entries(
    preserved_entries: list[dict[str, Any]],
    generated_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries_by_name: dict[str, dict[str, Any]] = {}
    ordered_names: list[str] = []
    for entry in [*preserved_entries, *generated_entries]:
        name = str(entry.get("logical_name", ""))
        if not name:
            continue
        if name not in entries_by_name:
            ordered_names.append(name)
        entries_by_name[name] = entry
    return [entries_by_name[name] for name in ordered_names]


def _failure_classes_from_results(results: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(result.get("failure_class"))
            for result in results
            if result.get("failure_class")
        }
    )


def _asset_names_for_failure(results: list[dict[str, Any]], failure_class: str) -> list[str]:
    names: list[str] = []
    for result in results:
        if result.get("failure_class") != failure_class:
            continue
        request = result.get("request")
        if isinstance(request, dict):
            name = str(request.get("name", "")).strip()
            if name:
                names.append(name)
    return sorted(set(names))


def _prompt_length_report_message(asset_names: list[str]) -> str:
    names = ", ".join(asset_names)
    return (
        f"Meshy mesh prompt length exceeded the {CONFIGS.meshy_request.prompt_max_chars}-character limit for: {names}. "
        "Planner should simplify the affected generated_mesh asset request once, especially purpose, simulation_role, "
        "and texture_needs, then retry start_mesh_assets for those asset_names."
    )


def _prompt_length_asset_message(
    request: dict[str, Any],
    mesh_prompt: str,
    exc: MeshyPromptLengthError,
) -> str:
    limit = exc.max_chars or CONFIGS.meshy_request.prompt_max_chars
    prompt_len = exc.prompt_len or len(mesh_prompt)
    name = str(request.get("name", "mesh_asset"))
    return (
        f"Meshy mesh prompt length exceeded ({prompt_len}>{limit}) for `{name}`. "
        "Planner should simplify this generated_mesh asset request once, especially purpose, simulation_role, and "
        "texture_needs, then retry mesh generation."
    )


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_genesis_validation_artifacts(bundle: Any) -> None:
    validation = getattr(bundle, "genesis_fem_import", None)
    if validation is None:
        return
    output_dir = Path(bundle.generation.output_dir)
    dump_json(validation.to_dict(), output_dir / "genesis_fem_import_check.json")
    metadata_path = bundle.generation.metadata_path
    metadata = _load_json_dict(metadata_path)
    metadata["genesis_fem_import"] = validation.to_dict()
    dump_json(metadata, metadata_path)


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
