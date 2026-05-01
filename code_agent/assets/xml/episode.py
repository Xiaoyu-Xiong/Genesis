from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from code_agent.assets.mesh.workflow.steps import slugify_prompt
from code_agent.assets.xml.agent import generate_xml_asset
from code_agent.assets.xml.request_adapter import select_xml_requests, xml_prompt_from_request, xml_requested_bbox
from code_agent.configs import CONFIGS
from code_agent.io_utils import dump_json


def generate_xml_assets_for_episode(
    *,
    case_dir: Path,
    task: str,
    planner_output: dict[str, Any],
    asset_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run Planner-requested XML/MJCF assets and write a partial episode asset manifest."""

    assets_dir = case_dir / "assets"
    manifest_path = assets_dir / "xml_asset_manifest.json"
    report_path = case_dir / "reports" / "xml_asset_generation_report.json"
    selected_requests, skipped_names = select_xml_requests(planner_output, asset_names)

    if not selected_requests:
        manifest = {
            "assets": [],
            "assumptions": ["No generated XML/MJCF asset requests were selected for this episode."],
            "unresolved_risks": [
                f"Requested asset name was not found or was not an XML/MJCF request: {name}" for name in skipped_names
            ],
        }
        report = {
            "ok": not skipped_names,
            "status": "no_xml_requests",
            "asset_manifest_path": str(manifest_path),
            "asset_generation_report_path": str(report_path),
            "num_assets": 0,
            "xml_parallel": False,
            "max_parallel_workers": 1,
            "selected_asset_names": [],
            "skipped_asset_names": skipped_names,
            "assets": [],
        }
        dump_json(manifest, manifest_path)
        dump_json(report, report_path)
        return report

    selected_names = [str(request.get("name", "")) for request in selected_requests]
    max_workers = _xml_max_workers(len(selected_requests))
    progress_report = {
        "ok": False,
        "status": "xml_asset_generation_running",
        "asset_manifest_path": str(manifest_path),
        "asset_generation_report_path": str(report_path),
        "num_assets": len(selected_requests),
        "xml_parallel": max_workers > 1,
        "max_parallel_workers": max_workers,
        "selected_asset_names": selected_names,
        "skipped_asset_names": skipped_names,
        "assets": [],
    }
    dump_json(progress_report, report_path)

    results: list[dict[str, Any] | None] = [None] * len(selected_requests)
    output_root = assets_dir / "xml"
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="xml_asset_worker") as executor:
        futures = {
            executor.submit(
                _generate_one_xml_asset,
                index=index,
                request=request,
                task=task,
                output_root=output_root,
            ): index
            for index, request in enumerate(selected_requests)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - keep one failed XML worker visible in the aggregate report.
                request = selected_requests[index]
                result = {
                    "ok": False,
                    "request": request,
                    "xml_prompt": xml_prompt_from_request(request, task),
                    "manifest_entry": failed_xml_manifest_entry(request, f"{type(exc).__name__}: {exc}"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results[index] = result
            completed = [item for item in results if item is not None]
            progress_report["assets"] = completed
            progress_report["completed_asset_names"] = [
                str(item.get("request", {}).get("name", "")) for item in completed
            ]
            partial_manifest = build_xml_manifest(
                [item["manifest_entry"] for item in completed],
                skipped_names=skipped_names,
            )
            dump_json(partial_manifest, manifest_path)
            dump_json(progress_report, report_path)

    final_results = [item for item in results if item is not None]
    manifest = build_xml_manifest([result["manifest_entry"] for result in final_results], skipped_names=skipped_names)
    ok = all(bool(result.get("ok")) for result in final_results) and not skipped_names
    report = {
        "ok": ok,
        "status": "xml_assets_generated" if ok else "xml_asset_generation_failed",
        "asset_manifest_path": str(manifest_path),
        "asset_generation_report_path": str(report_path),
        "num_assets": len(final_results),
        "xml_parallel": max_workers > 1,
        "max_parallel_workers": max_workers,
        "selected_asset_names": selected_names,
        "skipped_asset_names": skipped_names,
        "assets": final_results,
    }
    dump_json(manifest, manifest_path)
    dump_json(report, report_path)
    return report


def build_xml_manifest(entries: list[dict[str, Any]], *, skipped_names: list[str]) -> dict[str, Any]:
    unresolved = [str(entry["logical_name"]) for entry in entries if entry.get("status") != "ready"]
    unresolved.extend(f"Skipped asset request: {name}" for name in skipped_names)
    return {
        "assets": entries,
        "assumptions": [
            "Generated XML/MJCF assets are written before manifest-dependent code writers run.",
            "MJCF dimensions are baked into primitive geometry by the XML worker; do not expect mesh-style scaling.",
        ],
        "unresolved_risks": unresolved,
    }


def failed_xml_manifest_entry(request: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "logical_name": str(request.get("name", "xml_asset")),
        "source_type": "mjcf",
        "runtime_path": "unavailable",
        "visual_path": None,
        "scale": None,
        "bbox": xml_requested_bbox(request),
        "file_meshes_are_zup": None,
        "texture_path": None,
        "joints": [],
        "actuators": [],
        "control_interface": {},
        "base": {},
        "validation": {"errors": [error], "warnings": [], "model_summary": {}},
        "simulation_role": str(request.get("simulation_role", "articulated asset")),
        "status": "failed",
        "notes": [error],
    }


def _generate_one_xml_asset(
    *,
    index: int,
    request: dict[str, Any],
    task: str,
    output_root: Path,
) -> dict[str, Any]:
    logical_name = str(request.get("name", "xml_asset")) or "xml_asset"
    output_dir = output_root / f"{index:02d}_{slugify_prompt(logical_name)}"
    prompt = xml_prompt_from_request(request, task)
    generation = generate_xml_asset(
        task=prompt,
        output_dir=output_dir,
        logical_name=logical_name,
        max_attempts=CONFIGS.xml_asset.max_generation_attempts,
    )
    manifest_entry = _manifest_entry_from_generation(request, generation)
    ok = bool(generation.get("ok")) and manifest_entry.get("status") == "ready"
    return {
        "ok": ok,
        "request": request,
        "xml_prompt": prompt,
        "manifest_entry": manifest_entry,
        "generation": generation,
        "error": None if ok else _generation_error(generation),
    }


def _manifest_entry_from_generation(request: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    raw_entry = generation.get("manifest_entry")
    if not isinstance(raw_entry, dict):
        return failed_xml_manifest_entry(request, _generation_error(generation))
    entry = dict(raw_entry)
    entry["logical_name"] = str(request.get("name", entry.get("logical_name", "xml_asset")))
    entry["simulation_role"] = str(request.get("simulation_role", entry.get("simulation_role", "articulated asset")))
    entry["bbox"] = xml_requested_bbox(request)
    entry["scale"] = None
    notes = [str(item) for item in entry.get("notes", []) if item]
    if xml_requested_bbox(request) is not None:
        notes.append("Requested positive dimensions were passed to the XML worker and should be baked into the XML.")
    notes.append("Instantiate this MJCF asset from runtime_path; do not split simulation and rendering assets.")
    entry["notes"] = notes
    return entry


def _generation_error(generation: dict[str, Any]) -> str:
    if isinstance(generation.get("status"), str):
        return str(generation["status"])
    attempts = generation.get("attempts")
    if isinstance(attempts, list) and attempts:
        last = attempts[-1]
        if isinstance(last, dict):
            worker_error = last.get("worker_error")
            validation = last.get("validation_report")
            if worker_error:
                return str(worker_error)
            if isinstance(validation, dict) and validation.get("errors"):
                return "; ".join(str(item) for item in validation["errors"])
    return "XML asset generation failed."


def _xml_max_workers(num_requests: int) -> int:
    configured = CONFIGS.xml_asset.max_parallel_workers
    if configured is None or configured <= 0:
        return max(1, num_requests)
    return max(1, min(num_requests, configured))
