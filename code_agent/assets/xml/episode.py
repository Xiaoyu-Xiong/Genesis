from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from code_agent.assets.mesh.workflow.steps import slugify_prompt
from code_agent.assets.xml.agent import generate_xml_asset
from code_agent.assets.xml.request_adapter import select_xml_requests, xml_prompt_from_request, xml_requested_bbox
from code_agent.configs import CONFIGS
from code_agent.io_utils import dump_json, load_json_object


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
    selected_names = [str(request.get("name", "")) for request in selected_requests]
    mentioned_names = {name for name in selected_names if name} | {name for name in asset_names or [] if name}
    preserved_entries = _preserved_ready_xml_manifest_entries(manifest_path, mentioned_names)
    preserved_results = [
        {"ok": True, "preserved": True, "request": {"name": entry.get("logical_name", "")}, "manifest_entry": entry}
        for entry in preserved_entries
    ]
    preserved_names = [str(entry.get("logical_name", "")) for entry in preserved_entries if entry.get("logical_name")]

    if not selected_requests:
        manifest = build_xml_manifest(preserved_entries, skipped_names=skipped_names)
        manifest["assumptions"].append(
            "No generated XML/MJCF asset requests were selected; ready unselected XML entries were preserved."
        )
        report = {
            "ok": not skipped_names,
            "status": "xml_assets_ready" if preserved_entries and not skipped_names else "no_xml_requests",
            "asset_manifest_path": str(manifest_path),
            "asset_generation_report_path": str(report_path),
            "num_assets": len(preserved_entries),
            "xml_parallel": False,
            "max_parallel_workers": 1,
            "selected_asset_names": [],
            "preserved_asset_names": preserved_names,
            "skipped_asset_names": skipped_names,
            "assets": preserved_results,
        }
        dump_json(manifest, manifest_path)
        dump_json(report, report_path)
        return report

    configured_workers = CONFIGS.xml_asset.max_parallel_workers
    max_workers = (
        max(1, len(selected_requests))
        if configured_workers is None or configured_workers <= 0
        else max(1, min(len(selected_requests), configured_workers))
    )
    progress_report = {
        "ok": False,
        "status": "xml_asset_generation_running",
        "asset_manifest_path": str(manifest_path),
        "asset_generation_report_path": str(report_path),
        "num_assets": len(preserved_entries) + len(selected_requests),
        "xml_parallel": max_workers > 1,
        "max_parallel_workers": max_workers,
        "selected_asset_names": selected_names,
        "preserved_asset_names": preserved_names,
        "skipped_asset_names": skipped_names,
        "assets": preserved_results,
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
            except Exception as exc:
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
            progress_report["assets"] = preserved_results + completed
            progress_report["completed_asset_names"] = [
                str(item.get("request", {}).get("name", "")) for item in completed
            ]
            partial_manifest = build_xml_manifest(
                _merge_xml_manifest_entries(preserved_entries, [item["manifest_entry"] for item in completed]),
                skipped_names=skipped_names,
            )
            dump_json(partial_manifest, manifest_path)
            dump_json(progress_report, report_path)

    final_results = [item for item in results if item is not None]
    entries = _merge_xml_manifest_entries(preserved_entries, [result["manifest_entry"] for result in final_results])
    manifest = build_xml_manifest(entries, skipped_names=skipped_names)
    ok = all(bool(result.get("ok")) for result in final_results) and not skipped_names
    report = {
        "ok": ok,
        "status": "xml_assets_generated" if ok else "xml_asset_generation_failed",
        "asset_manifest_path": str(manifest_path),
        "asset_generation_report_path": str(report_path),
        "num_assets": len(entries),
        "xml_parallel": max_workers > 1,
        "max_parallel_workers": max_workers,
        "selected_asset_names": selected_names,
        "preserved_asset_names": preserved_names,
        "skipped_asset_names": skipped_names,
        "assets": preserved_results + final_results,
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
            "MJCF dimensions are baked into generated XML geometry by the XML worker; do not expect mesh-style scaling.",
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
        allow_passive_freejoint=_allows_passive_freejoint_asset(request),
        allowed_mesh_asset_roots=(output_root.parent,),
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


def _allows_passive_freejoint_asset(request: dict[str, Any]) -> bool:
    text = " ".join(
        str(request.get(key, ""))
        for key in ("name", "asset_type", "purpose", "simulation_role", "texture_needs")
    ).lower()
    negative_control_phrases = (
        "no actuator",
        "no actuators",
        "without actuator",
        "without actuators",
        "not actuated",
        "non-actuated",
    )
    marker_text = text
    for phrase in negative_control_phrases:
        marker_text = marker_text.replace(phrase, "")
    passive_markers = (
        "passive",
        "projectile",
        "loose",
        "free rigid",
        "freejoint",
        "free joint",
        "thrown",
        "toss",
        "flick",
        "puck",
        "ball",
        "ring",
        "coin",
    )
    controlled_markers = (
        "robot",
        "arm",
        "gripper",
        "hand",
        "actuator",
        "actuated",
        "motor",
        "servo",
        "hinge",
        "slider",
        "gate",
        "windmill",
        "drawer",
        "door",
        "mechanism",
        "striker",
        "flipper",
        "putter",
    )
    return any(marker in text for marker in passive_markers) and not any(
        marker in marker_text for marker in controlled_markers
    )


def _preserved_ready_xml_manifest_entries(manifest_path: Path, selected_names: set[str]) -> list[dict[str, Any]]:
    manifest = load_json_object(manifest_path) or {}
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
        if _is_ready_xml_manifest_entry(entry):
            preserved.append(entry)
    return preserved


def _is_ready_xml_manifest_entry(entry: dict[str, Any]) -> bool:
    if entry.get("source_type") != "mjcf" or entry.get("status") != "ready":
        return False
    runtime_path = entry.get("runtime_path")
    if not isinstance(runtime_path, str) or not runtime_path or runtime_path == "unavailable":
        return False
    return Path(runtime_path).is_file()


def _merge_xml_manifest_entries(
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
