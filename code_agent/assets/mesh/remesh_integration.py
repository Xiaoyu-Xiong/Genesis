from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import trimesh

from code_agent.configs import CONFIGS
from code_agent.io_utils import dump_json, load_json_object

from .remesh import IsotropicRemeshConfig, remesh_mesh_asset


REMESH_FAILURE_CLASS = "mesh_remesh.validation_failed"
REMESH_PRECONDITION_FAILURE_CLASS = "mesh_remesh.precondition_failed"


def apply_automatic_remesh_to_entry(
    entry: dict[str, Any],
    *,
    bundle: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply configured local remeshing after Meshy processing, with original-asset fallback."""

    if not CONFIGS.mesh_repair.auto_remesh_enabled:
        return deepcopy(entry), _automatic_outcome("skipped_disabled")
    if not _source_geometry_ready(entry):
        return deepcopy(entry), _automatic_outcome("skipped_source_geometry_invalid", fallback_used=True)

    provenance = _provenance_from_entry(entry, bundle=bundle)
    target_face_count = int(CONFIGS.mesh_repair.auto_remesh_target_face_count)
    target_face_tolerance = float(CONFIGS.mesh_repair.auto_remesh_target_face_tolerance)
    skip_face_count_upper_bound = int(target_face_count * (1.0 + target_face_tolerance))
    source_face_count = _source_face_count(provenance["base_runtime_path"])
    if source_face_count is not None and source_face_count <= skip_face_count_upper_bound:
        remesh_metadata = _remesh_metadata(
            mode="automatic",
            status="skipped_not_needed",
            provenance=provenance,
            target_face_count=target_face_count,
            target_edge_length=None,
            target_face_tolerance=target_face_tolerance,
            report=None,
            fallback_used=False,
        )
        remesh_metadata["source_face_count"] = source_face_count
        remesh_metadata["skip_face_count_upper_bound"] = skip_face_count_upper_bound
        return _annotated_entry(entry, remesh_metadata), _automatic_outcome(
            "skipped_not_needed",
            source_face_count=source_face_count,
            target_face_count=target_face_count,
            target_face_tolerance=target_face_tolerance,
            skip_face_count_upper_bound=skip_face_count_upper_bound,
        )

    output_dir = _asset_root(provenance["base_runtime_path"]) / "remesh" / f"auto_faces_{target_face_count}"
    try:
        report = _run_integrated_remesh(
            entry=entry,
            provenance=provenance,
            output_dir=output_dir,
            mode="automatic",
            target_face_count=target_face_count,
            target_edge_length=None,
            target_face_tolerance=target_face_tolerance,
        )
    except Exception as exc:  # noqa: BLE001
        failure = _exception_report(output_dir, exc, mode="automatic")
        updated = _entry_with_automatic_fallback(entry, provenance=provenance, report=failure)
        return updated, _automatic_outcome(
            "failed_fallback_original",
            fallback_used=True,
            error=failure["error"],
            report_path=failure["report_path"],
        )

    if not report.get("ok") or not _entry_specific_validation_ok(entry, report):
        updated = _entry_with_automatic_fallback(entry, provenance=provenance, report=report)
        return updated, _automatic_outcome(
            "failed_fallback_original",
            fallback_used=True,
            error=_report_error(report),
            report_path=_report_path(report),
        )

    updated = _entry_from_successful_remesh(
        entry,
        provenance=provenance,
        report=report,
        mode="automatic",
    )
    return updated, _automatic_outcome(
        "applied",
        applied=True,
        source_face_count=report.get("source", {}).get("face_count"),
        output_face_count=report.get("output", {}).get("face_count"),
        report_path=_report_path(report),
    )


def _automatic_outcome(
    status: str,
    *,
    applied: bool = False,
    fallback_used: bool = False,
    **details: Any,
) -> dict[str, Any]:
    return {"ok": True, "status": status, "applied": applied, "fallback_used": fallback_used, **details}


def remesh_mesh_assets_for_episode(
    *,
    case_dir: Path,
    asset_names: list[str] | None,
    target_face_count: int | None,
    target_edge_length: float | None,
    target_face_tolerance: float | None = None,
) -> dict[str, Any]:
    """Planner-facing remesh action that changes manifest paths only after complete validation."""

    case_dir = case_dir.resolve()
    manifest_path = case_dir / "assets" / "asset_manifest.json"
    report_path = case_dir / "reports" / "mesh_remesh_report.json"
    manifest = load_json_object(manifest_path)
    selected_names = sorted({name.strip() for name in asset_names or [] if name.strip()})
    tolerance = (
        float(CONFIGS.mesh_repair.auto_remesh_target_face_tolerance)
        if target_face_tolerance is None
        else float(target_face_tolerance)
    )

    precondition_error = _planner_precondition_error(
        manifest=manifest,
        selected_names=selected_names,
        target_face_count=target_face_count,
        target_edge_length=target_edge_length,
        target_face_tolerance=tolerance,
    )
    if precondition_error is not None:
        report = _episode_report(
            ok=False,
            status="precondition_failed",
            message=precondition_error,
            manifest_path=manifest_path,
            report_path=report_path,
            selected_names=selected_names,
            entries=[] if manifest is None else _manifest_entries(manifest),
            results=[],
            failure_classes=[REMESH_PRECONDITION_FAILURE_CLASS],
        )
        dump_json(report, report_path)
        return report

    assert manifest is not None
    entries = _manifest_entries(manifest)
    entries_by_name = {str(entry.get("logical_name", "")): entry for entry in entries}
    updated_by_name: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for name in selected_names:
        entry = entries_by_name.get(name)
        result, updated = _planner_remesh_one(
            entry,
            target_face_count=target_face_count,
            target_edge_length=target_edge_length,
            target_face_tolerance=tolerance,
        )
        results.append({"asset_name": name, **result})
        if updated is not None:
            updated_by_name[name] = updated

    if updated_by_name:
        manifest["assets"] = [updated_by_name.get(str(entry.get("logical_name", "")), entry) for entry in entries]
        dump_json(manifest, manifest_path)

    succeeded = sorted(result["asset_name"] for result in results if result.get("ok"))
    failures = [result for result in results if not result.get("ok")]
    ok = not failures
    status = "mesh_assets_remeshed" if ok else "mesh_asset_remesh_partial" if succeeded else "mesh_asset_remesh_failed"
    failure_classes = sorted({str(result.get("failure_class")) for result in failures if result.get("failure_class")})
    report = _episode_report(
        ok=ok,
        status=status,
        message=None
        if ok
        else "One or more remesh requests failed; failed assets retained their current manifest paths.",
        manifest_path=manifest_path,
        report_path=report_path,
        selected_names=selected_names,
        entries=_manifest_entries(manifest),
        results=results,
        failure_classes=failure_classes,
    )
    report["remeshed_asset_names"] = succeeded
    dump_json(report, report_path)
    return report


def _planner_remesh_one(
    entry: dict[str, Any] | None,
    *,
    target_face_count: int | None,
    target_edge_length: float | None,
    target_face_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if entry is None:
        return _failure_result("Asset is missing from the mesh manifest.", REMESH_PRECONDITION_FAILURE_CLASS), None
    if entry.get("source_type") not in {"generated_mesh", "cloth_mesh"}:
        return _failure_result(
            f"source_type={entry.get('source_type')!r} is not a Planner-remeshable generated mesh.",
            REMESH_PRECONDITION_FAILURE_CLASS,
        ), None
    if not _source_geometry_ready(entry):
        return _failure_result(
            "Asset source geometry did not pass manifold/TetGen preflight and cannot be rescued by downsampling.",
            REMESH_PRECONDITION_FAILURE_CLASS,
        ), None
    cloth_validation = (entry.get("validation") or {}).get("cloth_mesh")
    if entry.get("source_type") == "cloth_mesh" and isinstance(cloth_validation, dict):
        if cloth_validation.get("generation") != "meshy":
            return _failure_result(
                "Procedural open cloth meshes are not handled by the closed-manifold remesh tool.",
                REMESH_PRECONDITION_FAILURE_CLASS,
            ), None

    try:
        provenance = _provenance_from_entry(entry)
        source_face_count = _source_face_count(provenance["base_runtime_path"])
        if target_face_count is not None and source_face_count is not None and target_face_count >= source_face_count:
            return _failure_result(
                f"target_face_count ({target_face_count}) must be below source face count ({source_face_count}).",
                REMESH_PRECONDITION_FAILURE_CLASS,
            ), None
        output_dir = _planner_output_dir(
            provenance["base_runtime_path"],
            target_face_count=target_face_count,
            target_edge_length=target_edge_length,
        )
        report = _run_integrated_remesh(
            entry=entry,
            provenance=provenance,
            output_dir=output_dir,
            mode="planner",
            target_face_count=target_face_count,
            target_edge_length=target_edge_length,
            target_face_tolerance=target_face_tolerance,
        )
    except Exception as exc:  # noqa: BLE001
        return _failure_result(f"{type(exc).__name__}: {exc}", REMESH_FAILURE_CLASS), None

    if not report.get("ok") or not _entry_specific_validation_ok(entry, report):
        return {
            **_failure_result(_report_error(report), REMESH_FAILURE_CLASS),
            "report_path": _report_path(report),
            "failure_stage": report.get("failure_stage") or "entry_validation",
        }, None

    updated = _entry_from_successful_remesh(entry, provenance=provenance, report=report, mode="planner")
    return {
        "ok": True,
        "status": "applied",
        "report_path": _report_path(report),
        "source_face_count": report.get("source", {}).get("face_count"),
        "output_face_count": report.get("output", {}).get("face_count"),
        "failure_class": None,
        "error": None,
    }, updated


def _run_integrated_remesh(
    *,
    entry: dict[str, Any],
    provenance: dict[str, Any],
    output_dir: Path,
    mode: str,
    target_face_count: int | None,
    target_edge_length: float | None,
    target_face_tolerance: float,
) -> dict[str, Any]:
    report = remesh_mesh_asset(
        IsotropicRemeshConfig(
            input_mesh_path=Path(provenance["base_runtime_path"]),
            output_dir=output_dir,
            target_face_count=target_face_count,
            target_edge_length=target_edge_length,
            target_face_tolerance=target_face_tolerance,
            max_search_attempts=int(CONFIGS.mesh_repair.auto_remesh_max_search_attempts),
            iterations=int(CONFIGS.mesh_repair.auto_remesh_iterations),
            source_textured_mesh_path=_optional_path(provenance.get("source_textured_mesh_path")),
            source_base_color_path=_optional_path(provenance.get("source_base_color_path")),
            alignment_translation=_optional_vector3(provenance.get("alignment_translation")),
            scale=entry.get("scale") or 1.0,
            file_meshes_are_zup=bool(entry.get("file_meshes_are_zup", False)),
            validate_genesis=True,
            tet_resolution=int(CONFIGS.deformable.tet_resolution),
        )
    )
    report["standalone"] = False
    report["pipeline_integrated"] = True
    report["integration_mode"] = mode
    report_path = _report_path(report)
    if report_path:
        dump_json(report, Path(report_path))
    return report


def _entry_from_successful_remesh(
    entry: dict[str, Any],
    *,
    provenance: dict[str, Any],
    report: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    updated = deepcopy(entry)
    artifacts = report["artifacts"]
    runtime_path = str(Path(artifacts["runtime_mesh"]).resolve())
    visual_value = artifacts.get("visual_mesh")
    texture_value = artifacts.get("base_color_texture")
    updated["runtime_path"] = runtime_path
    updated["visual_path"] = str(Path(visual_value).resolve()) if visual_value else runtime_path
    updated["texture_path"] = str(Path(texture_value).resolve()) if texture_value else None
    updated["bbox"] = _scaled_genesis_bbox(report, updated)

    validation = deepcopy(updated.get("validation") or {})
    validation["remesh"] = {
        "ok": True,
        "mode": mode,
        "target_check": report.get("target_check"),
        "manifold": report.get("manifold_validation"),
        "texture": report.get("texture_validation"),
        "genesis_imports": report.get("genesis_fem_import_validation"),
    }
    genesis_validation = report.get("genesis_fem_import_validation") or {}
    if updated.get("source_type") == "cloth_mesh":
        cloth = deepcopy(validation.get("cloth_mesh") or {})
        cloth.update(
            {
                "ok": True,
                "face_count": report.get("output", {}).get("face_count"),
                "manifold": report.get("manifold_validation"),
                "genesis_cloth_import": genesis_validation.get("cloth_import"),
            }
        )
        validation["cloth_mesh"] = cloth
    else:
        validation["manifold"] = report.get("manifold_validation")
        validation["genesis_fem_import"] = genesis_validation.get("volumetric_fem_import")
    updated["validation"] = validation
    updated["remesh"] = _remesh_metadata(
        mode=mode,
        status="applied",
        provenance=provenance,
        target_face_count=report.get("request", {}).get("target_face_count"),
        target_edge_length=report.get("request", {}).get("target_edge_length"),
        target_face_tolerance=report.get("request", {}).get("target_face_tolerance"),
        report=report,
        fallback_used=False,
    )
    notes = list(updated.get("notes") or [])
    notes.append(
        f"Local isotropic remesh ({mode}) passed target, manifold/TetGen, texture, and rigid/FEM/cloth import checks; "
        "manifest geometry paths now reference the validated remesh output."
    )
    updated["notes"] = notes
    updated["status"] = "ready"
    return updated


def _entry_with_automatic_fallback(
    entry: dict[str, Any],
    *,
    provenance: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    metadata = _remesh_metadata(
        mode="automatic",
        status="failed_fallback_original",
        provenance=provenance,
        target_face_count=int(CONFIGS.mesh_repair.auto_remesh_target_face_count),
        target_edge_length=None,
        target_face_tolerance=float(CONFIGS.mesh_repair.auto_remesh_target_face_tolerance),
        report=report,
        fallback_used=True,
    )
    return _annotated_entry(
        entry,
        metadata,
        "Automatic local remesh failed validation; the original runtime/visual/texture paths were retained for their "
        "normal Genesis validation.",
    )


def _provenance_from_entry(entry: dict[str, Any], *, bundle: Any | None = None) -> dict[str, Any]:
    if bundle is not None:
        texture = getattr(bundle, "texture", None)
        repair = getattr(bundle, "repair", None)
        texture_ok = texture is not None and getattr(texture, "ok", False)
        return _provenance(
            base_runtime_path=entry.get("runtime_path"),
            base_visual_path=entry.get("visual_path"),
            base_texture_path=entry.get("texture_path"),
            source_textured_mesh_path=getattr(texture, "textured_mesh_path", None) if texture_ok else None,
            source_base_color_path=getattr(texture, "texture_paths", {}).get("base_color") if texture_ok else None,
            alignment_translation=getattr(repair, "centroid_before_translation", None),
        )

    existing = entry.get("remesh")
    if isinstance(existing, dict) and existing.get("base_runtime_path"):
        return _provenance(
            base_runtime_path=existing.get("base_runtime_path"),
            base_visual_path=existing.get("base_visual_path"),
            base_texture_path=existing.get("base_texture_path"),
            source_textured_mesh_path=existing.get("source_textured_mesh_path"),
            source_base_color_path=existing.get("source_base_color_path"),
            alignment_translation=existing.get("alignment_translation"),
        )

    runtime_path = Path(str(entry.get("runtime_path", ""))).resolve()
    metadata = load_json_object(_asset_root(runtime_path) / "metadata.json") or {}
    texture = metadata.get("texture") if isinstance(metadata.get("texture"), dict) else {}
    repair = metadata.get("repair") if isinstance(metadata.get("repair"), dict) else {}
    texture_paths = texture.get("texture_paths") if isinstance(texture.get("texture_paths"), dict) else {}
    source_textured_mesh_path = texture.get("textured_mesh_path")
    source_base_color_path = texture_paths.get("base_color")
    if entry.get("texture_path") and not (source_textured_mesh_path and source_base_color_path):
        raise RuntimeError(
            "Textured asset lacks original Meshy texture provenance; refusing remesh rather than dropping or "
            "incorrectly rebaking its texture."
        )
    return _provenance(
        base_runtime_path=runtime_path,
        base_visual_path=entry.get("visual_path"),
        base_texture_path=entry.get("texture_path"),
        source_textured_mesh_path=source_textured_mesh_path,
        source_base_color_path=source_base_color_path,
        alignment_translation=repair.get("centroid_before_translation"),
    )


def _provenance(
    *,
    base_runtime_path: Any,
    base_visual_path: Any,
    base_texture_path: Any,
    source_textured_mesh_path: Any,
    source_base_color_path: Any,
    alignment_translation: Any,
) -> dict[str, Any]:
    runtime_path = Path(str(base_runtime_path)).resolve()
    if not runtime_path.is_file():
        raise FileNotFoundError(runtime_path)
    source_textured = _optional_path(source_textured_mesh_path)
    source_texture = _optional_path(source_base_color_path)
    if (source_textured is None) != (source_texture is None):
        raise RuntimeError("Texture remesh provenance must contain both source mesh and base-color texture.")
    if source_textured is not None and (not source_textured.is_file() or not source_texture.is_file()):
        raise FileNotFoundError(source_textured if not source_textured.is_file() else source_texture)
    alignment = _optional_vector3(alignment_translation)
    return {
        "base_runtime_path": str(runtime_path),
        "base_visual_path": _resolved_text(base_visual_path),
        "base_texture_path": _resolved_text(base_texture_path),
        "source_textured_mesh_path": None if source_textured is None else str(source_textured.resolve()),
        "source_base_color_path": None if source_texture is None else str(source_texture.resolve()),
        "alignment_translation": None if alignment is None else list(alignment),
    }


def _annotated_entry(
    entry: dict[str, Any],
    remesh_metadata: dict[str, Any],
    note: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(entry)
    updated["remesh"] = remesh_metadata
    if note:
        updated["notes"] = [*(updated.get("notes") or []), note]
    return updated


def _remesh_metadata(
    *,
    mode: str,
    status: str,
    provenance: dict[str, Any],
    target_face_count: int | None,
    target_edge_length: float | None,
    target_face_tolerance: float | None,
    report: dict[str, Any] | None,
    fallback_used: bool,
) -> dict[str, Any]:
    return {
        **provenance,
        "mode": mode,
        "status": status,
        "attempted": status not in {"skipped_disabled", "skipped_not_needed"},
        "applied": status == "applied",
        "fallback_used": fallback_used,
        "target_face_count": target_face_count,
        "target_edge_length": target_edge_length,
        "target_face_tolerance": target_face_tolerance,
        "source_face_count": None if report is None else report.get("source", {}).get("face_count"),
        "output_face_count": None if report is None else report.get("output", {}).get("face_count"),
        "failure_stage": None if report is None else report.get("failure_stage"),
        "report_path": None if report is None else _report_path(report),
    }


def _entry_specific_validation_ok(entry: dict[str, Any], report: dict[str, Any]) -> bool:
    output_faces = report.get("output", {}).get("face_count")
    if (
        entry.get("source_type") != "cloth_mesh"
        or not isinstance(output_faces, int)
        or output_faces <= int(CONFIGS.deformable.cloth_max_faces)
    ):
        return bool(report.get("ok"))
    report["failure_stage"] = "cloth_face_budget"
    report["integration_error"] = (
        f"Remeshed cloth face count {output_faces} exceeds configured max {CONFIGS.deformable.cloth_max_faces}."
    )
    report["ok"] = False
    if report_path := _report_path(report):
        dump_json(report, Path(report_path))
    return False


def _source_geometry_ready(entry: dict[str, Any]) -> bool:
    validation = entry.get("validation") if isinstance(entry.get("validation"), dict) else {}
    if entry.get("source_type") == "cloth_mesh":
        cloth = validation.get("cloth_mesh") if isinstance(validation.get("cloth_mesh"), dict) else {}
        manifold = cloth.get("manifold") if isinstance(cloth.get("manifold"), dict) else {}
    else:
        manifold = validation.get("manifold") if isinstance(validation.get("manifold"), dict) else {}
    return bool(manifold.get("ok")) and manifold.get("tetgen_ready") is not False


def _scaled_genesis_bbox(report: dict[str, Any], entry: dict[str, Any]) -> list[float] | None:
    output = report.get("output") or {}
    bbox_min = output.get("bbox_min")
    bbox_max = output.get("bbox_max")
    if not (isinstance(bbox_min, list) and isinstance(bbox_max, list) and len(bbox_min) == len(bbox_max) == 3):
        return entry.get("bbox")
    size = [float(high) - float(low) for low, high in zip(bbox_min, bbox_max, strict=True)]
    if not bool(entry.get("file_meshes_are_zup", False)):
        size = [size[0], size[2], size[1]]
    scale = entry.get("scale")
    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        return [value * float(scale) for value in size]
    if isinstance(scale, list) and len(scale) == 3:
        return [value * float(axis_scale) for value, axis_scale in zip(size, scale, strict=True)]
    return size


def _planner_precondition_error(
    *,
    manifest: dict[str, Any] | None,
    selected_names: list[str],
    target_face_count: int | None,
    target_edge_length: float | None,
    target_face_tolerance: float,
) -> str | None:
    if manifest is None or not isinstance(manifest.get("assets"), list):
        return "assets/asset_manifest.json is missing or invalid."
    if not selected_names:
        return "remesh_mesh_assets requires one or more explicit asset_names."
    if int(target_face_count is not None) + int(target_edge_length is not None) != 1:
        return "Specify exactly one of target_face_count or target_edge_length."
    if target_face_count is not None and target_face_count < 4:
        return "target_face_count must be at least 4."
    if target_edge_length is not None and target_edge_length <= 0.0:
        return "target_edge_length must be positive."
    if not 0.0 < target_face_tolerance < 1.0:
        return "target_face_tolerance must be between 0 and 1."
    return None


def _episode_report(
    *,
    ok: bool,
    status: str,
    message: str | None,
    manifest_path: Path,
    report_path: Path,
    selected_names: list[str],
    entries: list[dict[str, Any]],
    results: list[dict[str, Any]],
    failure_classes: list[str],
) -> dict[str, Any]:
    return {
        "ok": ok,
        "status": status,
        "message": message,
        "asset_manifest_path": str(manifest_path),
        "asset_generation_report_path": str(report_path),
        "num_assets": len(entries),
        "selected_asset_names": selected_names,
        "skipped_asset_names": [],
        "failure_classes": failure_classes,
        "assets": results,
    }


def _exception_report(output_dir: Path, exc: Exception, *, mode: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "remesh_report.json"
    report = {
        "ok": False,
        "standalone": False,
        "pipeline_integrated": True,
        "integration_mode": mode,
        "failure_stage": "exception",
        "error": f"{type(exc).__name__}: {exc}",
        "report_path": str(report_path),
    }
    dump_json(report, report_path)
    return report


def _failure_result(error: str, failure_class: str) -> dict[str, Any]:
    return {
        "ok": False,
        "status": "failed",
        "error": error,
        "failure_class": failure_class,
    }


def _report_error(report: dict[str, Any]) -> str:
    return str(
        report.get("integration_error")
        or report.get("error")
        or (report.get("manifold_validation") or {}).get("error")
        or f"Remesh failed at {report.get('failure_stage', 'validation')}."
    )


def _report_path(report: dict[str, Any]) -> str | None:
    artifacts = report.get("artifacts")
    if isinstance(artifacts, dict) and artifacts.get("report"):
        return str(artifacts["report"])
    value = report.get("report_path")
    return str(value) if value else None


def _manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in manifest.get("assets", []) if isinstance(entry, dict)]


def _source_face_count(runtime_path: str) -> int | None:
    """Read the current runtime mesh directly; never use stale summary metadata for auto-skip."""

    try:
        mesh = trimesh.load_mesh(runtime_path, force="mesh", process=False, skip_texture=True)
    except Exception:  # noqa: BLE001
        return None
    return len(mesh.faces) if isinstance(mesh, trimesh.Trimesh) else None


def _asset_root(runtime_path: str | Path) -> Path:
    path = Path(runtime_path).resolve()
    if path.parent.name == "processed":
        return path.parent.parent
    remesh_parent = next((parent for parent in path.parents if parent.name == "remesh"), None)
    if remesh_parent is not None:
        return remesh_parent.parent
    return path.parent


def _planner_output_dir(
    runtime_path: str,
    *,
    target_face_count: int | None,
    target_edge_length: float | None,
) -> Path:
    if target_face_count is not None:
        label = f"planner_faces_{target_face_count}"
    else:
        edge_text = re.sub(r"[^0-9A-Za-z]+", "_", f"{target_edge_length:.9g}").strip("_")
        label = f"planner_edge_{edge_text}"
    return _asset_root(runtime_path) / "remesh" / label


def _resolved_text(value: Any) -> str | None:
    path = _optional_path(value)
    return None if path is None else str(path)


def _optional_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).resolve()


def _optional_vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return float(value[0]), float(value[1]), float(value[2])
