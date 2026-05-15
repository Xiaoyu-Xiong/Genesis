from __future__ import annotations

from pathlib import Path
from typing import Any

from code_agent.assets.mesh.request_adapter import vector3
from code_agent.assets.xml.preview import render_xml_preview
from code_agent.io_utils import dump_json, load_json_object


MESH_EXTENSIONS = {".obj", ".stl", ".ply", ".glb", ".gltf"}


def inspect_generated_assets(case_dir: Path, *, asset_names: list[str] | None = None) -> dict[str, Any]:
    """Inspect ready mesh/XML assets and write preview artifacts for planner debugging."""

    case_dir = case_dir.resolve()
    manifest_path = case_dir / "assets" / "asset_manifest.json"
    output_dir = case_dir / "reports" / "asset_inspection"
    report_path = case_dir / "reports" / "asset_inspection_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "ok": False,
        "status": "asset_inspection_complete",
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "selected_asset_names": asset_names or [],
        "available_asset_names": [],
        "assets": [],
        "errors": [],
        "warnings": [],
    }

    manifest = load_json_object(manifest_path)
    if manifest is None:
        report["status"] = "precondition_failed"
        report["errors"].append(f"Asset manifest missing or invalid JSON: {manifest_path}")
        dump_json(report, report_path)
        return report

    entries = [entry for entry in manifest.get("assets", []) if isinstance(entry, dict)]
    report["available_asset_names"] = sorted(
        str(entry.get("logical_name")) for entry in entries if entry.get("logical_name")
    )
    selected_names = {name for name in (asset_names or []) if name}
    selected_entries = [
        entry for entry in entries if not selected_names or str(entry.get("logical_name", "")) in selected_names
    ]
    missing = sorted(selected_names - {str(entry.get("logical_name", "")) for entry in entries})
    report["warnings"].extend(f"Requested asset not found in manifest: {name}" for name in missing)

    for entry in selected_entries:
        asset_report = _inspect_entry(entry, output_dir)
        report["assets"].append(asset_report)

    asset_errors = [
        error
        for asset_report in report["assets"]
        for error in asset_report.get("errors", [])
        if isinstance(error, str)
    ]
    report["ok"] = not report["errors"] and not asset_errors
    report["asset_error_count"] = len(asset_errors)
    report["asset_warning_count"] = len(report["warnings"]) + sum(
        len(asset_report.get("warnings", []))
        for asset_report in report["assets"]
        if isinstance(asset_report.get("warnings"), list)
    )
    dump_json(report, report_path)
    return report


def _inspect_entry(entry: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    logical_name = str(entry.get("logical_name") or "asset")
    source_type = str(entry.get("source_type") or "")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in logical_name.strip())
    asset_dir = output_dir / (safe_name or "asset")
    asset_dir.mkdir(parents=True, exist_ok=True)
    runtime_path_value = entry.get("runtime_path")
    runtime_path = Path(str(runtime_path_value)) if runtime_path_value else None
    report = {
        "logical_name": logical_name,
        "source_type": source_type,
        "runtime_path": str(runtime_path) if runtime_path is not None else None,
        "status": entry.get("status"),
        "preview_paths": [],
        "geometry": None,
        "errors": [],
        "warnings": [],
    }
    if runtime_path is None:
        report["warnings"].append(f"No preview renderer for source_type={source_type!r} without runtime_path.")
        return report
    if not runtime_path.exists():
        report["errors"].append(f"runtime_path does not exist: {runtime_path}")
        return report

    suffix = runtime_path.suffix.lower()
    if source_type == "generated_mesh" or suffix in MESH_EXTENSIONS:
        _inspect_mesh_entry(entry, runtime_path, asset_dir, report)
    elif source_type in {"mjcf", "urdf"} or suffix == ".xml":
        _inspect_xml_entry(runtime_path, asset_dir, report)
    else:
        report["warnings"].append(f"No preview renderer for source_type={source_type!r}.")
    return report


def _inspect_mesh_entry(entry: dict[str, Any], runtime_path: Path, asset_dir: Path, report: dict[str, Any]) -> None:
    try:
        import numpy as np
        import trimesh
    except Exception as exc:
        report["errors"].append(f"Mesh inspection imports failed: {type(exc).__name__}: {exc}")
        return

    try:
        mesh = _load_as_trimesh(runtime_path, trimesh)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        raw_zup = entry.get("file_meshes_are_zup")
        file_meshes_are_zup = True if raw_zup is None else bool(raw_zup)
        vertices = _vertices_in_genesis_frame(vertices, file_meshes_are_zup)
        scale = _scale_vector(entry.get("scale")) or [1.0, 1.0, 1.0]
        vertices = vertices * np.asarray(scale, dtype=np.float64)
        bounds = _bounds(vertices)
        extents = bounds[1] - bounds[0] if bounds is not None else np.zeros(3, dtype=np.float64)
        components = _component_count(mesh)
        geometry = {
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "bounds_min": bounds[0].tolist() if bounds is not None else None,
            "bounds_max": bounds[1].tolist() if bounds is not None else None,
            "extents": extents.tolist(),
            "scale_applied": scale,
            "file_meshes_are_zup": file_meshes_are_zup,
            "is_watertight": bool(getattr(mesh, "is_watertight", False)),
            "is_winding_consistent": bool(getattr(mesh, "is_winding_consistent", False)),
            "euler_number": _safe_int(getattr(mesh, "euler_number", None)),
            "volume": _safe_float(getattr(mesh, "volume", None)),
            "component_count": components,
        }
        report["geometry"] = geometry
        if len(vertices) == 0 or len(faces) == 0:
            report["errors"].append("Mesh has no vertices or faces.")
        if not geometry["is_watertight"]:
            report["warnings"].append("Mesh is not watertight; topology may be unsuitable for IPC/FEM contact.")
        genesis_views = _render_genesis_mesh_views(
            entry=entry,
            runtime_path=runtime_path,
            asset_dir=asset_dir,
            scale=scale,
            file_meshes_are_zup=file_meshes_are_zup,
        )
        contact_sheet = _write_contact_sheet(genesis_views, asset_dir / "genesis_preview" / "contact_sheet.png")
        if contact_sheet is not None:
            genesis_views["contact_sheet"] = str(contact_sheet)
        report["renderer"] = "genesis.Rasterizer"
        report["preview_paths"].extend(genesis_views.values())
        report["preview_views"] = genesis_views
    except Exception as exc:
        report["errors"].append(f"Mesh inspection failed: {type(exc).__name__}: {exc}")


def _inspect_xml_entry(runtime_path: Path, asset_dir: Path, report: dict[str, Any]) -> None:
    if runtime_path.suffix.lower() != ".xml":
        report["warnings"].append("XML preview currently supports MJCF .xml assets only.")
        return
    try:
        preview_report = render_xml_preview(runtime_path, asset_dir / "xml_preview")
    except Exception as exc:
        report["errors"].append(f"XML preview failed: {type(exc).__name__}: {exc}")
        return
    report["geometry"] = preview_report.get("model_summary")
    report["preview_report_path"] = str((asset_dir / "xml_preview" / "preview_report.json").resolve())
    for view in preview_report.get("views", []):
        if isinstance(view, dict) and isinstance(view.get("image_path"), str):
            report["preview_paths"].append(view["image_path"])
    report["warnings"].extend(str(item) for item in preview_report.get("warnings", []) if item)
    if not preview_report.get("ok"):
        report["errors"].extend(str(item) for item in preview_report.get("errors", []) if item)


def _load_as_trimesh(runtime_path: Path, trimesh_module: Any) -> Any:
    loaded = trimesh_module.load(runtime_path, force="scene", process=False)
    if isinstance(loaded, trimesh_module.Trimesh):
        return loaded
    geometry = [geom for geom in getattr(loaded, "geometry", {}).values() if isinstance(geom, trimesh_module.Trimesh)]
    if not geometry:
        raise ValueError(f"No Trimesh geometry found in {runtime_path}")
    return trimesh_module.util.concatenate(geometry)


def _vertices_in_genesis_frame(vertices: Any, file_meshes_are_zup: bool) -> Any:
    if file_meshes_are_zup:
        return vertices
    transformed = vertices.copy()
    transformed[:, 0] = vertices[:, 0]
    transformed[:, 1] = -vertices[:, 2]
    transformed[:, 2] = vertices[:, 1]
    return transformed


def _bounds(vertices: Any) -> Any:
    if len(vertices) == 0:
        return None
    return vertices.min(axis=0), vertices.max(axis=0)


def _component_count(mesh: Any) -> int | None:
    try:
        if len(mesh.faces) > 200_000:
            return None
        return len(mesh.split(only_watertight=False))
    except Exception:
        return None


def _render_genesis_mesh_views(
    *,
    entry: dict[str, Any],
    runtime_path: Path,
    asset_dir: Path,
    scale: list[float],
    file_meshes_are_zup: bool,
) -> dict[str, str]:
    from code_agent.assets.mesh.texture.render_views import render_textured_mesh_views

    visual_path = _existing_path(entry.get("visual_path"))
    texture_path = _existing_path(entry.get("texture_path"))
    return render_textured_mesh_views(
        mesh_path=runtime_path,
        visual_mesh_path=visual_path,
        texture_path=texture_path,
        scale=scale,
        file_meshes_are_zup=file_meshes_are_zup,
        out_dir=asset_dir / "genesis_preview",
        res=(900, 900),
    )


def _write_contact_sheet(views: dict[str, str], output_path: Path) -> Path | None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    ordered_names = [name for name in ("front", "side", "top", "iso") if name in views]
    if not ordered_names:
        return None

    tile_size = 440
    label_height = 28
    columns = 2
    rows = (len(ordered_names) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, name in enumerate(ordered_names):
        image = Image.open(views[name]).convert("RGB")
        image.thumbnail((tile_size, tile_size))
        col = index % columns
        row = index // columns
        x = col * tile_size + (tile_size - image.width) // 2
        y = row * (tile_size + label_height)
        draw.text((col * tile_size + 12, y + 6), name, fill=(20, 20, 20))
        sheet.paste(image, (x, y + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def _scale_vector(value: Any) -> list[float] | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        scale = float(value)
        return [scale, scale, scale] if scale > 0.0 else None
    return vector3(value)


def _existing_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
