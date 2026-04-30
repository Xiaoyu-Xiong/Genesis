from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import trimesh


def load_mesh_asset_summary(mesh_path: str | Path) -> dict[str, Any]:
    path = Path(mesh_path)
    summary: dict[str, Any] = {
        "mesh_path": str(path),
        "bbox_space": "mesh_local_pre_scale",
    }

    if not path.exists():
        summary["error"] = "mesh file not found"
        return summary

    try:
        summary["mesh_bytes"] = path.stat().st_size
    except Exception:  # noqa: BLE001
        pass

    asset_root = _asset_root_from_mesh_path(path)
    metadata = _load_json_dict(asset_root / "metadata.json")
    raw_manifold = _load_json_dict(asset_root / "raw_manifold_check.json")
    manifold = _load_json_dict(asset_root / "manifold_check.json")

    if metadata is not None:
        repair_payload = metadata.get("repair")
        if isinstance(repair_payload, dict):
            _merge_bbox_fields(summary, repair_payload)
            summary["centroid_before_translation"] = repair_payload.get("centroid_before_translation")
            summary["centroid_at_origin"] = bool(repair_payload.get("centroid_at_origin", False))
        summary["provider"] = metadata.get("provider")
        summary["from_generated_mesh_metadata"] = True

    if raw_manifold is not None:
        summary["raw_manifold_ok"] = bool(raw_manifold.get("ok"))
    if manifold is not None:
        summary["repaired_manifold_ok"] = bool(manifold.get("ok"))
        if "vertex_count" in manifold:
            summary["vertex_count"] = manifold.get("vertex_count")
        if "face_count" in manifold:
            summary["face_count"] = manifold.get("face_count")

    if "bbox_size" in summary:
        return summary

    try:
        mesh = trimesh.load_mesh(str(path), force="mesh", skip_texture=True, process=False)
        if isinstance(mesh, trimesh.Trimesh):
            bounds = mesh.bounds.astype(float, copy=False)
            bbox_min = bounds[0].tolist()
            bbox_max = bounds[1].tolist()
            bbox_size = (bounds[1] - bounds[0]).tolist()
            summary["bbox_min"] = bbox_min
            summary["bbox_max"] = bbox_max
            summary["bbox_size"] = bbox_size
            summary["vertex_count"] = int(len(mesh.vertices))
            summary["face_count"] = int(len(mesh.faces))
    except Exception as exc:  # noqa: BLE001
        summary.setdefault("error", f"{type(exc).__name__}: {exc}")

    return summary


def _asset_root_from_mesh_path(mesh_path: Path) -> Path:
    if mesh_path.parent.name in {"downloads", "processed", "textured"}:
        return mesh_path.parent.parent
    return mesh_path.parent


def _load_json_dict(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _merge_bbox_fields(summary: dict[str, Any], payload: dict[str, object]) -> None:
    for key in ("bbox_min", "bbox_max", "bbox_size"):
        value = payload.get(key)
        if isinstance(value, list) and len(value) == 3:
            summary[key] = value
