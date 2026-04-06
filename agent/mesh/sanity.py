from __future__ import annotations

from pathlib import Path

import trimesh

from .models import MeshManifoldCheckResult


def run_mesh_manifold_check(mesh_path: Path, *, face_cap_for_full_check: int = 100000) -> MeshManifoldCheckResult:
    quick_stats = _quick_mesh_stats(mesh_path)
    if quick_stats is not None and quick_stats["face_count"] > face_cap_for_full_check:
        return MeshManifoldCheckResult(
            ok=False,
            mesh_path=mesh_path,
            vertex_count=quick_stats["vertex_count"],
            face_count=quick_stats["face_count"],
            component_count=-1,
            is_watertight=False,
            is_winding_consistent=False,
            volume=None,
            error="Skipped expensive topology check because the raw mesh is too large.",
        )
    try:
        mesh = _load_mesh(mesh_path)
        component_count = _component_count(mesh)
        volume = float(mesh.volume) if mesh.is_watertight else None
        ok = bool(mesh.is_watertight and mesh.is_winding_consistent)
        return MeshManifoldCheckResult(
            ok=ok,
            mesh_path=mesh_path,
            vertex_count=int(len(mesh.vertices)),
            face_count=int(len(mesh.faces)),
            component_count=component_count,
            is_watertight=bool(mesh.is_watertight),
            is_winding_consistent=bool(mesh.is_winding_consistent),
            volume=volume,
        )
    except Exception as exc:
        return MeshManifoldCheckResult(
            ok=False,
            mesh_path=mesh_path,
            vertex_count=0,
            face_count=0,
            component_count=0,
            is_watertight=False,
            is_winding_consistent=False,
            volume=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(mesh_path), force="mesh", skip_texture=True, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh).__name__}")
    return mesh


def _component_count(mesh: trimesh.Trimesh) -> int:
    if len(mesh.faces) > 100000:
        return -1
    return max(1, len(mesh.split(only_watertight=False)))


def _quick_mesh_stats(mesh_path: Path) -> dict[str, int] | None:
    if mesh_path.suffix.lower() != ".obj":
        return None
    vertex_count = 0
    face_count = 0
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("v "):
                vertex_count += 1
            elif line.startswith("f "):
                face_count += 1
    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
    }
