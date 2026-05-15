from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import trimesh

from code_agent.configs import CONFIGS

from ..models import MeshManifoldCheckResult
from .components import connected_face_component_count, strip_texture_visuals


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
            tetgen_ready=None,
            tetgen_message=None,
            error="Skipped expensive topology/tetgen check because the raw mesh is too large.",
        )
    try:
        mesh = _load_mesh(mesh_path)
        component_count = _component_count(mesh)
        volume = float(mesh.volume) if mesh.is_watertight else None
        topo_ok = bool(mesh.is_watertight and mesh.is_winding_consistent)
        tetgen_ready = False
        tetgen_message = None
        if topo_ok:
            tetgen_ready, tetgen_message = _run_tetgen_sanity_check(mesh_path)
        else:
            tetgen_message = "Topology invalid; tetgen check skipped."
        ok = bool(topo_ok and tetgen_ready)
        error = None if ok else tetgen_message
        return MeshManifoldCheckResult(
            ok=ok,
            mesh_path=mesh_path,
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            component_count=component_count,
            is_watertight=topo_ok if topo_ok else bool(mesh.is_watertight),
            is_winding_consistent=bool(mesh.is_winding_consistent),
            volume=volume,
            tetgen_ready=tetgen_ready if topo_ok else False,
            tetgen_message=tetgen_message,
            error=error,
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
            tetgen_ready=False,
            tetgen_message=f"{type(exc).__name__}: {exc}",
            error=f"{type(exc).__name__}: {exc}",
        )


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(mesh_path), force="mesh", skip_texture=True, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh).__name__}")
    return strip_texture_visuals(mesh)


def _component_count(mesh: trimesh.Trimesh) -> int:
    return connected_face_component_count(mesh)


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


def _run_tetgen_sanity_check(mesh_path: Path) -> tuple[bool, str | None]:
    probe = """
import sys
import tetgen
import trimesh

mesh = trimesh.load_mesh(sys.argv[1], force='mesh', skip_texture=True, process=False)
tet = tetgen.TetGen(mesh.vertices.astype('float64', copy=False), mesh.faces.astype('int32', copy=False))
tet.tetrahedralize(switches='pq1.1/15')
print('TETGEN_OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(mesh_path)],
        text=True,
        capture_output=True,
        check=False,
        timeout=CONFIGS.mesh_repair.tetgen_sanity_timeout_sec,
    )
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    lowered = combined.lower()
    bad_tokens = (
        "self-intersections",
        "segment and a facet intersect",
        "two facets exactly intersect",
        "input triangles are skipped",
        "runtimeerror",
        "tetgen error",
    )
    if result.returncode == 0 and not any(token in lowered for token in bad_tokens):
        return True, None

    lines = [line.strip() for line in combined.splitlines() if line.strip()]
    if not lines:
        lines = [f"tetgen sanity check failed with exit code {result.returncode}"]
    return False, " | ".join(lines[-12:])
