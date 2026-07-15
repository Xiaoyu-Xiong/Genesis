from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from code_agent.configs import CONFIGS

from .models import MeshGenesisClothImportResult, MeshGenesisFEMImportResult


def run_genesis_fem_import_validation(
    manifest_entry: dict[str, Any],
    *,
    tet_resolution: int = CONFIGS.deformable.tet_resolution,
    timeout_sec: float = CONFIGS.mesh_repair.genesis_fem_import_timeout_sec,
) -> MeshGenesisFEMImportResult:
    """Validate that a generated mesh can enter the Genesis FEM mesh import path."""

    runtime_path = Path(str(manifest_entry.get("runtime_path", "")))
    visual_path_raw = manifest_entry.get("visual_path")
    visual_path = Path(str(visual_path_raw)) if visual_path_raw else None
    scale = _scale_tuple(manifest_entry.get("scale"))
    file_meshes_are_zup = manifest_entry.get("file_meshes_are_zup")
    payload = {
        "runtime_path": str(runtime_path),
        "visual_path": None if visual_path is None else str(visual_path),
        "scale": scale,
        "file_meshes_are_zup": file_meshes_are_zup,
        "tet_resolution": int(tet_resolution),
    }

    if not runtime_path.is_file():
        return MeshGenesisFEMImportResult(
            ok=False,
            runtime_path=runtime_path,
            visual_path=visual_path,
            scale=scale,
            file_meshes_are_zup=_optional_bool(file_meshes_are_zup),
            tet_resolution=int(tet_resolution),
            error=f"Runtime mesh path does not exist: {runtime_path}",
        )

    try:
        result = subprocess.run(
            [sys.executable, "-c", _GENESIS_FEM_IMPORT_PROBE, json.dumps(payload)],
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return MeshGenesisFEMImportResult(
            ok=False,
            runtime_path=runtime_path,
            visual_path=visual_path,
            scale=scale,
            file_meshes_are_zup=_optional_bool(file_meshes_are_zup),
            tet_resolution=int(tet_resolution),
            stdout_tail=_tail(exc.stdout or ""),
            stderr_tail=_tail(exc.stderr or ""),
            error=f"Genesis FEM import probe timed out after {timeout_sec} seconds.",
        )
    stdout_tail = _tail(result.stdout)
    stderr_tail = _tail(result.stderr)
    if result.returncode != 0:
        return MeshGenesisFEMImportResult(
            ok=False,
            runtime_path=runtime_path,
            visual_path=visual_path,
            scale=scale,
            file_meshes_are_zup=_optional_bool(file_meshes_are_zup),
            tet_resolution=int(tet_resolution),
            returncode=result.returncode,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            error=f"Genesis FEM import probe failed with exit code {result.returncode}.",
        )

    try:
        probe = _last_json_object(result.stdout)
    except Exception as exc:
        return MeshGenesisFEMImportResult(
            ok=False,
            runtime_path=runtime_path,
            visual_path=visual_path,
            scale=scale,
            file_meshes_are_zup=_optional_bool(file_meshes_are_zup),
            tet_resolution=int(tet_resolution),
            returncode=result.returncode,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            error=f"Unable to parse Genesis FEM import probe output: {type(exc).__name__}: {exc}",
        )

    texture_path_raw = probe.get("texture_path")
    return MeshGenesisFEMImportResult(
        ok=bool(probe.get("ok")),
        runtime_path=runtime_path,
        visual_path=visual_path,
        scale=scale,
        file_meshes_are_zup=_optional_bool(file_meshes_are_zup),
        tet_resolution=int(tet_resolution),
        vertex_count=int(probe.get("vertex_count") or 0),
        element_count=int(probe.get("element_count") or 0),
        surface_vertex_count=int(probe.get("surface_vertex_count") or 0),
        surface_visual_uv_shape=_shape_tuple(probe.get("surface_visual_uv_shape")),
        render_vertex_count=_optional_int(probe.get("render_vertex_count")),
        render_face_count=_optional_int(probe.get("render_face_count")),
        texture_path=Path(str(texture_path_raw)) if texture_path_raw else None,
        returncode=result.returncode,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        error=probe.get("error"),
    )


def run_genesis_cloth_import_validation(
    manifest_entry: dict[str, Any],
    *,
    timeout_sec: float = CONFIGS.mesh_repair.genesis_fem_import_timeout_sec,
) -> MeshGenesisClothImportResult:
    """Validate that a generated closed surface mesh can enter the Genesis FEM.Cloth path."""

    runtime_path = Path(str(manifest_entry.get("runtime_path", "")))
    visual_path_raw = manifest_entry.get("visual_path")
    visual_path = Path(str(visual_path_raw)) if visual_path_raw else None
    scale = _scale_tuple(manifest_entry.get("scale"))
    file_meshes_are_zup = manifest_entry.get("file_meshes_are_zup")
    payload = {
        "runtime_path": str(runtime_path),
        "visual_path": None if visual_path is None else str(visual_path),
        "scale": scale,
        "file_meshes_are_zup": file_meshes_are_zup,
    }

    if not runtime_path.is_file():
        return MeshGenesisClothImportResult(
            ok=False,
            runtime_path=runtime_path,
            visual_path=visual_path,
            scale=scale,
            file_meshes_are_zup=_optional_bool(file_meshes_are_zup),
            error=f"Runtime mesh path does not exist: {runtime_path}",
        )

    try:
        result = subprocess.run(
            [sys.executable, "-c", _GENESIS_CLOTH_IMPORT_PROBE, json.dumps(payload)],
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return MeshGenesisClothImportResult(
            ok=False,
            runtime_path=runtime_path,
            visual_path=visual_path,
            scale=scale,
            file_meshes_are_zup=_optional_bool(file_meshes_are_zup),
            stdout_tail=_tail(exc.stdout or ""),
            stderr_tail=_tail(exc.stderr or ""),
            error=f"Genesis FEM.Cloth import probe timed out after {timeout_sec} seconds.",
        )
    stdout_tail = _tail(result.stdout)
    stderr_tail = _tail(result.stderr)
    if result.returncode != 0:
        return MeshGenesisClothImportResult(
            ok=False,
            runtime_path=runtime_path,
            visual_path=visual_path,
            scale=scale,
            file_meshes_are_zup=_optional_bool(file_meshes_are_zup),
            returncode=result.returncode,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            error=f"Genesis FEM.Cloth import probe failed with exit code {result.returncode}.",
        )

    try:
        probe = _last_json_object(result.stdout)
    except Exception as exc:
        return MeshGenesisClothImportResult(
            ok=False,
            runtime_path=runtime_path,
            visual_path=visual_path,
            scale=scale,
            file_meshes_are_zup=_optional_bool(file_meshes_are_zup),
            returncode=result.returncode,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            error=f"Unable to parse Genesis FEM.Cloth import probe output: {type(exc).__name__}: {exc}",
        )

    return MeshGenesisClothImportResult(
        ok=bool(probe.get("ok")),
        runtime_path=runtime_path,
        visual_path=visual_path,
        scale=scale,
        file_meshes_are_zup=_optional_bool(file_meshes_are_zup),
        vertex_count=int(probe.get("vertex_count") or 0),
        element_count=int(probe.get("element_count") or 0),
        surface_vertex_count=int(probe.get("surface_vertex_count") or 0),
        surface_face_count=int(probe.get("surface_face_count") or 0),
        surface_visual_uv_shape=_shape_tuple(probe.get("surface_visual_uv_shape")),
        returncode=result.returncode,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        error=probe.get("error"),
    )


def _scale_tuple(value: object) -> tuple[float, float, float] | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        scale = float(value)
        return (scale, scale, scale) if scale > 0.0 else None
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _shape_tuple(value: object) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tail(text: str | bytes, *, max_chars: int = 4000) -> str | None:
    if not text:
        return None
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text[-max_chars:]


def _last_json_object(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found in subprocess stdout.")


_GENESIS_FEM_IMPORT_PROBE = r"""
import json
import sys

payload = json.loads(sys.argv[1])

import genesis as gs

gs.init(backend=gs.cpu, precision="32", performance_mode=True, logging_level="warning")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0), floor_height=-2.0),
    fem_options=gs.options.FEMOptions(floor_height=-2.0),
    show_viewer=False,
    show_FPS=False,
)
entity = scene.add_entity(
    morph=gs.morphs.Mesh(
        file=payload["runtime_path"],
        scale=payload["scale"] or 1.0,
        file_meshes_are_zup=payload["file_meshes_are_zup"],
        tet_resolution=int(payload["tet_resolution"]),
        convexify=False,
        decimate=False,
    ),
    material=gs.materials.FEM.Elastic(E=1.0e5, nu=0.35, rho=1000.0, model="stable_neohookean"),
    surface=gs.surfaces.Default(vis_mode="visual", smooth=True),
    name="mesh_agent_fem_import_validation",
)

from genesis.utils import element as eu

artifact = eu.get_mesh_to_elements_render_artifact(
    payload["runtime_path"],
    payload["scale"] or 1.0,
    entity.tet_cfg,
    payload["file_meshes_are_zup"],
)
artifact = artifact if isinstance(artifact, dict) else {}
render_faces = artifact.get("render_faces")
render_indices = artifact.get("render_vertex_src_indices")
surface_uvs = entity.surface_visual_uvs
texture_path = artifact.get("texture_path")
error = None
if payload.get("visual_path") and payload.get("visual_path") != payload["runtime_path"] and render_indices is None:
    error = "Genesis FEM import succeeded, but no seam-aware render artifact was registered for visual_path."

print(json.dumps({
    "ok": error is None,
    "vertex_count": int(entity.n_vertices),
    "element_count": int(entity.n_elements),
    "surface_vertex_count": int(entity.n_surface_vertices),
    "surface_visual_uv_shape": None if surface_uvs is None else list(surface_uvs.shape),
    "render_vertex_count": None if render_indices is None else int(len(render_indices)),
    "render_face_count": None if render_faces is None else int(len(render_faces)),
    "texture_path": texture_path,
    "error": error,
}))
"""


_GENESIS_CLOTH_IMPORT_PROBE = r"""
import json
import sys

payload = json.loads(sys.argv[1])

import genesis as gs

gs.init(backend=gs.gpu, precision="32", performance_mode=True, logging_level="warning")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0), floor_height=-2.0),
    fem_options=gs.options.FEMOptions(floor_height=-2.0),
    coupler_options=gs.options.IPCCouplerOptions(contact_enable=True, enable_rigid_rigid_contact=True),
    show_viewer=False,
    show_FPS=False,
)
entity = scene.add_entity(
    morph=gs.morphs.Mesh(
        file=payload["runtime_path"],
        scale=payload["scale"] or 1.0,
        file_meshes_are_zup=payload["file_meshes_are_zup"],
        convexify=False,
        decimate=False,
    ),
    material=gs.materials.FEM.Cloth(
        E=1.0e4,
        nu=0.3,
        rho=200.0,
        thickness=0.001,
        bending_stiffness=None,
        friction_mu=0.5,
    ),
    surface=gs.surfaces.Default(vis_mode="visual", smooth=True),
    name="mesh_agent_fem_cloth_import_validation",
)

scene.build()
surface_uvs = entity.surface_visual_uvs
error = None

print(json.dumps({
    "ok": error is None,
    "vertex_count": int(entity.n_vertices),
    "element_count": int(entity.n_elements),
    "surface_vertex_count": int(entity.n_surface_vertices),
    "surface_face_count": int(entity.n_surfaces),
    "surface_visual_uv_shape": None if surface_uvs is None else list(surface_uvs.shape),
    "error": error,
}))
"""
