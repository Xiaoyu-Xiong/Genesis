from __future__ import annotations

import shutil
import time
from pathlib import Path

import trimesh

from ....io_utils import dump_json
from ..models import MeshTextureTransferResult
from .bake import bake_texture_from_source_mesh, read_bake_texture_size
from .obj_io import copy_with_vertex_translation, rewrite_obj_mtllib, write_base_color_mtl
from .parameterization import parameterize_target_mesh_xatlas


def transfer_texture_to_repaired_mesh(
    *,
    source_mesh_path: Path,
    source_base_color_path: Path,
    target_mesh_path: Path,
    output_dir: Path,
    alignment_translation: tuple[float, float, float] | None,
) -> MeshTextureTransferResult:
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = processed_dir / "repaired_texture_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    output_mesh_path = target_mesh_path
    output_mtl_path = processed_dir / "repaired.mtl"
    output_texture_path = processed_dir / "base_color.png"
    aligned_target_path = diagnostics_dir / "repaired_aligned.obj"
    exported_target_path = diagnostics_dir / "repaired_textured_export.obj"
    exported_texture_path = diagnostics_dir / "baked_base_color.png"
    diagnostics_json_path = diagnostics_dir / "transfer_diagnostics.json"

    stage_durations_sec: dict[str, float] = {}
    parameterization_filter: str | None = None
    transfer_filter: str | None = None

    try:
        for stale_path in (
            output_mtl_path,
            output_texture_path,
            aligned_target_path,
            exported_target_path,
            exported_texture_path,
        ):
            if stale_path.exists():
                stale_path.unlink()

        texture_size = read_bake_texture_size(source_base_color_path)
        # The repaired mesh is centered before export; shift it back temporarily so the
        # textured raw mesh and repaired geometry live in the same frame for baking.
        copy_with_vertex_translation(
            src_path=target_mesh_path,
            dst_path=aligned_target_path,
            delta=alignment_translation,
        )

        stage_start = time.monotonic()
        source_mesh = trimesh.load_mesh(str(source_mesh_path), force="mesh", process=False, skip_texture=False)
        if not isinstance(source_mesh, trimesh.Trimesh):
            raise TypeError(f"Expected source mesh to load as Trimesh, got {type(source_mesh).__name__}")
        target_mesh = trimesh.load_mesh(str(aligned_target_path), force="mesh", process=False, skip_texture=True)
        if not isinstance(target_mesh, trimesh.Trimesh):
            raise TypeError(f"Expected target mesh to load as Trimesh, got {type(target_mesh).__name__}")
        stage_durations_sec["load_meshes"] = time.monotonic() - stage_start

        stage_start = time.monotonic()
        parameterization_filter = parameterize_target_mesh_xatlas(
            target_mesh=target_mesh,
            output_mesh_path=exported_target_path,
            texture_size=texture_size,
        )
        stage_durations_sec["parameterize_target_mesh"] = time.monotonic() - stage_start

        stage_start = time.monotonic()
        bake_diagnostics = bake_texture_from_source_mesh(
            source_base_color_path=source_base_color_path,
            target_parameterized_mesh_path=exported_target_path,
            output_texture_path=exported_texture_path,
            texture_size=texture_size,
            source_mesh=source_mesh,
        )
        transfer_filter = "custom_per_texel_source_uv_bake"
        stage_durations_sec["bake_texture"] = time.monotonic() - stage_start

        stage_start = time.monotonic()
        copy_with_vertex_translation(
            src_path=exported_target_path,
            dst_path=output_mesh_path,
            delta=None if alignment_translation is None else tuple(-value for value in alignment_translation),
        )
        if exported_texture_path.exists():
            shutil.copyfile(exported_texture_path, output_texture_path)
        if not output_texture_path.exists():
            raise RuntimeError(f"Expected baked texture was not created: {output_texture_path}")
        write_base_color_mtl(output_mtl_path, texture_name=output_texture_path.name)
        rewrite_obj_mtllib(output_mesh_path, mtl_name=output_mtl_path.name)
        stage_durations_sec["canonicalize_outputs"] = time.monotonic() - stage_start

        diagnostics_payload = {
            "ok": True,
            "source_mesh_path": str(source_mesh_path),
            "source_base_color_path": str(source_base_color_path),
            "target_mesh_path": str(target_mesh_path),
            "aligned_target_path": str(aligned_target_path),
            "exported_target_path": str(exported_target_path),
            "output_mesh_path": str(output_mesh_path),
            "output_mtl_path": str(output_mtl_path),
            "output_texture_path": str(output_texture_path),
            "alignment_translation": list(alignment_translation) if alignment_translation is not None else None,
            "source_texture_size": list(texture_size),
            "parameterization_filter": parameterization_filter,
            "transfer_filter": transfer_filter,
            "stage_durations_sec": stage_durations_sec,
            "bake_diagnostics": bake_diagnostics,
        }
        dump_json(diagnostics_payload, diagnostics_json_path)

        return MeshTextureTransferResult(
            ok=True,
            source_mesh_path=source_mesh_path,
            source_base_color_path=source_base_color_path,
            target_mesh_path=target_mesh_path,
            output_mesh_path=output_mesh_path,
            output_mtl_path=output_mtl_path,
            output_texture_path=output_texture_path,
            alignment_translation=alignment_translation,
            source_texture_size=texture_size,
            parameterization_filter=parameterization_filter,
            transfer_filter=transfer_filter,
            diagnostics_dir=diagnostics_dir,
            stage_durations_sec=stage_durations_sec,
        )
    except Exception as exc:  # noqa: BLE001
        diagnostics_payload = {
            "ok": False,
            "source_mesh_path": str(source_mesh_path),
            "source_base_color_path": str(source_base_color_path),
            "target_mesh_path": str(target_mesh_path),
            "output_mesh_path": str(output_mesh_path),
            "output_mtl_path": str(output_mtl_path),
            "output_texture_path": str(output_texture_path),
            "alignment_translation": list(alignment_translation) if alignment_translation is not None else None,
            "parameterization_filter": parameterization_filter,
            "transfer_filter": transfer_filter,
            "stage_durations_sec": stage_durations_sec,
            "error": f"{type(exc).__name__}: {exc}",
        }
        dump_json(diagnostics_payload, diagnostics_json_path)
        return MeshTextureTransferResult(
            ok=False,
            source_mesh_path=source_mesh_path,
            source_base_color_path=source_base_color_path,
            target_mesh_path=target_mesh_path,
            output_mesh_path=None,
            output_mtl_path=None,
            output_texture_path=None,
            alignment_translation=alignment_translation,
            source_texture_size=None,
            parameterization_filter=parameterization_filter,
            transfer_filter=transfer_filter,
            diagnostics_dir=diagnostics_dir,
            stage_durations_sec=stage_durations_sec,
            error=f"{type(exc).__name__}: {exc}",
        )
