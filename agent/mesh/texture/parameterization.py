from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import xatlas

from .obj_io import write_parameterized_obj


def parameterize_target_mesh_xatlas(
    *,
    target_mesh: trimesh.Trimesh,
    output_mesh_path: Path,
    texture_size: tuple[int, int],
) -> str:
    vertices = np.asarray(target_mesh.vertices, dtype=np.float32)
    faces = np.asarray(target_mesh.faces, dtype=np.uint32)
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.resolution = max(int(texture_size[0]), int(texture_size[1]), 256)
    pack_options.padding = 2
    pack_options.bilinear = True
    atlas.generate(chart_options, pack_options)
    vmapping, indices, uvs = atlas.get_mesh(0)
    out_vertices = vertices[np.asarray(vmapping, dtype=np.int64)]
    out_faces = np.asarray(indices, dtype=np.int64)
    out_uvs = np.asarray(uvs, dtype=np.float64)
    write_parameterized_obj(
        obj_path=output_mesh_path,
        vertices=out_vertices,
        faces=out_faces,
        uvs=out_uvs,
    )
    return "xatlas.parametrize"
