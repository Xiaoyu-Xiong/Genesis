import os
import pickle as pkl
import math
from pathlib import Path

import numpy as np
import trimesh
import igl

import genesis as gs

from . import mesh as mu

_TEXTURED_RENDER_ARTIFACTS: dict[str, dict[str, object]] = {}


def _mesh_to_elements_texture_key(file, scale, tet_cfg: dict) -> str:
    if isinstance(file, (str, os.PathLike)):
        return mu.get_hashkey(Path(file), np.asarray(scale), tet_cfg)
    if hasattr(file, "vertices") and hasattr(file, "faces"):
        return mu.get_hashkey(file.vertices, file.faces, np.asarray(scale), tet_cfg)
    return mu.get_hashkey(str(file), np.asarray(scale), tet_cfg)

def get_mesh_to_elements_render_artifact(file, scale=1.0, tet_cfg=dict()):
    return _TEXTURED_RENDER_ARTIFACTS.get(_mesh_to_elements_texture_key(file, scale, tet_cfg))


def _locate_textured_source_assets(file) -> tuple[Path | None, Path | None]:
    if not isinstance(file, (str, os.PathLike)):
        return None, None
    mesh_path = Path(file)
    if mesh_path.parent.name == "processed":
        asset_root = mesh_path.parent.parent
        textured_mesh = asset_root / "textured" / "model.obj"
        base_color = asset_root / "textured" / "base_color.png"
        if textured_mesh.exists() and base_color.exists():
            return textured_mesh, base_color
    return None, None


def _build_remeshed_textured_source(*, file, remeshed):
    textured_mesh_path, base_color_path = _locate_textured_source_assets(file)
    if textured_mesh_path is None or base_color_path is None:
        return None, None

    cache_stem = Path(
        mu.get_uv_transfer_path(
            "remesh_textured_surface",
            textured_mesh_path,
            base_color_path,
            remeshed.vertices,
            remeshed.faces,
        )
    ).with_suffix("")
    cache_dir = cache_stem
    processed_dir = cache_dir / "processed"
    remeshed_obj = processed_dir / "remeshed_source.obj"
    remeshed_png = processed_dir / "base_color.png"

    if not remeshed_obj.exists() or not remeshed_png.exists():
        from agent.mesh.texture_transfer import transfer_texture_to_repaired_mesh

        processed_dir.mkdir(parents=True, exist_ok=True)
        remeshed.export(remeshed_obj)
        transfer_texture_to_repaired_mesh(
            source_mesh_path=textured_mesh_path,
            source_base_color_path=base_color_path,
            target_mesh_path=remeshed_obj,
            output_dir=cache_dir,
            alignment_translation=None,
        )

    remeshed_textured_mesh = trimesh.load_mesh(
        str(remeshed_obj),
        force="mesh",
        skip_texture=False,
        process=False,
    )
    return remeshed_textured_mesh, remeshed_png


def box_to_elements(pos=(0, 0, 0), size=(1, 1, 1), tet_cfg=dict()):
    resolution = _tet_resolution(tet_cfg)
    target_edge = _primitive_target_edge(feature_sizes=size, resolution=resolution)
    trimesh_obj = trimesh.creation.box(extents=size)
    trimesh_obj = mu.remesh_surface_mesh(trimesh_obj, edge_len_abs=target_edge, fix=False)
    tet_cfg = _primitive_tet_cfg(tet_cfg, target_edge=target_edge)
    trimesh_obj.vertices += np.array(pos)
    verts, elems = mu.tetrahedralize_mesh(trimesh_obj, tet_cfg)

    return verts, elems


def sphere_to_elements(pos=(0, 0, 0), radius=0.5, tet_cfg=dict()):
    resolution = _tet_resolution(tet_cfg)
    target_edge = _primitive_target_edge(feature_sizes=(2.0 * radius,), resolution=resolution)
    trimesh_obj = trimesh.creation.icosphere(subdivisions=max(1, resolution), radius=radius)
    trimesh_obj = mu.remesh_surface_mesh(trimesh_obj, edge_len_abs=target_edge, fix=False)
    tet_cfg = _primitive_tet_cfg(tet_cfg, target_edge=target_edge)
    trimesh_obj.vertices += np.array(pos)
    verts, elems = mu.tetrahedralize_mesh(trimesh_obj, tet_cfg)

    return verts, elems


def cylinder_to_elements(pos=(0, 0, 0), radius=0.5, height=1.0, tet_cfg=dict()):
    resolution = _tet_resolution(tet_cfg)
    target_edge = _primitive_target_edge(feature_sizes=(2.0 * radius, height), resolution=resolution)
    sections = max(24, int(math.ceil((2.0 * math.pi * radius) / max(target_edge, 1e-9))))
    trimesh_obj = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    trimesh_obj = mu.remesh_surface_mesh(trimesh_obj, edge_len_abs=target_edge, fix=False)
    tet_cfg = _primitive_tet_cfg(tet_cfg, target_edge=target_edge)
    trimesh_obj.vertices += np.array(pos)
    verts, elems = mu.tetrahedralize_mesh(trimesh_obj, tet_cfg)

    return verts, elems


def mesh_to_elements(file, pos=(0, 0, 0), scale=1.0, tet_cfg=dict()):
    texture_key = _mesh_to_elements_texture_key(file, scale, tet_cfg)
    mesh = mu.load_mesh(file)

    mesh.vertices = mesh.vertices * scale
    textured_source_mesh = mesh.copy() if mu.mesh_has_texture(mesh) else None

    resolution = _tet_resolution(tet_cfg)
    feature_sizes = tuple(np.maximum(mesh.bounding_box.extents.astype(np.float64, copy=False), 1e-6).tolist())
    base_target_edge = _primitive_target_edge(feature_sizes=feature_sizes, resolution=resolution)
    remeshed_textured_mesh = None
    render_texture_path = None

    # Retry isotropic remeshing with progressively less aggressive edge lengths until
    # the remeshed surface stays manifold-ready and TetGen accepts it.
    retry_factors = (1.0, 0.9, 0.8, 0.7, 0.62, 0.55, 0.48, 0.4, 0.32, 0.25)
    remesh_error: str | None = None
    for attempt_index, factor in enumerate(retry_factors, start=1):
        target_edge = base_target_edge * factor
        remeshed = mu.remesh_surface_mesh(mesh, edge_len_abs=target_edge, fix=False)
        candidate_cfg = _primitive_tet_cfg(tet_cfg, target_edge=target_edge)
        if textured_source_mesh is not None:
            candidate_cfg["nobisect"] = True

        if not (remeshed.is_watertight and remeshed.is_winding_consistent):
            remesh_error = (
                f"attempt {attempt_index}: remeshed surface is not watertight/winding-consistent "
                f"(edge_len_abs={target_edge})"
            )
            continue

        try:
            # Probe TetGen acceptance on the remeshed surface before committing to this mesh.
            if textured_source_mesh is not None:
                mu.tetrahedralize_mesh_with_boundary(remeshed, candidate_cfg)
            else:
                mu.tetrahedralize_mesh(remeshed, candidate_cfg)
        except Exception as exc:  # noqa: BLE001
            remesh_error = (
                f"attempt {attempt_index}: remeshed surface failed tetgen check with edge_len_abs={target_edge}: {exc}"
            )
            continue

        mesh = remeshed
        remeshed_textured_mesh = None
        if textured_source_mesh is not None:
            remeshed_textured_mesh, remeshed_texture_path = _build_remeshed_textured_source(file=file, remeshed=remeshed)
            render_texture_path = None if remeshed_texture_path is None else str(remeshed_texture_path)
        tet_cfg = candidate_cfg
        break
    else:
        gs.raise_exception(
            f"Unable to produce a tetgen-ready remeshed surface from the mesh input after {len(retry_factors)} attempts. "
            f"Last error: {remesh_error}"
        )

    # compute file name via hashing for caching
    tet_file_path = mu.get_tet_path(mesh.vertices, mesh.faces, tet_cfg)

    # loading pre-computed cache if available
    is_cached_loaded = False
    if os.path.exists(tet_file_path):
        gs.logger.debug("Tetrahedra file (`.tet`) found in cache.")
        try:
            with open(tet_file_path, "rb") as tet_file:
                verts, elems = pkl.load(tet_file)
            is_cached_loaded = True
        except (EOFError, ModuleNotFoundError, pkl.UnpicklingError, TypeError, MemoryError):
            gs.logger.info("Ignoring corrupted cache.")

    if not is_cached_loaded:
        with gs.logger.timer(f"Tetrahedralization with configuration {tet_cfg} and generating `.tet` file:"):
            if textured_source_mesh is not None:
                verts, elems, boundary_faces = mu.tetrahedralize_mesh_with_boundary(mesh, tet_cfg)
            else:
                verts, elems = mu.tetrahedralize_mesh(mesh, tet_cfg)
                boundary_faces = None

            os.makedirs(os.path.dirname(tet_file_path), exist_ok=True)
            with open(tet_file_path, "wb") as tet_file:
                pkl.dump((verts, elems), tet_file)
    else:
        boundary_faces = None

    uvs = None
    if textured_source_mesh is not None:
        source_for_boundary = remeshed_textured_mesh if remeshed_textured_mesh is not None else textured_source_mesh
        uvs = _transfer_remeshed_uvs_to_tet_boundary(
            remeshed_textured_mesh=source_for_boundary,
            tet_verts=verts,
            tet_elems=elems,
            tet_boundary_faces=boundary_faces,
        )
        _TEXTURED_RENDER_ARTIFACTS[texture_key] = _build_render_artifact(
            remeshed_textured_mesh=source_for_boundary,
            tet_verts=verts,
            tet_boundary_faces=boundary_faces,
            tet_elems=elems,
            texture_path=render_texture_path,
        )
    else:
        _TEXTURED_RENDER_ARTIFACTS[texture_key] = {}

    verts += np.array(pos)

    return verts, elems, uvs


def _transfer_remeshed_uvs_to_tet_boundary(*, remeshed_textured_mesh, tet_verts, tet_elems, tet_boundary_faces):
    if remeshed_textured_mesh is None or not mu.mesh_has_texture(remeshed_textured_mesh):
        return None

    if tet_boundary_faces is None:
        boundary_faces = igl.boundary_facets(tet_elems)
        if isinstance(boundary_faces, tuple):
            boundary_faces = boundary_faces[0]
        tet_boundary_faces = np.asarray(boundary_faces, dtype=np.int32)
    else:
        tet_boundary_faces = np.asarray(tet_boundary_faces, dtype=np.int32)

    if tet_boundary_faces.size == 0:
        return np.zeros((len(tet_verts), 2), dtype=gs.np_float)

    source_vertices = np.asarray(remeshed_textured_mesh.vertices, dtype=np.float64)
    source_uvs = np.asarray(remeshed_textured_mesh.visual.uv, dtype=gs.np_float)
    boundary_vertex_ids = np.unique(tet_boundary_faces.reshape(-1))
    boundary_positions = np.asarray(tet_verts[boundary_vertex_ids], dtype=np.float64)

    source_index_by_key: dict[tuple[int, int, int], int] = {}
    scale = 1e8
    for idx, vertex in enumerate(source_vertices):
        key = tuple(np.round(vertex * scale).astype(np.int64).tolist())
        source_index_by_key[key] = idx

    mapped_uvs = np.zeros((len(tet_verts), 2), dtype=gs.np_float)
    unmatched = []
    for tet_idx, vertex_id in enumerate(boundary_vertex_ids):
        pos = boundary_positions[tet_idx]
        key = tuple(np.round(pos * scale).astype(np.int64).tolist())
        source_idx = source_index_by_key.get(key)
        if source_idx is None:
            unmatched.append((tet_idx, pos))
            continue
        mapped_uvs[vertex_id] = source_uvs[source_idx]

    if unmatched:
        source_pos = source_vertices
        for tet_idx, pos in unmatched:
            delta = source_pos - pos[None, :]
            dist2 = np.einsum("ij,ij->i", delta, delta)
            source_idx = int(np.argmin(dist2))
            if float(dist2[source_idx]) > 1e-10:
                gs.raise_exception(
                    f"Unable to build stable TetGen boundary correspondence for texture transfer; max mismatch {dist2[source_idx]}"
                )
            mapped_uvs[boundary_vertex_ids[tet_idx]] = source_uvs[source_idx]

    return mapped_uvs


def _build_render_artifact(*, remeshed_textured_mesh, tet_verts, tet_boundary_faces, tet_elems, texture_path):
    if remeshed_textured_mesh is None or not mu.mesh_has_texture(remeshed_textured_mesh):
        return {}

    if tet_boundary_faces is None:
        boundary_faces = igl.boundary_facets(tet_elems)
        if isinstance(boundary_faces, tuple):
            boundary_faces = boundary_faces[0]
        tet_boundary_faces = np.asarray(boundary_faces, dtype=np.int32)
    tet_boundary_faces = np.asarray(tet_boundary_faces, dtype=np.int32)
    if tet_boundary_faces.size == 0:
        return {}

    boundary_vertex_ids = np.unique(tet_boundary_faces.reshape(-1))
    boundary_positions = np.asarray(tet_verts[boundary_vertex_ids], dtype=np.float64)
    boundary_index_by_key: dict[tuple[int, int, int], int] = {}
    scale = 1e8
    for idx, vertex_id in enumerate(boundary_vertex_ids):
        key = tuple(np.round(boundary_positions[idx] * scale).astype(np.int64).tolist())
        boundary_index_by_key[key] = int(vertex_id)

    render_vertices = np.asarray(remeshed_textured_mesh.vertices, dtype=np.float64)
    render_faces = np.asarray(remeshed_textured_mesh.faces, dtype=np.int32)
    render_uvs = np.asarray(remeshed_textured_mesh.visual.uv, dtype=gs.np_float)
    render_src_indices = np.full(len(render_vertices), -1, dtype=np.int32)

    unmatched = []
    for idx, pos in enumerate(render_vertices):
        key = tuple(np.round(pos * scale).astype(np.int64).tolist())
        mapped = boundary_index_by_key.get(key)
        if mapped is None:
            unmatched.append((idx, pos))
            continue
        render_src_indices[idx] = mapped

    if unmatched:
        boundary_positions_full = np.asarray(tet_verts[boundary_vertex_ids], dtype=np.float64)
        for idx, pos in unmatched:
            delta = boundary_positions_full - pos[None, :]
            dist2 = np.einsum("ij,ij->i", delta, delta)
            match_local = int(np.argmin(dist2))
            if float(dist2[match_local]) > 1e-10:
                gs.raise_exception(
                    f"Unable to build seam-aware FEM render correspondence; max mismatch {dist2[match_local]}"
                )
            render_src_indices[idx] = int(boundary_vertex_ids[match_local])

    return {
        "render_vertex_src_indices": render_src_indices.astype(gs.np_int, copy=False),
        "render_faces": render_faces.astype(gs.np_int, copy=False),
        "render_uvs": render_uvs.astype(gs.np_float, copy=False),
        "texture_path": texture_path,
    }


def split_all_surface_tets(verts, elems, uvs=None):
    """
    Splits tetrahedras that have 4 vertices on the surface into 4 smaller tetrahedras.

    This is useful for the hydroelastic contact model.
    """
    F, *_ = igl.boundary_facets(elems)
    on_surface = np.zeros(verts.shape[0], dtype=bool)
    on_surface[F.reshape(-1)] = True
    all_on_surface = np.all(on_surface[elems], axis=1)
    if not all_on_surface.any():
        if uvs is None:
            return verts, elems
        return verts, elems, uvs
    bad_elems = elems[all_on_surface]
    new_verts = np.mean(verts[bad_elems], axis=1, dtype=np.float32)
    new_elems = []
    for idx, (v0, v1, v2, v3) in enumerate(bad_elems, len(verts)):
        new_elems.append([v0, v1, v2, idx])
        new_elems.append([v0, v1, idx, v3])
        new_elems.append([v0, idx, v2, v3])
        new_elems.append([idx, v1, v2, v3])
    new_elems = np.array(new_elems, dtype=np.int32)
    verts = np.concatenate([verts, new_verts], axis=0)
    if uvs is not None:
        zero_uvs = np.zeros((len(new_verts), 2), dtype=uvs.dtype)
        uvs = np.concatenate([uvs, zero_uvs], axis=0)
    # remove the bad elements from the original elements
    elems = np.concatenate([elems[~all_on_surface], new_elems], axis=0)
    if uvs is None:
        return verts, elems
    return verts, elems, uvs


def _tet_resolution(tet_cfg: dict) -> int:
    return max(1, int(tet_cfg.get("tet_resolution", 3)))


def _primitive_target_edge(*, feature_sizes, resolution: int) -> float:
    min_feature = max(float(min(feature_sizes)), 1e-6)
    return min_feature / float(2 * resolution + 1)


def _primitive_tet_cfg(tet_cfg: dict, *, target_edge: float) -> dict:
    cfg = dict(tet_cfg)
    cfg["quality"] = True
    cfg["nobisect"] = False
    cfg["mindihedral"] = max(int(cfg.get("mindihedral", 10)), 15)
    if float(cfg.get("maxvolume", -1.0)) > 0:
        return cfg
    cfg["maxvolume"] = max((target_edge**3) * math.sqrt(2.0) / 12.0, 1e-9)
    return cfg
