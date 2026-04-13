import os
import pickle as pkl
import math

import numpy as np
import trimesh
import igl

import genesis as gs

from . import mesh as mu


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
    mesh = mu.load_mesh(file)

    mesh.vertices = mesh.vertices * scale

    resolution = _tet_resolution(tet_cfg)
    feature_sizes = tuple(np.maximum(mesh.bounding_box.extents.astype(np.float64, copy=False), 1e-6).tolist())
    base_target_edge = _primitive_target_edge(feature_sizes=feature_sizes, resolution=resolution)

    # Retry isotropic remeshing with progressively less aggressive edge lengths until
    # the remeshed surface stays manifold-ready and TetGen accepts it.
    retry_factors = (1.0, 0.9, 0.8, 0.7, 0.62, 0.55, 0.48, 0.4, 0.32, 0.25)
    remesh_error: str | None = None
    for attempt_index, factor in enumerate(retry_factors, start=1):
        target_edge = base_target_edge * factor
        remeshed = mu.remesh_surface_mesh(mesh, edge_len_abs=target_edge, fix=False)
        candidate_cfg = _primitive_tet_cfg(tet_cfg, target_edge=target_edge)

        if not (remeshed.is_watertight and remeshed.is_winding_consistent):
            remesh_error = (
                f"attempt {attempt_index}: remeshed surface is not watertight/winding-consistent "
                f"(edge_len_abs={target_edge})"
            )
            continue

        try:
            # Probe TetGen acceptance on the remeshed surface before committing to this mesh.
            mu.tetrahedralize_mesh(remeshed, candidate_cfg)
        except Exception as exc:  # noqa: BLE001
            remesh_error = (
                f"attempt {attempt_index}: remeshed surface failed tetgen check with edge_len_abs={target_edge}: {exc}"
            )
            continue

        mesh = remeshed
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
            verts, elems = mu.tetrahedralize_mesh(mesh, tet_cfg)

            os.makedirs(os.path.dirname(tet_file_path), exist_ok=True)
            with open(tet_file_path, "wb") as tet_file:
                pkl.dump((verts, elems), tet_file)

    verts += np.array(pos)

    # Surface remeshing changes topology, so original UV correspondence is no longer reliable.
    return verts, elems, None


def split_all_surface_tets(verts, elems):
    """
    Splits tetrahedras that have 4 vertices on the surface into 4 smaller tetrahedras.

    This is useful for the hydroelastic contact model.
    """
    F, *_ = igl.boundary_facets(elems)
    on_surface = np.zeros(verts.shape[0], dtype=bool)
    on_surface[F.reshape(-1)] = True
    all_on_surface = np.all(on_surface[elems], axis=1)
    if not all_on_surface.any():
        return verts, elems
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
    # remove the bad elements from the original elements
    elems = np.concatenate([elems[~all_on_surface], new_elems], axis=0)
    return verts, elems


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
