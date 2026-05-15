from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from trimesh import repair as trimesh_repair

from ..models import MeshRepairConfig, MeshRepairResult
from .components import connected_face_component_count, strip_texture_visuals


def repair_mesh_with_ftetwild(
    mesh_path: Path,
    output_dir: Path,
    config: MeshRepairConfig,
    *,
    attempt_index: int = 1,
    strategy_name: str = "ftetwild",
    output_mesh_path: Path | None = None,
) -> MeshRepairResult:
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    if output_mesh_path is None:
        output_mesh_path = processed_dir / f"repaired{mesh_path.suffix.lower()}"
    return _repair_mesh_with_ftetwild_attempt(
        mesh_path,
        config,
        output_mesh_path=output_mesh_path,
        attempt_index=attempt_index,
        strategy_name=strategy_name,
    )


def _repair_mesh_with_ftetwild_attempt(
    mesh_path: Path,
    config: MeshRepairConfig,
    *,
    output_mesh_path: Path,
    attempt_index: int,
    strategy_name: str,
) -> MeshRepairResult:

    mesh = _load_mesh(mesh_path)
    before_vertices = len(mesh.vertices)
    before_faces = len(mesh.faces)
    before_components = _component_count(mesh, face_cap=config.component_count_face_cap)
    operations: list[str] = []
    stage_durations: dict[str, float] = {}

    try:
        pytetwild = _import_pytetwild()

        stage_start = _now()
        mesh.remove_infinite_values()
        operations.append("remove_infinite_values")
        stage_durations["remove_infinite_values"] = _now() - stage_start

        if config.merge_vertices:
            stage_start = _now()
            mesh.merge_vertices(digits_vertex=config.merge_digits_vertex)
            operations.append("pre_repair_merge_vertices")
            stage_durations["pre_repair_merge_vertices"] = _now() - stage_start
        else:
            stage_durations["pre_repair_merge_vertices"] = 0.0

        if config.process_validate:
            stage_start = _now()
            mesh.process(validate=True)
            operations.append("process_validate")
            stage_durations["pre_repair_validate"] = _now() - stage_start
        else:
            stage_durations["pre_repair_validate"] = 0.0

        stage_start = _now()
        tet_vertices, tet_elems = pytetwild.tetrahedralize(
            mesh.vertices.astype("float64", copy=False),
            mesh.faces.astype("int32", copy=False),
            edge_length_fac=config.ftetwild_edge_length_fac,
            edge_length_abs=config.ftetwild_edge_length_abs,
            optimize=config.ftetwild_optimize,
            simplify=config.ftetwild_simplify,
            epsilon=config.ftetwild_epsilon,
            stop_energy=config.ftetwild_stop_energy,
            coarsen=config.ftetwild_coarsen,
            num_threads=config.ftetwild_num_threads,
            num_opt_iter=config.ftetwild_num_opt_iter,
            quiet=config.ftetwild_quiet,
            disable_filtering=config.ftetwild_disable_filtering,
        )
        operations.append("ftetwild_tetrahedralize")
        stage_durations["ftetwild_tetrahedralize"] = _now() - stage_start

        stage_start = _now()
        surface_faces = _extract_boundary_faces(tet_elems)
        if surface_faces is None or len(surface_faces) == 0:
            raise RuntimeError("fTetWild tetrahedralization produced no boundary facets.")
        repaired_mesh = trimesh.Trimesh(
            vertices=tet_vertices,
            faces=surface_faces,
            process=False,
        )
        operations.append("extract_boundary_surface")
        stage_durations["extract_boundary_surface"] = _now() - stage_start

        if config.min_component_faces > 1:
            stage_start = _now()
            components = repaired_mesh.split(only_watertight=False)
            kept = [component for component in components if len(component.faces) >= config.min_component_faces]
            if kept and len(kept) != len(components):
                repaired_mesh = trimesh.util.concatenate(kept)
                operations.append("drop_small_components")
            stage_durations["drop_small_components"] = _now() - stage_start
        else:
            stage_durations["drop_small_components"] = 0.0

        if config.keep_largest_component:
            stage_start = _now()
            components = repaired_mesh.split(only_watertight=False)
            if len(components) > 1:
                repaired_mesh = max(components, key=lambda item: len(item.faces))
                operations.append("keep_largest_component")
            stage_durations["keep_largest_component"] = _now() - stage_start
        else:
            stage_durations["keep_largest_component"] = 0.0

        if config.process_validate:
            stage_start = _now()
            repaired_mesh.process(validate=True)
            operations.append("post_validate")
            stage_durations["post_validate"] = _now() - stage_start
        else:
            stage_durations["post_validate"] = 0.0

        if config.fix_normals:
            stage_start = _now()
            trimesh_repair.fix_normals(repaired_mesh, multibody=True)
            operations.append("trimesh_fix_normals_repair")
            stage_durations["trimesh_fix_normals_repair"] = _now() - stage_start
        else:
            stage_durations["trimesh_fix_normals_repair"] = 0.0

        stage_start = _now()
        centroid_before_translation, bbox_min, bbox_max, bbox_size = _center_mesh_at_centroid(repaired_mesh)
        operations.append("center_centroid_at_origin")
        stage_durations["center_centroid_at_origin"] = _now() - stage_start

        stage_start = _now()
        repaired_mesh.export(str(output_mesh_path))
        operations.append("export_repaired_mesh")
        stage_durations["export_repaired_mesh"] = _now() - stage_start

        stage_start = _now()
        if _repair_non_manifold_artifacts(output_mesh_path):
            operations.append("repair_non_manifold_edges_and_close_holes")
        stage_durations["repair_non_manifold_edges_and_close_holes"] = _now() - stage_start

        stage_start = _now()
        repaired_mesh = _load_mesh(output_mesh_path)
        if repaired_mesh.is_watertight and repaired_mesh.volume < 0:
            repaired_mesh.invert()
            repaired_mesh.export(str(output_mesh_path))
            repaired_mesh = _load_mesh(output_mesh_path)
            operations.append("orient_positive_volume")
        stage_durations["orient_positive_volume"] = _now() - stage_start

        after_vertices = len(repaired_mesh.vertices)
        after_faces = len(repaired_mesh.faces)
        after_components = _component_count(repaired_mesh, face_cap=config.component_count_face_cap)
        return MeshRepairResult(
            ok=True,
            input_mesh_path=mesh_path,
            output_mesh_path=output_mesh_path,
            attempt_index=attempt_index,
            strategy_name=strategy_name,
            operations=tuple(operations),
            vertex_count_before=before_vertices,
            face_count_before=before_faces,
            component_count_before=before_components,
            vertex_count_after=after_vertices,
            face_count_after=after_faces,
            component_count_after=after_components,
            centroid_before_translation=centroid_before_translation,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            bbox_size=bbox_size,
            centroid_at_origin=True,
            config_snapshot=config.to_dict(),
            stage_durations_sec=stage_durations,
        )
    except Exception as exc:
        return MeshRepairResult(
            ok=False,
            input_mesh_path=mesh_path,
            output_mesh_path=output_mesh_path,
            attempt_index=attempt_index,
            strategy_name=strategy_name,
            operations=tuple(operations),
            vertex_count_before=before_vertices,
            face_count_before=before_faces,
            component_count_before=before_components,
            vertex_count_after=0,
            face_count_after=0,
            component_count_after=0,
            centroid_before_translation=None,
            bbox_min=None,
            bbox_max=None,
            bbox_size=None,
            centroid_at_origin=False,
            config_snapshot=config.to_dict(),
            stage_durations_sec=stage_durations,
            error=f"{type(exc).__name__}: {exc}",
        )


def _import_pytetwild():
    try:
        import pytetwild  # type: ignore

        return pytetwild
    except Exception as exc:
        raise RuntimeError(
            "pytetwild is not installed in the current environment. Install it in your container first "
            "(for example: `uv pip install pytetwild`)."
        ) from exc


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(mesh_path), force="mesh", skip_texture=True, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh).__name__}")
    return strip_texture_visuals(mesh)


def _component_count(mesh: trimesh.Trimesh, face_cap: int) -> int:
    return connected_face_component_count(mesh, face_cap=face_cap)


def _repair_non_manifold_artifacts(mesh_path: Path) -> bool:
    try:
        import pymeshlab  # type: ignore
    except Exception:
        return False

    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(mesh_path))
    for filter_name, kwargs in (
        ("meshing_remove_duplicate_faces", {}),
        ("meshing_remove_duplicate_vertices", {}),
        ("meshing_remove_null_faces", {}),
        ("meshing_remove_folded_faces", {}),
        ("meshing_repair_non_manifold_edges", {}),
        ("meshing_repair_non_manifold_vertices", {}),
        ("meshing_close_holes", {"maxholesize": 100}),
        ("meshing_remove_unreferenced_vertices", {}),
    ):
        getattr(mesh_set, filter_name)(**kwargs)
    mesh_set.save_current_mesh(str(mesh_path))
    return True


def _extract_boundary_faces(tet_elems: np.ndarray) -> np.ndarray:
    tets = np.asarray(tet_elems, dtype=np.int32)
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"Expected tetrahedral connectivity with shape (n, 4), got {tets.shape}.")

    # Consistent local orientation for tetra boundary faces.
    faces = np.concatenate(
        [
            tets[:, [0, 2, 1]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 3, 2]],
            tets[:, [1, 2, 3]],
        ],
        axis=0,
    )
    sorted_faces = np.sort(faces, axis=1)
    _, _first_idx, inverse, counts = np.unique(
        sorted_faces,
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    boundary_mask = counts[inverse] == 1
    boundary_faces = faces[boundary_mask]
    if len(boundary_faces) == 0:
        return boundary_faces

    # Remove exact duplicates while preserving first occurrence ordering.
    boundary_sorted = np.sort(boundary_faces, axis=1)
    _, keep_idx = np.unique(boundary_sorted, axis=0, return_index=True)
    keep_idx.sort()
    return boundary_faces[keep_idx]


def _center_mesh_at_centroid(
    mesh: trimesh.Trimesh,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    centroid = mesh.centroid.astype(float)
    mesh.vertices -= centroid
    bounds = mesh.bounds.astype(float)
    bbox_min = tuple(float(value) for value in bounds[0])
    bbox_max = tuple(float(value) for value in bounds[1])
    bbox_size = tuple(float(value) for value in (bounds[1] - bounds[0]))
    return tuple(float(value) for value in centroid), bbox_min, bbox_max, bbox_size


def _now() -> float:
    import time

    return time.monotonic()
