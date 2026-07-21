from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import trimesh

from code_agent.configs import CONFIGS
from code_agent.io_utils import dump_json

from .repair.sanity import run_mesh_manifold_check
from .texture.obj_io import parse_obj_with_uv
from .texture.transfer import transfer_texture_to_repaired_mesh
from .validation import (
    run_genesis_cloth_import_validation,
    run_genesis_fem_import_validation,
    run_genesis_rigid_import_validation,
)


@dataclass(slots=True, frozen=True)
class IsotropicRemeshConfig:
    """Configuration for the standalone, pre-simulation remesh tool."""

    input_mesh_path: Path
    output_dir: Path
    target_face_count: int | None = None
    target_edge_length: float | None = None
    target_face_tolerance: float = 0.50
    max_search_attempts: int = 6
    iterations: int = 10
    source_textured_mesh_path: Path | None = None
    source_base_color_path: Path | None = None
    alignment_translation: tuple[float, float, float] | None = None
    scale: float | tuple[float, float, float] = 1.0
    file_meshes_are_zup: bool = False
    validate_genesis: bool = True
    tet_resolution: int = CONFIGS.deformable.tet_resolution

    def __post_init__(self) -> None:
        target_count = int(self.target_face_count is not None) + int(self.target_edge_length is not None)
        if target_count != 1:
            raise ValueError("Specify exactly one of target_face_count or target_edge_length.")
        if self.target_face_count is not None and self.target_face_count < 4:
            raise ValueError("target_face_count must be at least 4.")
        if self.target_edge_length is not None and self.target_edge_length <= 0.0:
            raise ValueError("target_edge_length must be positive.")
        if not 0.0 < self.target_face_tolerance < 1.0:
            raise ValueError("target_face_tolerance must be between 0 and 1.")
        if self.max_search_attempts < 1:
            raise ValueError("max_search_attempts must be at least 1.")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1.")
        if (self.source_textured_mesh_path is None) != (self.source_base_color_path is None):
            raise ValueError("source_textured_mesh_path and source_base_color_path must be supplied together.")
        if self.tet_resolution < 1:
            raise ValueError("tet_resolution must be at least 1.")


def remesh_mesh_asset(config: IsotropicRemeshConfig) -> dict[str, Any]:
    """Isotropically downsample a mesh, transfer base color, and validate the result.

    This function is intentionally standalone. It does not mutate an asset manifest or
    register the output with the training pipeline.
    """

    started_at = time.monotonic()
    input_path = config.input_mesh_path.resolve()
    output_dir = config.output_dir.resolve()
    processed_dir = output_dir / "processed"
    output_mesh_path = processed_dir / "repaired.obj"
    report_path = output_dir / "remesh_report.json"
    processed_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_outputs(processed_dir, report_path)

    source_mesh = _load_triangle_mesh(input_path, skip_texture=True)
    source_stats = _mesh_stats(source_mesh)
    if config.target_face_count is not None and config.target_face_count >= source_stats["face_count"]:
        raise ValueError(
            f"target_face_count ({config.target_face_count}) must be below the source face count "
            f"({source_stats['face_count']}) for this downsampling tool."
        )

    remeshed, attempts = _find_remesh(source_mesh, config)
    if len(remeshed.faces) >= len(source_mesh.faces):
        raise RuntimeError(
            f"Isotropic remesh did not downsample the mesh: {len(source_mesh.faces)} -> {len(remeshed.faces)} faces."
        )
    if source_mesh.is_watertight and float(source_mesh.volume) * float(remeshed.volume) < 0.0:
        remeshed.invert()
    remeshed.export(output_mesh_path)

    output_stats = _mesh_stats(remeshed)
    target_check = _target_check(config, output_stats["face_count"])
    manifold = run_mesh_manifold_check(output_mesh_path)
    report = {
        "standalone": True,
        "pipeline_integrated": False,
        "method": "pymeshlab.meshing_isotropic_explicit_remeshing",
        "input_mesh_path": str(input_path),
        "output_dir": str(output_dir),
        "source": source_stats,
        "request": {
            "target_face_count": config.target_face_count,
            "target_edge_length": config.target_edge_length,
            "target_face_tolerance": config.target_face_tolerance,
            "iterations": config.iterations,
        },
        "search_attempts": attempts,
        "target_check": target_check,
        "output": output_stats,
        "manifold_validation": manifold.to_dict(),
    }

    if not target_check["ok"] or not manifold.ok:
        failure_stage = "target_check" if not target_check["ok"] else "manifold_validation"
        reason = f"Skipped after {failure_stage} failure."
        return _write_report(
            {
                **report,
                "ok": False,
                "failure_stage": failure_stage,
                "recommended_action": "Return this failure to Planner and let Planner select new remesh parameters.",
                "texture_transfer": None,
                "texture_validation": _skipped_validation(config.source_textured_mesh_path is not None, reason),
                "genesis_fem_import_validation": _skipped_validation(config.validate_genesis, reason),
                "artifacts": _artifacts(output_mesh_path, None, None, report_path),
            },
            report_path,
            started_at,
        )

    texture_transfer = None
    texture_validation: dict[str, Any] = {"requested": False, "ok": True}
    output_visual_path: Path | None = None
    output_texture_path: Path | None = None
    if config.source_textured_mesh_path is not None and config.source_base_color_path is not None:
        texture_transfer_result = transfer_texture_to_repaired_mesh(
            source_mesh_path=config.source_textured_mesh_path.resolve(),
            source_base_color_path=config.source_base_color_path.resolve(),
            target_mesh_path=output_mesh_path,
            output_dir=output_dir,
            alignment_translation=config.alignment_translation,
        )
        texture_transfer = texture_transfer_result.to_dict()
        output_visual_path = texture_transfer_result.output_mesh_path
        output_texture_path = texture_transfer_result.output_texture_path
        texture_validation = _validate_texture_outputs(
            transfer_ok=texture_transfer_result.ok,
            runtime_mesh_path=output_mesh_path,
            visual_mesh_path=output_visual_path,
            texture_path=output_texture_path,
            expected_face_count=output_stats["face_count"],
        )

    genesis_validation = _validate_genesis_imports(
        config,
        runtime_path=output_mesh_path,
        visual_path=output_visual_path,
        texture_requested=bool(texture_validation["requested"]),
    )
    return _write_report(
        {
            **report,
            "ok": bool(texture_validation["ok"] and genesis_validation["ok"]),
            "texture_transfer": texture_transfer,
            "texture_validation": texture_validation,
            "genesis_fem_import_validation": genesis_validation,
            "artifacts": _artifacts(output_mesh_path, output_visual_path, output_texture_path, report_path),
        },
        report_path,
        started_at,
    )


def _skipped_validation(requested: bool, reason: str) -> dict[str, Any]:
    return {"requested": requested, "ok": False, "skipped": True, "reason": reason}


def _artifacts(runtime: Path, visual: Path | None, texture: Path | None, report: Path) -> dict[str, str | None]:
    return {
        "runtime_mesh": str(runtime),
        "visual_mesh": None if visual is None else str(visual),
        "base_color_texture": None if texture is None else str(texture),
        "report": str(report),
    }


def _write_report(report: dict[str, Any], path: Path, started_at: float) -> dict[str, Any]:
    report["duration_sec"] = time.monotonic() - started_at
    dump_json(report, path)
    return report


def _validate_genesis_imports(
    config: IsotropicRemeshConfig,
    *,
    runtime_path: Path,
    visual_path: Path | None,
    texture_requested: bool,
) -> dict[str, Any]:
    if not config.validate_genesis:
        skipped = {"requested": False, "ok": True}
        return {
            "requested": False,
            "ok": True,
            "texture_discovery_ok": True,
            "rigid_import": skipped,
            "volumetric_fem_import": skipped,
            "cloth_import": skipped,
        }

    entry = {
        "runtime_path": str(runtime_path),
        "visual_path": None if visual_path is None else str(visual_path),
        "scale": config.scale,
        "file_meshes_are_zup": config.file_meshes_are_zup,
    }
    rigid = run_genesis_rigid_import_validation(entry)
    fem = run_genesis_fem_import_validation(entry, tet_resolution=config.tet_resolution)
    cloth = run_genesis_cloth_import_validation(entry)
    texture_ok = not texture_requested or bool(
        rigid.ok
        and rigid.texture_attached
        and fem.ok
        and fem.surface_visual_uv_shape
        and fem.surface_visual_uv_shape[0] > 0
        and fem.texture_path
        and fem.texture_path.is_file()
        and fem.seam_mapping_ok is True
        and cloth.surface_visual_uv_shape
        and cloth.texture_path
        and cloth.texture_path.is_file()
        and cloth.seam_mapping_ok is True
    )
    return {
        "requested": True,
        "ok": bool(rigid.ok and fem.ok and cloth.ok and texture_ok),
        "texture_discovery_ok": texture_ok,
        "rigid_import": rigid.to_dict(),
        "volumetric_fem_import": fem.to_dict(),
        "cloth_import": cloth.to_dict(),
    }


def _find_remesh(
    source_mesh: trimesh.Trimesh,
    config: IsotropicRemeshConfig,
) -> tuple[trimesh.Trimesh, list[dict[str, Any]]]:
    if config.target_edge_length is not None:
        result = _remesh_once(source_mesh, config.target_edge_length, iterations=config.iterations)
        return result, [_attempt_record(1, config.target_edge_length, result, config.target_face_count)]

    assert config.target_face_count is not None
    target_faces = config.target_face_count
    edge_length = math.sqrt(4.0 * float(source_mesh.area) / (math.sqrt(3.0) * target_faces))
    attempts: list[dict[str, Any]] = []
    best_mesh: trimesh.Trimesh | None = None
    best_error = math.inf

    for attempt_index in range(1, config.max_search_attempts + 1):
        candidate = _remesh_once(source_mesh, edge_length, iterations=config.iterations)
        face_count = len(candidate.faces)
        relative_error = abs(face_count - target_faces) / target_faces
        attempts.append(_attempt_record(attempt_index, edge_length, candidate, target_faces))
        if not candidate.is_watertight or not candidate.is_winding_consistent:
            return candidate, attempts
        if relative_error < best_error:
            best_mesh = candidate
            best_error = relative_error
        if relative_error <= config.target_face_tolerance:
            break

        correction = math.sqrt(max(face_count, 1) / target_faces)
        edge_length *= float(np.clip(correction, 0.70, 1.50))

    assert best_mesh is not None
    return best_mesh, attempts


def _clear_stale_outputs(processed_dir: Path, report_path: Path) -> None:
    for path in (
        processed_dir / "repaired.obj",
        processed_dir / "repaired_textured.obj",
        processed_dir / "repaired_textured.mtl",
        processed_dir / "base_color.png",
        report_path,
    ):
        if path.exists():
            path.unlink()


def _remesh_once(mesh: trimesh.Trimesh, edge_length: float, *, iterations: int) -> trimesh.Trimesh:
    import pymeshlab

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        )
    )
    mesh_set.meshing_isotropic_explicit_remeshing(
        iterations=iterations,
        adaptive=False,
        targetlen=pymeshlab.PureValue(float(edge_length)),
        featuredeg=30.0,
        checksurfdist=True,
        maxsurfdist=pymeshlab.PureValue(float(edge_length) * 0.5),
        splitflag=True,
        collapseflag=True,
        swapflag=True,
        smoothflag=True,
        reprojectflag=True,
    )
    result = mesh_set.current_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(result.vertex_matrix(), dtype=np.float64),
        faces=np.asarray(result.face_matrix(), dtype=np.int64),
        process=False,
    )


def _attempt_record(
    attempt_index: int,
    edge_length: float,
    mesh: trimesh.Trimesh,
    target_face_count: int | None,
) -> dict[str, Any]:
    face_count = len(mesh.faces)
    return {
        "attempt": attempt_index,
        "target_edge_length": float(edge_length),
        "vertex_count": len(mesh.vertices),
        "face_count": face_count,
        "target_face_relative_error": (
            None if target_face_count is None else abs(face_count - target_face_count) / target_face_count
        ),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
    }


def _target_check(config: IsotropicRemeshConfig, face_count: int) -> dict[str, Any]:
    if config.target_face_count is None:
        return {
            "ok": True,
            "mode": "target_edge_length",
            "requested_edge_length": config.target_edge_length,
            "achieved_face_count": face_count,
        }
    relative_error = abs(face_count - config.target_face_count) / config.target_face_count
    return {
        "ok": relative_error <= config.target_face_tolerance,
        "mode": "target_face_count",
        "requested_face_count": config.target_face_count,
        "achieved_face_count": face_count,
        "relative_error": relative_error,
        "tolerance": config.target_face_tolerance,
    }


def _load_triangle_mesh(path: Path, *, skip_texture: bool) -> trimesh.Trimesh:
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = trimesh.load_mesh(str(path), force="mesh", process=False, skip_texture=skip_texture)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh at {path}, got {type(mesh).__name__}.")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0 or mesh.faces.shape[1] != 3:
        raise ValueError(f"Expected a non-empty triangle mesh at {path}.")
    if not np.all(np.isfinite(mesh.vertices)):
        raise ValueError(f"Mesh contains non-finite vertices: {path}")
    return mesh


def _mesh_stats(mesh: trimesh.Trimesh) -> dict[str, Any]:
    edge_lengths = np.asarray(mesh.edges_unique_length, dtype=np.float64)
    return {
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "area": float(mesh.area),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "component_count": len(mesh.split(only_watertight=False)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "bbox_min": np.asarray(mesh.bounds[0], dtype=np.float64).tolist(),
        "bbox_max": np.asarray(mesh.bounds[1], dtype=np.float64).tolist(),
        "bbox_diagonal": float(np.linalg.norm(mesh.extents)),
        "edge_length_min": float(np.min(edge_lengths)),
        "edge_length_median": float(np.median(edge_lengths)),
        "edge_length_max": float(np.max(edge_lengths)),
    }


def _validate_texture_outputs(
    *,
    transfer_ok: bool,
    runtime_mesh_path: Path,
    visual_mesh_path: Path | None,
    texture_path: Path | None,
    expected_face_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested": True,
        "ok": False,
        "visual_mesh_path": None if visual_mesh_path is None else str(visual_mesh_path),
        "texture_path": None if texture_path is None else str(texture_path),
    }
    if not transfer_ok or visual_mesh_path is None or texture_path is None:
        result["error"] = "Texture transfer did not produce all required outputs."
        return result
    try:
        obj = parse_obj_with_uv(visual_mesh_path)
        runtime_mesh = _load_triangle_mesh(runtime_mesh_path, skip_texture=True)
        seam_mapping = _validate_visual_vertex_mapping(runtime_mesh, obj.vertices)
        with Image.open(texture_path) as image:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
            texture_size = list(image.size)
        rgb_std = np.std(rgba[..., :3], axis=(0, 1))
        result.update(
            {
                "ok": bool(
                    len(obj.face_vertex_indices) == expected_face_count
                    and len(obj.texcoords) > 0
                    and float(np.max(rgb_std)) > 1.0
                    and seam_mapping["ok"]
                ),
                "runtime_vertex_count": len(runtime_mesh.vertices),
                "visual_vertex_count": len(obj.vertices),
                "uv_count": len(obj.texcoords),
                "uv_face_count": len(obj.face_vertex_indices),
                "expected_face_count": expected_face_count,
                "texture_size": texture_size,
                "rgb_channel_std": rgb_std.tolist(),
                "texture_non_uniform": bool(float(np.max(rgb_std)) > 1.0),
                "seam_mapping": seam_mapping,
            }
        )
        if not result["ok"]:
            result["error"] = "Transferred texture failed UV, face-count, or non-uniform-image validation."
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _validate_visual_vertex_mapping(
    runtime_mesh: trimesh.Trimesh,
    visual_vertices: np.ndarray,
) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    runtime_vertices = np.asarray(runtime_mesh.vertices, dtype=np.float64)
    visual_vertices = np.asarray(visual_vertices, dtype=np.float64)
    tolerance = max(1e-8, 1e-6 * float(np.linalg.norm(runtime_mesh.extents)))
    distances, source_indices = cKDTree(runtime_vertices).query(visual_vertices, k=1)
    source_indices = np.asarray(source_indices, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)
    unique_source_count = len(np.unique(source_indices))
    duplicate_visual_count = len(visual_vertices) - unique_source_count
    return {
        "ok": bool(len(visual_vertices) > 0 and np.all(distances <= tolerance)),
        "many_to_one": duplicate_visual_count > 0,
        "unique_runtime_source_vertex_count": unique_source_count,
        "duplicate_visual_vertex_count": duplicate_visual_count,
        "max_correspondence_distance": float(np.max(distances)),
        "tolerance": tolerance,
    }


def _parse_scale(values: list[float]) -> float | tuple[float, float, float]:
    if len(values) == 1:
        return values[0]
    if len(values) == 3:
        return values[0], values[1], values[2]
    raise argparse.ArgumentTypeError("--scale expects either one value or three values.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone PyMeshLab isotropic mesh downsampler with texture transfer and Genesis validation."
    )
    parser.add_argument("--input-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-face-count", type=int)
    target.add_argument("--target-edge-length", type=float)
    parser.add_argument("--target-face-tolerance", type=float, default=0.50)
    parser.add_argument("--max-search-attempts", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--source-textured-mesh", type=Path)
    parser.add_argument("--source-base-color", type=Path)
    parser.add_argument("--alignment-translation", type=float, nargs=3)
    parser.add_argument("--scale", type=float, nargs="+", default=[1.0])
    parser.add_argument("--file-meshes-are-zup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tet-resolution", type=int, default=CONFIGS.deformable.tet_resolution)
    parser.add_argument("--skip-genesis-validation", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        scale = _parse_scale(args.scale)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    config = IsotropicRemeshConfig(
        input_mesh_path=args.input_mesh,
        output_dir=args.output_dir,
        target_face_count=args.target_face_count,
        target_edge_length=args.target_edge_length,
        target_face_tolerance=args.target_face_tolerance,
        max_search_attempts=args.max_search_attempts,
        iterations=args.iterations,
        source_textured_mesh_path=args.source_textured_mesh,
        source_base_color_path=args.source_base_color,
        alignment_translation=(None if args.alignment_translation is None else tuple(args.alignment_translation)),
        scale=scale,
        file_meshes_are_zup=args.file_meshes_are_zup,
        validate_genesis=not args.skip_genesis_validation,
        tet_resolution=args.tet_resolution,
    )
    report = remesh_mesh_asset(config)
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "source_faces": report["source"]["face_count"],
                "output_faces": report["output"]["face_count"],
                "runtime_mesh": report["artifacts"]["runtime_mesh"],
                "visual_mesh": report["artifacts"]["visual_mesh"],
                "report": report["artifacts"]["report"],
            },
            indent=2,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
