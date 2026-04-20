from __future__ import annotations

from dataclasses import dataclass
import shutil
import time
from pathlib import Path
from typing import Any

import igl
import numpy as np
from PIL import Image
import trimesh
import xatlas

from ..configs import CONFIGS
from ..io_utils import dump_json
from .models import MeshTextureTransferResult


@dataclass(slots=True)
class ObjUvMesh:
    vertices: np.ndarray
    texcoords: np.ndarray
    face_vertex_indices: np.ndarray
    face_texcoord_indices: np.ndarray


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
    debug_dir = processed_dir / "repaired_texture_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    output_mesh_path = target_mesh_path
    output_mtl_path = processed_dir / "repaired.mtl"
    output_texture_path = processed_dir / "base_color.png"
    aligned_target_path = debug_dir / "repaired_aligned.obj"
    exported_target_path = debug_dir / "repaired_textured_export.obj"
    exported_texture_path = debug_dir / "baked_base_color.png"
    debug_json_path = debug_dir / "transfer_debug.json"

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

        texture_size = _read_bake_texture_size(source_base_color_path)
        # The repaired mesh is centered before export; shift it back temporarily so the
        # textured raw mesh and repaired geometry live in the same frame for baking.
        _copy_with_vertex_translation(
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
        parameterization_filter = _parameterize_target_mesh_xatlas(
            target_mesh=target_mesh,
            output_mesh_path=exported_target_path,
            texture_size=texture_size,
        )
        stage_durations_sec["parameterize_target_mesh"] = time.monotonic() - stage_start

        stage_start = time.monotonic()
        bake_debug = _bake_texture_from_source_mesh(
            source_base_color_path=source_base_color_path,
            target_parameterized_mesh_path=exported_target_path,
            output_texture_path=exported_texture_path,
            texture_size=texture_size,
            source_mesh=source_mesh,
        )
        transfer_filter = "custom_per_texel_source_uv_bake"
        stage_durations_sec["bake_texture"] = time.monotonic() - stage_start

        stage_start = time.monotonic()
        _copy_with_vertex_translation(
            src_path=exported_target_path,
            dst_path=output_mesh_path,
            delta=None if alignment_translation is None else tuple(-value for value in alignment_translation),
        )
        if exported_texture_path.exists():
            shutil.copyfile(exported_texture_path, output_texture_path)
        if not output_texture_path.exists():
            raise RuntimeError(f"Expected baked texture was not created: {output_texture_path}")
        _write_base_color_mtl(output_mtl_path, texture_name=output_texture_path.name)
        _rewrite_obj_mtllib(output_mesh_path, mtl_name=output_mtl_path.name)
        stage_durations_sec["canonicalize_outputs"] = time.monotonic() - stage_start

        debug_payload = {
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
            "bake_debug": bake_debug,
        }
        dump_json(debug_payload, debug_json_path)

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
            debug_dir=debug_dir,
            stage_durations_sec=stage_durations_sec,
        )
    except Exception as exc:  # noqa: BLE001
        debug_payload = {
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
        dump_json(debug_payload, debug_json_path)
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
            debug_dir=debug_dir,
            stage_durations_sec=stage_durations_sec,
            error=f"{type(exc).__name__}: {exc}",
        )


def _parameterize_target_mesh_xatlas(*, target_mesh: trimesh.Trimesh, output_mesh_path: Path, texture_size: tuple[int, int]) -> str:
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
    _write_parameterized_obj(
        obj_path=output_mesh_path,
        vertices=out_vertices,
        faces=out_faces,
        uvs=out_uvs,
    )
    return "xatlas.parametrize"


def _bake_texture_from_source_mesh(
    *,
    source_base_color_path: Path,
    target_parameterized_mesh_path: Path,
    output_texture_path: Path,
    texture_size: tuple[int, int],
    source_mesh: trimesh.Trimesh,
) -> dict[str, Any]:
    source_uv = getattr(source_mesh.visual, "uv", None)
    if source_uv is None:
        raise RuntimeError("Source textured mesh does not expose UV coordinates through trimesh.")
    source_uv = np.asarray(source_uv, dtype=np.float64)
    if source_uv.ndim != 2 or source_uv.shape[1] != 2:
        raise RuntimeError(f"Unexpected source UV shape: {source_uv.shape}")

    target_obj = _parse_obj_with_uv(target_parameterized_mesh_path)
    source_image = _load_texture_image(source_base_color_path)

    baked = _rasterize_source_uv_to_texture(
        obj_mesh=target_obj,
        source_mesh=source_mesh,
        source_uv=source_uv,
        source_image=source_image,
        texture_size=texture_size,
    )
    output_texture_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(baked["image"], mode="RGBA").save(output_texture_path)

    distances = np.asarray(baked["distances"], dtype=np.float64)
    return {
        "target_vertex_count": int(len(target_obj.vertices)),
        "target_face_count": int(len(target_obj.face_vertex_indices)),
        "source_face_hit_count": int(len(np.unique(np.asarray(baked["face_indices"], dtype=np.int64)))),
        "distance_min": float(np.min(distances)) if len(distances) else 0.0,
        "distance_mean": float(np.mean(distances)) if len(distances) else 0.0,
        "distance_max": float(np.max(distances)) if len(distances) else 0.0,
        "texture_size": [int(texture_size[0]), int(texture_size[1])],
        "bake_mode": "per_texel_source_uv",
        "target_texel_count": int(baked["texel_count"]),
    }


def _parse_obj_with_uv(obj_path: Path) -> ObjUvMesh:
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    face_vertex_indices: list[list[int]] = []
    face_texcoord_indices: list[list[int]] = []

    for raw_line in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            continue
        if line.startswith("vt "):
            parts = line.split()
            if len(parts) >= 3:
                texcoords.append([float(parts[1]), float(parts[2])])
            continue
        if not line.startswith("f "):
            continue

        tokens = line.split()[1:]
        polygon_v: list[int] = []
        polygon_vt: list[int] = []
        for token in tokens:
            chunks = token.split("/")
            if len(chunks) < 2 or not chunks[1]:
                raise RuntimeError(f"Face is missing UV indices in parameterized OBJ: {obj_path}")
            polygon_v.append(_obj_index_to_zero_based(chunks[0], len(vertices)))
            polygon_vt.append(_obj_index_to_zero_based(chunks[1], len(texcoords)))

        for idx in range(1, len(polygon_v) - 1):
            face_vertex_indices.append([polygon_v[0], polygon_v[idx], polygon_v[idx + 1]])
            face_texcoord_indices.append([polygon_vt[0], polygon_vt[idx], polygon_vt[idx + 1]])

    if not vertices or not texcoords or not face_vertex_indices:
        raise RuntimeError(f"Failed to parse parameterized OBJ with UVs: {obj_path}")

    return ObjUvMesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        texcoords=np.asarray(texcoords, dtype=np.float64),
        face_vertex_indices=np.asarray(face_vertex_indices, dtype=np.int64),
        face_texcoord_indices=np.asarray(face_texcoord_indices, dtype=np.int64),
    )


def _obj_index_to_zero_based(token: str, item_count: int) -> int:
    value = int(token)
    if value > 0:
        return value - 1
    if value < 0:
        return item_count + value
    raise RuntimeError("OBJ indices are 1-based; found invalid 0 index.")


def _write_parameterized_obj(*, obj_path: Path, vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray) -> None:
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for vertex in np.asarray(vertices, dtype=np.float64):
        lines.append(f"v {float(vertex[0]):.9f} {float(vertex[1]):.9f} {float(vertex[2]):.9f}")
    for uv in np.asarray(uvs, dtype=np.float64):
        lines.append(f"vt {float(uv[0]):.9f} {float(uv[1]):.9f}")
    for face in np.asarray(faces, dtype=np.int64):
        a, b, c = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_texture_image(texture_path: Path) -> np.ndarray:
    with Image.open(texture_path) as image:
        rgba = image.convert("RGBA")
        return np.asarray(rgba, dtype=np.float32) / 255.0


def _sample_image_bilinear(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 4:
        raise RuntimeError(f"Expected RGBA image array, got shape {image.shape}")

    height, width, _ = image.shape
    uv = np.asarray(uv, dtype=np.float64)
    u = np.clip(uv[:, 0], 0.0, 1.0)
    v = np.clip(uv[:, 1], 0.0, 1.0)

    x = u * (width - 1)
    y = (1.0 - v) * (height - 1)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    c00 = image[y0, x0]
    c10 = image[y0, x1]
    c01 = image[y1, x0]
    c11 = image[y1, x1]

    top = c00 * (1.0 - wx[:, None]) + c10 * wx[:, None]
    bottom = c01 * (1.0 - wx[:, None]) + c11 * wx[:, None]
    return top * (1.0 - wy[:, None]) + bottom * wy[:, None]


def _rasterize_source_uv_to_texture(
    *,
    obj_mesh: ObjUvMesh,
    source_mesh: trimesh.Trimesh,
    source_uv: np.ndarray,
    source_image: np.ndarray,
    texture_size: tuple[int, int],
) -> dict[str, Any]:
    width = max(int(texture_size[0]), 1)
    height = max(int(texture_size[1]), 1)
    texture = np.zeros((height, width, 4), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)
    sample_points: list[np.ndarray] = []
    sample_pixels: list[np.ndarray] = []

    for face_vertices, face_texcoords in zip(obj_mesh.face_vertex_indices, obj_mesh.face_texcoord_indices, strict=False):
        target_uv = obj_mesh.texcoords[face_texcoords]
        target_vertices = obj_mesh.vertices[face_vertices]

        px = target_uv[:, 0] * (width - 1)
        py = (1.0 - target_uv[:, 1]) * (height - 1)
        tri = np.stack([px, py], axis=1)

        min_x = max(int(np.floor(np.min(px))), 0)
        max_x = min(int(np.ceil(np.max(px))), width - 1)
        min_y = max(int(np.floor(np.min(py))), 0)
        max_y = min(int(np.ceil(np.max(py))), height - 1)
        if min_x > max_x or min_y > max_y:
            continue

        xs, ys = np.meshgrid(
            np.arange(min_x, max_x + 1, dtype=np.float32),
            np.arange(min_y, max_y + 1, dtype=np.float32),
            indexing="xy",
        )
        points = np.stack([xs + 0.5, ys + 0.5], axis=-1)
        bary = _barycentric_2d(points, tri)
        mask = np.all(bary >= -1e-6, axis=-1)
        if not np.any(mask):
            continue

        masked_bary = bary[mask]
        target_points = np.einsum("ni,ij->nj", masked_bary, target_vertices)
        yy = ys.astype(np.int64)[mask]
        xx = xs.astype(np.int64)[mask]
        sample_points.append(target_points.astype(np.float64, copy=False))
        sample_pixels.append(np.stack([yy, xx], axis=1))

    if not sample_points:
        raise RuntimeError("Target UV parameterization produced no covered texels for baking.")

    all_points = np.concatenate(sample_points, axis=0)
    all_pixels = np.concatenate(sample_pixels, axis=0)
    all_distances: list[np.ndarray] = []
    all_face_indices: list[np.ndarray] = []

    chunk_size = max(int(CONFIGS.mesh_repair.texture_transfer_chunk_size), 1)
    for start in range(0, len(all_points), chunk_size):
        end = min(start + chunk_size, len(all_points))
        chunk_points = all_points[start:end]
        sqr_distances, face_indices, closest_points = igl.point_mesh_squared_distance(
            np.asarray(chunk_points, dtype=np.float64),
            np.asarray(source_mesh.vertices, dtype=np.float64),
            np.asarray(source_mesh.faces, dtype=np.int64),
        )
        face_indices = np.asarray(face_indices, dtype=np.int64)
        if np.any(face_indices < 0):
            raise RuntimeError("Closest-point query returned invalid source face indices.")

        source_triangles = np.asarray(source_mesh.vertices[source_mesh.faces[face_indices]], dtype=np.float64)
        barycentric = trimesh.triangles.points_to_barycentric(
            source_triangles,
            np.asarray(closest_points, dtype=np.float64),
        )
        source_face_uv = source_uv[np.asarray(source_mesh.faces[face_indices], dtype=np.int64)]
        sampled_source_uv = np.einsum("ni,nij->nj", barycentric, source_face_uv)
        colors = _sample_image_bilinear(source_image, sampled_source_uv)

        chunk_pixels = all_pixels[start:end]
        yy = chunk_pixels[:, 0]
        xx = chunk_pixels[:, 1]
        texture[yy, xx] += colors
        weight[yy, xx, 0] += 1.0
        all_distances.append(np.sqrt(np.asarray(sqr_distances, dtype=np.float64)))
        all_face_indices.append(face_indices)

    valid = weight[..., 0] > 0.0
    texture[valid] /= weight[valid]
    if np.any(valid):
        texture[~valid] = np.mean(texture[valid], axis=0, keepdims=False)
    texture[..., 3] = 1.0
    return {
        "image": np.clip(np.round(texture * 255.0), 0, 255).astype(np.uint8),
        "distances": np.concatenate(all_distances, axis=0),
        "face_indices": np.concatenate(all_face_indices, axis=0),
        "texel_count": int(len(all_points)),
    }


def _barycentric_2d(points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a = triangle[0]
    b = triangle[1]
    c = triangle[2]
    v0 = b - a
    v1 = c - a
    v2 = points - a
    denom = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(float(denom)) < 1e-12:
        return np.full(points.shape[:-1] + (3,), -1.0, dtype=np.float32)
    inv = 1.0 / denom
    w1 = (v2[..., 0] * v1[1] - v1[0] * v2[..., 1]) * inv
    w2 = (v0[0] * v2[..., 1] - v2[..., 0] * v0[1]) * inv
    w0 = 1.0 - w1 - w2
    return np.stack([w0, w1, w2], axis=-1).astype(np.float32)


def _read_texture_size(texture_path: Path) -> tuple[int, int]:
    with Image.open(texture_path) as image:
        width, height = image.size
    return max(int(width), 1), max(int(height), 1)


def _read_bake_texture_size(texture_path: Path) -> tuple[int, int]:
    width, height = _read_texture_size(texture_path)
    limit = max(int(CONFIGS.mesh_repair.texture_transfer_max_resolution), 1)
    max_dim = max(width, height)
    if max_dim <= limit:
        return width, height
    scale = float(limit) / float(max_dim)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _copy_with_vertex_affine(
    *,
    src_path: Path,
    dst_path: Path,
    scale: float | tuple[float, float, float] | None = None,
    delta: tuple[float, float, float] | None,
) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    scale_arr = np.asarray((1.0, 1.0, 1.0) if scale is None else scale, dtype=np.float64)
    if scale_arr.ndim == 0:
        scale_arr = np.repeat(scale_arr, 3)
    if scale_arr.shape != (3,):
        raise ValueError(f"Expected scalar or 3-vector scale, got shape {scale_arr.shape}")

    if delta is None and np.allclose(scale_arr, 1.0):
        shutil.copyfile(src_path, dst_path)
        return

    dx, dy, dz = (0.0, 0.0, 0.0) if delta is None else (float(delta[0]), float(delta[1]), float(delta[2]))
    rewritten_lines: list[str] = []
    for raw_line in src_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw_line.startswith("v "):
            parts = raw_line.split()
            if len(parts) >= 4:
                x = float(parts[1]) * float(scale_arr[0]) + dx
                y = float(parts[2]) * float(scale_arr[1]) + dy
                z = float(parts[3]) * float(scale_arr[2]) + dz
                tail = f" {' '.join(parts[4:])}" if len(parts) > 4 else ""
                rewritten_lines.append(f"v {x:.9f} {y:.9f} {z:.9f}{tail}")
                continue
        rewritten_lines.append(raw_line)
    dst_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")


def _copy_with_vertex_translation(
    *,
    src_path: Path,
    dst_path: Path,
    delta: tuple[float, float, float] | None,
) -> None:
    _copy_with_vertex_affine(src_path=src_path, dst_path=dst_path, scale=None, delta=delta)


def _rewrite_obj_mtllib(obj_path: Path, *, mtl_name: str) -> None:
    lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rewritten_lines: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("mtllib "):
            rewritten_lines.append(f"mtllib {mtl_name}")
            replaced = True
        else:
            rewritten_lines.append(line)
    if not replaced:
        rewritten_lines.insert(0, f"mtllib {mtl_name}")
    obj_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")


def _write_base_color_mtl(mtl_path: Path, *, texture_name: str) -> None:
    mtl_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl material_0",
                "Ka 0.000000 0.000000 0.000000",
                "Kd 1.000000 1.000000 1.000000",
                "Ks 0.000000 0.000000 0.000000",
                "d 1.0",
                "illum 2",
                f"map_Kd {texture_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
