from __future__ import annotations

from pathlib import Path
from typing import Any

import igl
import numpy as np
from PIL import Image
import trimesh

from ....configs import CONFIGS
from .obj_io import ObjUvMesh, parse_obj_with_uv


def bake_texture_from_source_mesh(
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

    target_obj = parse_obj_with_uv(target_parameterized_mesh_path)
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
        "target_vertex_count": len(target_obj.vertices),
        "target_face_count": len(target_obj.face_vertex_indices),
        "source_face_hit_count": len(np.unique(np.asarray(baked["face_indices"], dtype=np.int64))),
        "distance_min": float(np.min(distances)) if len(distances) else 0.0,
        "distance_mean": float(np.mean(distances)) if len(distances) else 0.0,
        "distance_max": float(np.max(distances)) if len(distances) else 0.0,
        "texture_size": [int(texture_size[0]), int(texture_size[1])],
        "bake_mode": "per_texel_source_uv",
        "target_texel_count": int(baked["texel_count"]),
    }


def read_bake_texture_size(texture_path: Path) -> tuple[int, int]:
    width, height = read_texture_size(texture_path)
    limit = max(int(CONFIGS.mesh_repair.texture_transfer_max_resolution), 1)
    max_dim = max(width, height)
    if max_dim <= limit:
        return width, height
    scale = float(limit) / float(max_dim)
    return max(1, round(width * scale)), max(1, round(height * scale))


def read_texture_size(texture_path: Path) -> tuple[int, int]:
    with Image.open(texture_path) as image:
        width, height = image.size
    return max(int(width), 1), max(int(height), 1)


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
        "texel_count": len(all_points),
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
        return np.full((*points.shape[:-1], 3), -1.0, dtype=np.float32)
    inv = 1.0 / denom
    w1 = (v2[..., 0] * v1[1] - v1[0] * v2[..., 1]) * inv
    w2 = (v0[0] * v2[..., 1] - v2[..., 0] * v0[1]) * inv
    w0 = 1.0 - w1 - w2
    return np.stack([w0, w1, w2], axis=-1).astype(np.float32)
