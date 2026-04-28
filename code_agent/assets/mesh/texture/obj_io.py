from __future__ import annotations

from dataclasses import dataclass
import shutil
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ObjUvMesh:
    vertices: np.ndarray
    texcoords: np.ndarray
    face_vertex_indices: np.ndarray
    face_texcoord_indices: np.ndarray


def parse_obj_with_uv(obj_path: Path) -> ObjUvMesh:
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


def write_parameterized_obj(*, obj_path: Path, vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray) -> None:
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


def copy_with_vertex_translation(
    *,
    src_path: Path,
    dst_path: Path,
    delta: tuple[float, float, float] | None,
) -> None:
    copy_with_vertex_affine(src_path=src_path, dst_path=dst_path, scale=None, delta=delta)


def copy_with_vertex_affine(
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


def rewrite_obj_mtllib(obj_path: Path, *, mtl_name: str) -> None:
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


def rewrite_mtl_base_color(mtl_path: Path, *, texture_name: str) -> None:
    if not mtl_path.exists():
        raise FileNotFoundError(f"Expected MTL file not found: {mtl_path}")

    lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rewritten_lines: list[str] = []
    saw_map_kd = False
    saw_newmtl = False
    inserted_in_current_block = False

    for line in lines:
        if line.startswith("newmtl "):
            if saw_newmtl and not inserted_in_current_block:
                rewritten_lines.append(f"map_Kd {texture_name}")
            rewritten_lines.append(line)
            saw_newmtl = True
            inserted_in_current_block = False
            continue
        if line.startswith("map_Kd "):
            rewritten_lines.append(f"map_Kd {texture_name}")
            saw_map_kd = True
            inserted_in_current_block = True
            continue
        rewritten_lines.append(line)

    if saw_newmtl and not inserted_in_current_block:
        rewritten_lines.append(f"map_Kd {texture_name}")
        saw_map_kd = True
    if not saw_newmtl and not saw_map_kd:
        rewritten_lines.extend(["newmtl material_0", f"map_Kd {texture_name}"])

    mtl_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")


def write_base_color_mtl(mtl_path: Path, *, texture_name: str) -> None:
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


def _obj_index_to_zero_based(token: str, item_count: int) -> int:
    value = int(token)
    if value > 0:
        return value - 1
    if value < 0:
        return item_count + value
    raise RuntimeError("OBJ indices are 1-based; found invalid 0 index.")
