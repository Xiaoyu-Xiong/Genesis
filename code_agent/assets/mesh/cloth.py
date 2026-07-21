from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from code_agent.configs import CONFIGS
from code_agent.io_utils import dump_json


CLOTH_TARGET_EDGE_LENGTH_FIELD = "cloth_target_edge_length"

CLOTH_MESH_ASSET_TYPES = {
    "cloth_mesh",
    "cloth_mesh_square",
    "cloth_mesh_rectangle",
    "cloth_mesh_disk",
    "cloth_mesh_cylinder",
    "cloth_mesh_sphere",
}

MESHY_CLOTH_ASSET_TYPE = "generated_mesh"


class UnsupportedClothMeshShapeError(ValueError):
    """Raised when a procedural cloth request asks for a shape the generator cannot build."""


def is_cloth_mesh_request(request: dict[str, Any]) -> bool:
    return str(request.get("asset_type", "")).strip().lower() in CLOTH_MESH_ASSET_TYPES


def is_meshy_generated_cloth_request(request: dict[str, Any]) -> bool:
    if str(request.get("asset_type", "")).strip().lower() != MESHY_CLOTH_ASSET_TYPE:
        return False
    text = _request_cloth_semantics_text(request)
    if not text:
        return False
    explicit_cloth_markers = (
        "fem.cloth",
        "fem cloth",
        "cloth shell",
        "cloth-like shell",
        "clothlike shell",
        "thin-shell cloth",
        "thin shell cloth",
        "manifold cloth",
        "closed cloth",
        "cloth surface mesh",
        "fabric shell",
    )
    if any(marker in text for marker in explicit_cloth_markers):
        return True
    return "cloth" in text and "shell" in text and "manifold" in text


def generate_cloth_mesh_asset(
    *,
    request: dict[str, Any],
    output_root: Path,
    index: int,
) -> dict[str, Any]:
    name = str(request.get("name") or "cloth_mesh")
    output_dir = output_root / f"{index:02d}_{_slugify(name)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / f"{_slugify(name)}.obj"
    try:
        unsupported_reason = _unsupported_complex_cloth_shape_reason(request)
        if unsupported_reason is not None:
            raise UnsupportedClothMeshShapeError(unsupported_reason)
        shape = _cloth_shape_from_request(request)
        target_edge_length = _target_edge_length(request)
        vertices, faces = _build_cloth_mesh(shape, request, target_edge_length=target_edge_length)
        stats = _mesh_stats(vertices, faces)
        stats.update(
            {
                "target_edge_length": target_edge_length,
                "target_edge_length_source": (
                    "asset_request" if request.get(CLOTH_TARGET_EDGE_LENGTH_FIELD) is not None else "config"
                ),
                "max_faces": int(CONFIGS.deformable.cloth_max_faces),
            }
        )
        _validate_cloth_mesh(stats)
        _write_obj(mesh_path, vertices, faces)
        manifest_entry = _manifest_entry_from_cloth_mesh(
            request=request,
            mesh_path=mesh_path,
            shape=shape,
            stats=stats,
        )
        payload = {
            "ok": True,
            "request": request,
            "manifest_entry": manifest_entry,
            "cloth_mesh": {
                "shape": shape,
                "mesh_path": str(mesh_path),
                "stats": stats,
            },
        }
    except UnsupportedClothMeshShapeError as exc:
        manifest_entry = failed_cloth_mesh_manifest_entry(request, str(exc))
        payload = {
            "ok": False,
            "request": request,
            "manifest_entry": manifest_entry,
            "error": str(exc),
            "failure_class": "cloth_mesh.unsupported_shape",
            "recommended_owner": "planner",
            "repair_summary": (
                "Rewrite this procedural FEM.Cloth request to square, rectangle/ribbon/strip, disk/circle, "
                "cylinder/tube/sleeve, or sphere/balloon; use generated_mesh only for an explicitly closed "
                "manifold cloth shell."
            ),
        }
    except Exception as exc:  # noqa: BLE001 - record asset-level failure.
        manifest_entry = failed_cloth_mesh_manifest_entry(request, f"{type(exc).__name__}: {exc}")
        payload = {
            "ok": False,
            "request": request,
            "manifest_entry": manifest_entry,
            "error": manifest_entry["notes"][0],
            "failure_class": "cloth_mesh.generation_failed",
        }
    dump_json(payload, output_dir / "cloth_mesh_report.json")
    return payload


def failed_cloth_mesh_manifest_entry(request: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "logical_name": str(request.get("name", "cloth_mesh")),
        "source_type": "cloth_mesh",
        "runtime_path": "unavailable",
        "visual_path": None,
        "scale": 1.0,
        "bbox": _request_size(request),
        "file_meshes_are_zup": True,
        "texture_path": None,
        "validation": {"cloth_mesh": {"ok": False, "error": error}},
        "asset_request": {str(key): value for key, value in request.items()},
        "simulation_role": str(request.get("simulation_role", "FEM cloth surface mesh")),
        "status": "failed",
        "notes": [error],
    }


def _cloth_shape_from_request(request: dict[str, Any]) -> str:
    asset_type = str(request.get("asset_type", "")).strip().lower()
    if asset_type.startswith("cloth_mesh_"):
        return asset_type.removeprefix("cloth_mesh_")
    text = " ".join(
        str(request.get(key) or "").lower() for key in ("name", "purpose", "simulation_role", "texture_needs")
    )
    if any(token in text for token in ("rectangular", "rectangle", "ribbon", "strip")):
        return "rectangle"
    if any(token in text for token in ("circular", "circle", "disk", "disc")):
        return "disk"
    if any(token in text for token in ("cylindrical", "cylinder", "tube", "sleeve")):
        return "cylinder"
    if any(token in text for token in ("spherical", "sphere", "balloon", "shell ball")):
        return "sphere"
    if any(token in text for token in ("square", "sheet", "cloth")):
        return "square"
    return "square"


def _unsupported_complex_cloth_shape_reason(request: dict[str, Any]) -> str | None:
    text = " ".join(
        str(request.get(key) or "").lower() for key in ("name", "purpose", "simulation_role", "texture_needs")
    )
    complex_shape_tokens = (
        "silhouette",
        "cutout",
        "cut-out",
        "icon",
        "logo",
        "profile",
        "outline",
        "animal-shaped",
        "animal shaped",
        "character-shaped",
        "character shaped",
        "custom shape",
        "custom-shaped",
        "armadillo",
        "dragon",
        "bunny",
        "panda",
        "giraffe",
        "person",
        "human",
        "star",
        "heart",
        "letter",
        "number",
    )
    if not any(token in text for token in complex_shape_tokens):
        return None

    basic_shape_tokens = (
        "square",
        "rectangular",
        "rectangle",
        "ribbon",
        "strip",
        "circular",
        "circle",
        "disk",
        "disc",
        "cylindrical",
        "cylinder",
        "tube",
        "sleeve",
        "spherical",
        "sphere",
        "balloon",
        "shell ball",
    )
    if any(token in text for token in basic_shape_tokens):
        return None

    return (
        "Unsupported procedural cloth_mesh shape. The local procedural FEM.Cloth mesh generator only supports "
        "square, rectangle/ribbon/strip, disk/circle, cylinder/tube, and sphere/balloon surface meshes. It cannot generate "
        "arbitrary 2D silhouettes, icons, logos, animal-shaped cutouts, or other complex cloth outlines."
    )


def _build_cloth_mesh(
    shape: str,
    request: dict[str, Any],
    *,
    target_edge_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    bbox = _request_size(request)
    if shape == "square":
        side = _positive_at(bbox, 0, 1.0)
        return _square_or_rectangle(width=side, height=side, target_edge_length=target_edge_length)
    if shape == "rectangle":
        width = _positive_at(bbox, 0, 1.0)
        height = _positive_at(bbox, 1, 0.5)
        return _square_or_rectangle(width=width, height=height, target_edge_length=target_edge_length)
    if shape == "disk":
        diameter = min(_positive_at(bbox, 0, 1.0), _positive_at(bbox, 1, 1.0))
        return _disk(radius=0.5 * diameter, target_edge_length=target_edge_length)
    if shape == "cylinder":
        diameter = min(_positive_at(bbox, 0, 0.5), _positive_at(bbox, 1, 0.5))
        height = _positive_at(bbox, 2, 0.8)
        return _cylindrical_shell(radius=0.5 * diameter, height=height, target_edge_length=target_edge_length)
    if shape == "sphere":
        diameter = min(_positive_at(bbox, 0, 0.8), _positive_at(bbox, 1, 0.8), _positive_at(bbox, 2, 0.8))
        return _spherical_shell(radius=0.5 * diameter, target_edge_length=target_edge_length)
    raise ValueError(f"Unsupported cloth_mesh shape: {shape}")


def _square_or_rectangle(
    *,
    width: float,
    height: float,
    target_edge_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = _grid_counts(width, height, target_edge_length=target_edge_length)
    xs = np.linspace(-0.5 * width, 0.5 * width, nx + 1)
    ys = np.linspace(-0.5 * height, 0.5 * height, ny + 1)
    vertices = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b = a + 1
            c = a + (nx + 1)
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return vertices, np.asarray(faces, dtype=np.int64)


def _cylindrical_shell(
    *,
    radius: float,
    height: float,
    target_edge_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    circumference = 2.0 * math.pi * radius
    n_theta, n_z = _grid_counts(circumference, height, target_edge_length=target_edge_length)
    n_theta = max(8, n_theta)
    n_z = max(2, n_z)
    vertices = []
    for iz in range(n_z + 1):
        z = -0.5 * height + height * iz / n_z
        for it in range(n_theta):
            theta = 2.0 * math.pi * it / n_theta
            vertices.append((radius * math.cos(theta), radius * math.sin(theta), z))
    faces = []
    for iz in range(n_z):
        for it in range(n_theta):
            a = iz * n_theta + it
            b = iz * n_theta + (it + 1) % n_theta
            c = (iz + 1) * n_theta + it
            d = (iz + 1) * n_theta + (it + 1) % n_theta
            faces.append((a, b, d))
            faces.append((a, d, c))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _disk(*, radius: float, target_edge_length: float) -> tuple[np.ndarray, np.ndarray]:
    # Eight vertices per radial band keep triangles near-uniform and put boundary
    # vertices on both coordinate axes for an exact requested diameter.
    ring_count = max(1, int(math.ceil(radius / target_edge_length)))
    ring_count = min(ring_count, max(1, int(math.sqrt(CONFIGS.deformable.cloth_max_faces / 8))))

    vertices = [(0.0, 0.0, 0.0)]
    rings: list[list[int]] = []
    for ring_index in range(1, ring_count + 1):
        ring_radius = radius * ring_index / ring_count
        vertex_count = 8 * ring_index
        ring = []
        for vertex_index in range(vertex_count):
            theta = 2.0 * math.pi * vertex_index / vertex_count
            ring.append(len(vertices))
            vertices.append((ring_radius * math.cos(theta), ring_radius * math.sin(theta), 0.0))
        rings.append(ring)

    faces = [(0, rings[0][i], rings[0][(i + 1) % len(rings[0])]) for i in range(len(rings[0]))]
    for inner, outer in zip(rings, rings[1:]):
        faces.extend(_stitch_concentric_rings(inner, outer))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _stitch_concentric_rings(inner: list[int], outer: list[int]) -> list[tuple[int, int, int]]:
    faces: list[tuple[int, int, int]] = []
    i = j = 0
    while i < len(inner) or j < len(outer):
        a, b = inner[i % len(inner)], outer[j % len(outer)]
        inner_next = (i + 1) * len(outer)
        outer_next = (j + 1) * len(inner)
        if inner_next < outer_next:
            i += 1
            faces.append((a, b, inner[i % len(inner)]))
        elif outer_next < inner_next:
            j += 1
            faces.append((a, b, outer[j % len(outer)]))
        else:
            i += 1
            j += 1
            faces.append((a, b, outer[j % len(outer)]))
            faces.append((a, outer[j % len(outer)], inner[i % len(inner)]))
    return faces


def _spherical_shell(*, radius: float, target_edge_length: float) -> tuple[np.ndarray, np.ndarray]:
    n_lat = max(4, int(math.ceil(math.pi * radius / target_edge_length)))
    n_lon = max(8, 2 * n_lat)
    while 2 * n_lon * max(1, n_lat - 1) > CONFIGS.deformable.cloth_max_faces and n_lat > 4:
        n_lat -= 1
        n_lon = max(8, 2 * n_lat)

    vertices = [(0.0, 0.0, radius)]
    for ilat in range(1, n_lat):
        phi = math.pi * ilat / n_lat
        z = radius * math.cos(phi)
        ring_radius = radius * math.sin(phi)
        for ilon in range(n_lon):
            theta = 2.0 * math.pi * ilon / n_lon
            vertices.append((ring_radius * math.cos(theta), ring_radius * math.sin(theta), z))
    south_idx = len(vertices)
    vertices.append((0.0, 0.0, -radius))

    faces = []
    first_ring = 1
    for ilon in range(n_lon):
        faces.append((0, first_ring + ilon, first_ring + (ilon + 1) % n_lon))
    for ilat in range(n_lat - 2):
        ring_a = first_ring + ilat * n_lon
        ring_b = ring_a + n_lon
        for ilon in range(n_lon):
            a = ring_a + ilon
            b = ring_a + (ilon + 1) % n_lon
            c = ring_b + ilon
            d = ring_b + (ilon + 1) % n_lon
            faces.append((a, c, d))
            faces.append((a, d, b))
    last_ring = first_ring + (n_lat - 2) * n_lon
    for ilon in range(n_lon):
        faces.append((south_idx, last_ring + (ilon + 1) % n_lon, last_ring + ilon))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _grid_counts(width: float, height: float, *, target_edge_length: float) -> tuple[int, int]:
    nx = max(1, int(math.ceil(width / target_edge_length)))
    ny = max(1, int(math.ceil(height / target_edge_length)))
    while 2 * nx * ny > CONFIGS.deformable.cloth_max_faces and (nx > 1 or ny > 1):
        if nx >= ny and nx > 1:
            nx -= 1
        elif ny > 1:
            ny -= 1
    return nx, ny


def _target_edge_length(request: dict[str, Any]) -> float:
    value = request.get(CLOTH_TARGET_EDGE_LENGTH_FIELD)
    if value is None:
        value = CONFIGS.deformable.cloth_target_edge_length_default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{CLOTH_TARGET_EDGE_LENGTH_FIELD} must be a positive finite number in meters.")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{CLOTH_TARGET_EDGE_LENGTH_FIELD} must be a positive finite number in meters.")
    return max(value, 1e-4)


def _positive_at(values: list[float] | None, index: int, default: float) -> float:
    if values is None or index >= len(values):
        return default
    value = float(values[index])
    return value if value > 0.0 and math.isfinite(value) else default


def _request_size(request: dict[str, Any]) -> list[float] | None:
    value = request.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    output: list[float] = []
    for item in value:
        if not isinstance(item, int | float) or isinstance(item, bool) or float(item) <= 0.0:
            return None
        output.append(float(item))
    return output


def _mesh_stats(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min
    areas = 0.5 * np.linalg.norm(
        np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]),
        axis=1,
    )
    return {
        "ok": True,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "edge_count": int(len(edges)),
        "bbox_min": [float(item) for item in bbox_min],
        "bbox_max": [float(item) for item in bbox_max],
        "bbox_size": [float(item) for item in bbox_size],
        "median_edge_length": float(np.median(lengths)),
        "min_face_area": float(np.min(areas)),
        "max_face_area": float(np.max(areas)),
        "is_watertight_required": False,
    }


def _validate_cloth_mesh(stats: dict[str, Any]) -> None:
    if stats["vertex_count"] <= 0 or stats["face_count"] <= 0:
        raise ValueError("Cloth mesh has no vertices or faces.")
    if stats["face_count"] > CONFIGS.deformable.cloth_max_faces:
        raise ValueError(
            f"Cloth mesh face count {stats['face_count']} exceeds configured max {CONFIGS.deformable.cloth_max_faces}."
        )
    if stats["min_face_area"] <= 1e-14:
        raise ValueError("Cloth mesh contains degenerate triangles.")


def _manifest_entry_from_cloth_mesh(
    *,
    request: dict[str, Any],
    mesh_path: Path,
    shape: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "logical_name": str(request.get("name", "cloth_mesh")),
        "source_type": "cloth_mesh",
        "runtime_path": str(mesh_path.resolve()),
        "visual_path": str(mesh_path.resolve()),
        "scale": 1.0,
        "bbox": stats["bbox_size"],
        "file_meshes_are_zup": True,
        "texture_path": None,
        "validation": {"cloth_mesh": {"shape": shape, **stats}},
        "asset_request": {str(key): value for key, value in request.items()},
        "simulation_role": str(request.get("simulation_role", "FEM cloth surface mesh")),
        "status": "ready",
        "notes": [
            "Procedural cloth_mesh generated locally for FEM.Cloth IPC shell simulation.",
            "This mesh is an open or closed surface mesh and intentionally does not use watertight/tetgen repair.",
            "Use runtime_path with gs.morphs.Mesh and gs.materials.FEM.Cloth; do not use PBD cloth.",
        ],
    }


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for x, y, z in vertices:
        lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
    for a, b, c in faces:
        lines.append(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug or "cloth_mesh"


def _request_cloth_semantics_text(request: dict[str, Any]) -> str:
    return " ".join(str(request.get(key) or "").strip().lower() for key in ("name", "purpose", "simulation_role"))
