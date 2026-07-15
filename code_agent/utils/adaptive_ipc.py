from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import trimesh

from code_agent.io_utils import dump_json, load_json_object

MESH_EXTENSIONS = {".obj", ".stl", ".ply", ".glb", ".gltf", ".dae"}
DIRECT_PRIMITIVE_SOURCE_FILES = ("src/body.py", "src/scene.py", "src/main.py")
DIRECT_PRIMITIVE_TYPES = {"Box", "Cylinder", "Sphere"}
ADAPTIVE_CONTACT_D_HAT_FACTOR = 0.2
ADAPTIVE_CONTACT_D_HAT_MIN_BBOX_FACTOR = 1e-5
ADAPTIVE_CONTACT_D_HAT_MAX_BBOX_FACTOR = 2e-3


def adaptive_contact_d_hat_report(
    *,
    case_root: Path,
    default_deformable_cfg: dict[str, object] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    case_root = case_root.resolve()
    repo_root = repo_root.resolve() if repo_root is not None else case_root
    default_deformable_cfg = dict(default_deformable_cfg or {})
    assets, manifest_loaded = _adaptive_assets(case_root=case_root, repo_root=repo_root)
    candidates = _adaptive_candidates(
        assets=assets,
        case_root=case_root,
        repo_root=repo_root,
        default_deformable_cfg=default_deformable_cfg,
    )
    if not candidates:
        return None
    return _adaptive_report_payload(candidates, manifest_loaded=manifest_loaded)


def _adaptive_report_payload(candidates: list[dict[str, Any]], *, manifest_loaded: bool) -> dict[str, Any] | None:
    selected = min(candidates, key=lambda item: item["raw_ipc_contact_d_hat"])
    global_bbox_diag = max(candidate["bbox_diag"] for candidate in candidates)
    min_d_hat = ADAPTIVE_CONTACT_D_HAT_MIN_BBOX_FACTOR * global_bbox_diag
    max_d_hat = ADAPTIVE_CONTACT_D_HAT_MAX_BBOX_FACTOR * global_bbox_diag
    ipc_contact_d_hat = float(np.clip(selected["raw_ipc_contact_d_hat"], min_d_hat, max_d_hat))
    if not np.isfinite(ipc_contact_d_hat) or ipc_contact_d_hat <= 0.0:
        return None

    return {
        "source": _adaptive_report_source(candidates, manifest_loaded=manifest_loaded),
        "rule": (
            "ipc_contact_d_hat = clamp("
            "min(0.2 * median_mesh_edge_or_bbox_feature), "
            "1e-5 * max_bbox_diag, 2e-3 * max_bbox_diag)"
        ),
        "source_assets": [candidate["logical_name"] for candidate in candidates],
        "source_asset_types": sorted({candidate["source_type"] for candidate in candidates}),
        "asset_reports": candidates,
        "selected_asset": selected["logical_name"],
        "selected_source_kind": selected["source_kind"],
        "edge_count": int(sum(candidate["edge_count"] for candidate in candidates)),
        "median_feature_length": selected["median_feature_length"],
        "bbox_diag": global_bbox_diag,
        "selected_bbox_diag": selected["bbox_diag"],
        "global_bbox_diag": global_bbox_diag,
        "raw_ipc_contact_d_hat": selected["raw_ipc_contact_d_hat"],
        "min_ipc_contact_d_hat": min_d_hat,
        "max_ipc_contact_d_hat": max_d_hat,
        "ipc_contact_d_hat": ipc_contact_d_hat,
    }


def _adaptive_report_source(candidates: list[dict[str, Any]], *, manifest_loaded: bool) -> str:
    has_direct_primitive = any(candidate.get("source_type") == "direct_primitive" for candidate in candidates)
    if manifest_loaded:
        if has_direct_primitive:
            return "assets/asset_manifest.json + generated source primitive morphs"
        return "assets/asset_manifest.json"
    if has_direct_primitive:
        has_asset_request = any(candidate.get("source_type") != "direct_primitive" for candidate in candidates)
        if has_asset_request:
            return "contracts/planner_output.json asset_requests + generated source primitive morphs"
        return "generated source primitive morphs"
    return "contracts/planner_output.json asset_requests"


def _adaptive_candidates(
    *,
    assets: list[dict[str, Any]],
    case_root: Path,
    repo_root: Path,
    default_deformable_cfg: dict[str, object],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for asset in assets:
        stats = None
        if asset.get("status") in {None, "ready"}:
            stats = _mesh_asset_geometry_stats(asset, case_root=case_root, repo_root=repo_root)
            if stats is None:
                stats = _bbox_asset_geometry_stats(asset)
        candidate = _adaptive_geometry_candidate(
            logical_name=_asset_label(asset),
            source_type=str(asset.get("source_type") or "unknown"),
            stats=stats,
        )
        if candidate is not None:
            candidates.append(candidate)
        candidates.extend(_mjcf_primitive_geom_candidates(asset, case_root=case_root, repo_root=repo_root))
    candidates.extend(_direct_primitive_candidates(case_root=case_root, repo_root=repo_root, cfg=default_deformable_cfg))
    return candidates


def apply_adaptive_contact_d_hat(
    deformable_cfg: dict[str, object],
    out_dir: Path,
    *,
    case_root: Path,
    default_deformable_cfg: dict[str, object] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    if not bool(deformable_cfg.get("ipc_contact_d_hat_adaptive", False)):
        return None
    report = adaptive_contact_d_hat_report(
        case_root=case_root,
        default_deformable_cfg=default_deformable_cfg,
        repo_root=repo_root,
    )
    if report is None:
        print("[adaptive-ipc] no ready asset geometry stats found; keeping configured ipc_contact_d_hat")
        return None
    deformable_cfg["ipc_contact_d_hat"] = float(report["ipc_contact_d_hat"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(report, out_dir / "adaptive_ipc_config.json")
    print(
        "[adaptive-ipc] "
        f"ipc_contact_d_hat={report['ipc_contact_d_hat']:.6g} "
        f"from asset={report['selected_asset']} "
        f"feature={report['median_feature_length']:.6g} "
        f"global_bbox_diag={report['global_bbox_diag']:.6g} "
        f"over {report['edge_count']} mesh edges"
    )
    return report


def _resolve_case_path(path_value: str | Path, *, case_root: Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    for candidate in (case_root / path, repo_root / path):
        if candidate.exists():
            return candidate
    return case_root / path


def _asset_label(asset: dict[str, Any]) -> str:
    return str(asset.get("logical_name") or asset.get("runtime_path") or asset.get("source_type") or "asset")


def _adaptive_assets(*, case_root: Path, repo_root: Path) -> tuple[list[dict[str, Any]], bool]:
    manifest_path = _resolve_case_path("assets/asset_manifest.json", case_root=case_root, repo_root=repo_root)
    assets: list[dict[str, Any]] = []
    manifest_loaded = False
    manifest = load_json_object(manifest_path)
    if manifest is not None:
        manifest_assets = manifest.get("assets")
        if isinstance(manifest_assets, list):
            assets.extend(asset for asset in manifest_assets if isinstance(asset, dict))
            manifest_loaded = True

    manifest_names = {
        str(asset.get("logical_name") or "")
        for asset in assets
        if asset.get("logical_name")
    }
    assets.extend(_planner_asset_request_assets(manifest_names, case_root=case_root, repo_root=repo_root))
    return assets, manifest_loaded


def _extent_vector3(value: object) -> np.ndarray | None:
    if value is None:
        return None
    try:
        extents = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if extents.size != 3 or not np.all(np.isfinite(extents)):
        return None
    extents = np.abs(extents)
    if not np.any(extents > 0.0):
        return None
    return extents


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0.0:
        return None
    return number


def _float_list(value: object) -> list[float] | None:
    if value is None:
        return None
    raw_items = value if isinstance(value, (list, tuple)) else str(value).replace(",", " ").split()
    values = []
    for item in raw_items:
        number = _positive_float(item)
        if number is None:
            return None
        values.append(number)
    return values


def _primitive_target_edge(feature_sizes: object, resolution: object, cfg: dict[str, object]) -> float | None:
    positive = [float(value) for value in feature_sizes if np.isfinite(value) and value > 0.0]
    if not positive:
        return None
    try:
        resolution = max(1, int(resolution))
    except (TypeError, ValueError):
        resolution = max(1, int(cfg.get("tet_resolution", 2) or 2))
    return max(float(min(positive)), 1e-6) / float(2 * resolution + 1)


def _primitive_stats_from_extents(
    extents: object,
    *,
    source_kind: str,
    feature_length: object = None,
) -> dict[str, Any] | None:
    extents = _extent_vector3(extents)
    if extents is None:
        return None
    positive_extents = extents[extents > 0.0]
    if positive_extents.size == 0:
        return None
    bbox_diag = float(np.linalg.norm(extents))
    if not np.isfinite(bbox_diag) or bbox_diag <= 0.0:
        return None
    feature = _positive_float(feature_length)
    if feature is None:
        feature = float(np.min(positive_extents))
    return {
        "source_kind": source_kind,
        "edge_count": 0,
        "median_feature_length": feature,
        "bbox_diag": bbox_diag,
    }


def _vertices_in_genesis_frame(vertices: np.ndarray, file_meshes_are_zup: object) -> np.ndarray:
    if file_meshes_are_zup is not False:
        return vertices
    transformed = vertices.copy()
    transformed[:, 0] = vertices[:, 0]
    transformed[:, 1] = -vertices[:, 2]
    transformed[:, 2] = vertices[:, 1]
    return transformed


def _apply_asset_scale(vertices: np.ndarray, asset: dict[str, Any]) -> np.ndarray:
    scale = asset.get("scale")
    if scale is None:
        return vertices
    try:
        scale_arr = np.asarray(scale, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return vertices
    if scale_arr.size == 1 and np.isfinite(scale_arr[0]):
        return vertices * float(scale_arr[0])
    if scale_arr.size == 3 and np.all(np.isfinite(scale_arr)):
        return vertices * scale_arr.reshape(1, 3)
    return vertices


def _mesh_asset_geometry_stats(
    asset: dict[str, Any],
    *,
    case_root: Path,
    repo_root: Path,
) -> dict[str, Any] | None:
    runtime_path = asset.get("runtime_path")
    if not runtime_path:
        return None
    mesh_path = _resolve_case_path(runtime_path, case_root=case_root, repo_root=repo_root)
    if not mesh_path.is_file() or mesh_path.suffix.lower() not in MESH_EXTENSIONS:
        return None
    try:
        mesh = trimesh.load_mesh(str(mesh_path), force="mesh", process=False, skip_texture=True)
    except Exception:
        return None
    if hasattr(mesh, "dump"):
        dumped = mesh.dump(concatenate=True)
        if dumped is not None:
            mesh = dumped
    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if vertices.size == 0 or edges.size == 0:
        return None
    vertices = _vertices_in_genesis_frame(vertices, asset.get("file_meshes_are_zup"))
    vertices = _apply_asset_scale(vertices, asset)
    bounds_min = np.min(vertices, axis=0)
    bounds_max = np.max(vertices, axis=0)
    bbox_diag = float(np.linalg.norm(bounds_max - bounds_min))
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    if lengths.size == 0 or not np.isfinite(bbox_diag) or bbox_diag <= 0.0:
        return None
    return {
        "source_kind": "mesh_edges",
        "edge_count": int(lengths.size),
        "median_feature_length": float(np.median(lengths)),
        "bbox_diag": bbox_diag,
    }


def _bbox_asset_geometry_stats(asset: dict[str, Any]) -> dict[str, Any] | None:
    extents = _extent_vector3(asset.get("bbox"))
    if extents is None:
        extents = _extent_vector3(asset.get("scale"))
    if extents is None:
        return None
    positive_extents = extents[extents > 0.0]
    if positive_extents.size == 0:
        return None
    bbox_diag = float(np.linalg.norm(extents))
    if not np.isfinite(bbox_diag) or bbox_diag <= 0.0:
        return None
    return {
        "source_kind": "bbox_fallback",
        "edge_count": 0,
        "median_feature_length": float(np.min(positive_extents)),
        "bbox_diag": bbox_diag,
    }


def _adaptive_geometry_candidate(
    *,
    logical_name: str,
    source_type: str,
    stats: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if stats is None:
        return None
    raw_d_hat = ADAPTIVE_CONTACT_D_HAT_FACTOR * stats["median_feature_length"]
    if not np.isfinite(raw_d_hat) or raw_d_hat <= 0.0:
        return None
    return {
        "logical_name": str(logical_name),
        "source_type": str(source_type or "unknown"),
        "source_kind": stats["source_kind"],
        "edge_count": stats["edge_count"],
        "median_feature_length": stats["median_feature_length"],
        "bbox_diag": stats["bbox_diag"],
        "raw_ipc_contact_d_hat": raw_d_hat,
    }


def _mjcf_geom_collision_enabled(geom: ET.Element) -> bool:
    geom_type = str(geom.attrib.get("type") or "sphere")
    if geom_type in {"plane", "mesh", "hfield"}:
        return False
    geom_name = str(geom.attrib.get("name") or "").lower()
    material_name = str(geom.attrib.get("material") or "").lower()
    if "dummy" in geom_name or "dummy" in material_name:
        return False
    contype = str(geom.attrib.get("contype", "1")).strip()
    conaffinity = str(geom.attrib.get("conaffinity", "1")).strip()
    return contype not in {"0", "0.0"} and conaffinity not in {"0", "0.0"}


def _mjcf_geom_stats(geom: ET.Element) -> dict[str, Any] | None:
    geom_type = str(geom.attrib.get("type") or "sphere")
    size = _float_list(geom.attrib.get("size")) or []
    fromto = _float_list(geom.attrib.get("fromto")) or []
    if geom_type == "box" and len(size) >= 3:
        return _primitive_stats_from_extents(
            [2.0 * size[0], 2.0 * size[1], 2.0 * size[2]],
            source_kind="mjcf_primitive_geom",
        )
    if geom_type in {"sphere", "ellipsoid"} and len(size) >= 1:
        extents = (
            [2.0 * size[0], 2.0 * size[1], 2.0 * size[2]]
            if geom_type == "ellipsoid" and len(size) >= 3
            else [2.0 * size[0], 2.0 * size[0], 2.0 * size[0]]
        )
        return _primitive_stats_from_extents(extents, source_kind="mjcf_primitive_geom")
    if geom_type in {"cylinder", "capsule"} and len(size) >= 2:
        radius = size[0]
        axial_length = 2.0 * size[1]
        if len(fromto) == 6:
            axial_length = float(np.linalg.norm(np.asarray(fromto[3:], dtype=float) - np.asarray(fromto[:3], dtype=float)))
        if geom_type == "capsule":
            axial_length += 2.0 * radius
        return _primitive_stats_from_extents(
            [2.0 * radius, 2.0 * radius, axial_length],
            source_kind="mjcf_primitive_geom",
        )
    return None


def _mjcf_primitive_geom_candidates(
    asset: dict[str, Any],
    *,
    case_root: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    runtime_path = asset.get("runtime_path")
    if not runtime_path or str(asset.get("source_type") or "") not in {"mjcf", "urdf"}:
        return []
    xml_path = _resolve_case_path(runtime_path, case_root=case_root, repo_root=repo_root)
    if not xml_path.is_file() or xml_path.suffix.lower() != ".xml":
        return []
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return []
    source_type = str(asset.get("source_type") or "mjcf")
    asset_name = _asset_label(asset)
    candidates = []
    for index, geom in enumerate(root.findall(".//geom")):
        if not _mjcf_geom_collision_enabled(geom):
            continue
        stats = _mjcf_geom_stats(geom)
        if stats is None:
            continue
        geom_name = geom.attrib.get("name") or f"geom_{index}"
        candidate = _adaptive_geometry_candidate(
            logical_name=f"{asset_name}/geom:{geom_name}", source_type=source_type, stats=stats
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _attribute_path(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_path(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _safe_eval_ast(node: ast.AST, constants: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_ast(item, constants) for item in node.elts)
    if isinstance(node, ast.List):
        return [_safe_eval_ast(item, constants) for item in node.elts]
    if isinstance(node, ast.UnaryOp):
        value = _safe_eval_ast(node.operand, constants)
        if value is None:
            return None
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = _safe_eval_ast(node.left, constants)
        right = _safe_eval_ast(node.right, constants)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    return None


def _collect_literal_assignments(statements: list[ast.stmt], constants: dict[str, Any]) -> None:
    for node in statements:
        if isinstance(node, ast.Assign):
            try:
                value = _safe_eval_ast(node.value, constants)
            except Exception:
                value = None
            if value is not None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _collect_literal_assignments(list(node.body), constants)
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt):
                    _collect_literal_assignments([child], constants)
                elif isinstance(child, list):
                    _collect_literal_assignments([item for item in child if isinstance(item, ast.stmt)], constants)


def _literal_constants(tree: ast.AST, cfg: dict[str, object]) -> dict[str, Any]:
    constants: dict[str, Any] = {"DEFAULT_TET_RESOLUTION": int(cfg.get("tet_resolution", 2) or 2)}
    if isinstance(tree, ast.Module):
        _collect_literal_assignments(list(tree.body), constants)
    return constants


def _call_arg(call: ast.Call, name: str, index: int | None, constants: dict[str, Any], default: Any = None) -> Any:
    for keyword in call.keywords:
        if keyword.arg == name:
            try:
                value = _safe_eval_ast(keyword.value, constants)
            except Exception:
                value = None
            return default if value is None else value
    if index is not None and index < len(call.args):
        try:
            value = _safe_eval_ast(call.args[index], constants)
        except Exception:
            value = None
        return default if value is None else value
    return default


def _call_collision_enabled(call: ast.Call, constants: dict[str, Any]) -> bool:
    collision = _call_arg(call, "collision", None, constants, True)
    if collision is False:
        return False
    contype = _call_arg(call, "contype", None, constants, 1)
    conaffinity = _call_arg(call, "conaffinity", None, constants, 1)
    try:
        return int(contype) != 0 and int(conaffinity) != 0
    except (TypeError, ValueError):
        return True


def _direct_primitive_stats(
    primitive_type: str,
    call: ast.Call,
    constants: dict[str, Any],
    cfg: dict[str, object],
) -> dict[str, Any] | None:
    if not _call_collision_enabled(call, constants):
        return None
    resolution = _call_arg(call, "tet_resolution", None, constants, constants.get("DEFAULT_TET_RESOLUTION", 2))
    if primitive_type == "Box":
        size = _extent_vector3(_call_arg(call, "size", None, constants))
        lower = _extent_vector3(_call_arg(call, "lower", None, constants))
        upper = _extent_vector3(_call_arg(call, "upper", None, constants))
        if size is None and lower is not None and upper is not None:
            size = np.abs(upper - lower)
        extents = size
    elif primitive_type == "Cylinder":
        radius = _positive_float(_call_arg(call, "radius", None, constants, 0.5))
        height = _positive_float(_call_arg(call, "height", None, constants, 1.0))
        extents = None if radius is None or height is None else [2.0 * radius, 2.0 * radius, height]
    elif primitive_type == "Sphere":
        radius = _positive_float(_call_arg(call, "radius", None, constants, 0.5))
        extents = None if radius is None else [2.0 * radius, 2.0 * radius, 2.0 * radius]
    else:
        return None
    if extents is None:
        return None
    feature = _primitive_target_edge(extents, resolution, cfg)
    return _primitive_stats_from_extents(extents, source_kind="direct_primitive_morph", feature_length=feature)


def _direct_primitive_candidates(*, case_root: Path, repo_root: Path, cfg: dict[str, object]) -> list[dict[str, Any]]:
    candidates = []
    for source_path in DIRECT_PRIMITIVE_SOURCE_FILES:
        path = _resolve_case_path(source_path, case_root=case_root, repo_root=repo_root)
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        constants = _literal_constants(tree, cfg)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            primitive_type = _attribute_path(node.func).rsplit(".", 1)[-1]
            if primitive_type not in DIRECT_PRIMITIVE_TYPES:
                continue
            stats = _direct_primitive_stats(primitive_type, node, constants, cfg)
            if stats is None:
                continue
            candidate = _adaptive_geometry_candidate(
                logical_name=f"{source_path}:{primitive_type}",
                source_type="direct_primitive",
                stats=stats,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _source_type_from_planner_asset_type(asset_type: object) -> str:
    asset_type = str(asset_type or "")
    if asset_type in {"generated_xml", "xml", "mjcf"}:
        return "mjcf"
    if asset_type in {"generated_mesh", "mesh", "input_mesh", "repo_asset", "derived_mesh"}:
        return asset_type
    if "primitive" in asset_type:
        return "primitive"
    return asset_type or "planner_asset_request"


def _planner_asset_request_assets(
    existing_names: set[str],
    *,
    case_root: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    planner_path = _resolve_case_path("contracts/planner_output.json", case_root=case_root, repo_root=repo_root)
    planner = load_json_object(planner_path)
    if planner is None:
        return []
    requests = planner.get("asset_requests")
    if not isinstance(requests, list):
        return []
    assets = []
    for request in requests:
        if not isinstance(request, dict):
            continue
        name = str(request.get("name") or "")
        if not name or name in existing_names:
            continue
        assets.append(
            {
                "logical_name": name,
                "source_type": _source_type_from_planner_asset_type(request.get("asset_type")),
                "runtime_path": None,
                "scale": request.get("scale"),
                "bbox": request.get("bbox"),
                "file_meshes_are_zup": None,
                "simulation_role": str(request.get("simulation_role") or request.get("purpose") or ""),
                "status": "ready",
            }
        )
    return assets
