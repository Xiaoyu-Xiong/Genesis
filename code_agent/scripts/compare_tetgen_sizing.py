from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REGULAR_TET_VOLUME_FACTOR = math.sqrt(2.0) / 12.0
FIXED_SCHEMES = (
    "bbox_current",
    "bbox_resolution1",
    "surface_edge_p50",
    "surface_area_equivalent",
    "volume_budget",
    "quality_only_no_a",
    "quality_only_relaxed",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TetGen sizing policies on generated FEM mesh assets.")
    parser.add_argument("--case-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-tets", type=int, default=20_000)
    parser.add_argument("--tet-resolution", type=int, default=2)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--worker-memory-gb", type=float, default=4.0)
    parser.add_argument("--quality-sample-size", type=int, default=200_000)
    parser.add_argument("--validate-results", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--genesis-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mesh", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--visual-mesh", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--scale", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--file-meshes-are-zup", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scheme", help=argparse.SUPPRESS)
    parser.add_argument("--maxvolume", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--minratio", type=float, default=1.1, help=argparse.SUPPRESS)
    parser.add_argument("--mindihedral", type=float, default=15.0, help=argparse.SUPPRESS)
    return parser.parse_args()


def _percentiles(values, points=(1, 10, 50, 90, 99)) -> dict[str, float]:
    import numpy as np

    result = np.percentile(values, points)
    return {f"p{point:02d}": float(value) for point, value in zip(points, result, strict=True)}


def _tet_quality(nodes, elems, sample_size: int) -> dict[str, Any]:
    import numpy as np

    tet_count = len(elems)
    sample_count = min(tet_count, sample_size)
    sample_ids = np.linspace(0, tet_count - 1, sample_count, dtype=np.int64)
    points = nodes[elems[sample_ids]]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_lengths = np.stack([np.linalg.norm(points[:, i] - points[:, j], axis=1) for i, j in edge_pairs], axis=1)
    sample_volumes = np.abs(
        np.einsum(
            "ij,ij->i",
            points[:, 0] - points[:, 3],
            np.cross(points[:, 1] - points[:, 3], points[:, 2] - points[:, 3]),
        )
    ) / 6.0
    max_edges = np.max(edge_lengths, axis=1)
    min_edges = np.min(edge_lengths, axis=1)
    volume_edge_quality = 6.0 * math.sqrt(2.0) * sample_volumes / np.maximum(max_edges**3, 1e-30)

    signs = Counter()
    min_volume = math.inf
    max_volume = 0.0
    zero_volume_count = 0
    for start in range(0, tet_count, 200_000):
        tet = nodes[elems[start : start + 200_000]]
        signed = np.einsum(
            "ij,ij->i",
            tet[:, 0] - tet[:, 3],
            np.cross(tet[:, 1] - tet[:, 3], tet[:, 2] - tet[:, 3]),
        ) / 6.0
        absolute = np.abs(signed)
        min_volume = min(min_volume, float(np.min(absolute)))
        max_volume = max(max_volume, float(np.max(absolute)))
        tolerance = max(max_volume * 1e-12, 1e-30)
        zero_volume_count += int(np.count_nonzero(absolute <= tolerance))
        signs.update({"positive": int(np.count_nonzero(signed > 0)), "negative": int(np.count_nonzero(signed < 0))})

    return {
        "sample_count": sample_count,
        "tet_volume": {"min": min_volume, "max": max_volume, **_percentiles(sample_volumes)},
        "tet_edge_length": _percentiles(edge_lengths.reshape(-1)),
        "edge_ratio_min_over_max": _percentiles(min_edges / np.maximum(max_edges, 1e-30)),
        "volume_edge_quality": _percentiles(volume_edge_quality),
        "orientation_counts": dict(signs),
        "zero_volume_count": zero_volume_count,
    }


def _run_worker(args: argparse.Namespace) -> int:
    memory_bytes = int(args.worker_memory_gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    started = time.perf_counter()
    result: dict[str, Any] = {"scheme": args.scheme, "maxvolume": args.maxvolume}
    try:
        import numpy as np
        import tetgen
        import trimesh

        mesh = trimesh.load_mesh(args.mesh, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * args.scale
        faces = np.asarray(mesh.faces, dtype=np.int32)
        switches = f"pq{args.minratio:g}/{args.mindihedral:g}"
        if args.maxvolume is not None:
            switches += f"a{args.maxvolume:.17g}"
        switches += "YQ"
        nodes, elems, *_ = tetgen.TetGen(vertices, faces).tetrahedralize(switches=switches)
        result.update(
            status="ok",
            switches=switches,
            vertex_count=len(nodes),
            tet_count=len(elems),
            quality=_tet_quality(nodes, elems, args.quality_sample_size),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
    except BaseException as exc:  # noqa: BLE001
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    result["duration_sec"] = time.perf_counter() - started
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "ok" else 1


def _run_genesis_worker(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    result: dict[str, Any] = {"scheme": args.scheme, "maxvolume": args.maxvolume}
    try:
        import genesis as gs

        gs.init(backend=gs.cpu, precision="32", performance_mode=True, logging_level="warning")
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0), floor_height=-2.0),
            fem_options=gs.options.FEMOptions(floor_height=-2.0),
            show_viewer=False,
            show_FPS=False,
        )
        entity = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(args.mesh),
                visual_file=None if args.visual_mesh is None else str(args.visual_mesh),
                scale=args.scale,
                file_meshes_are_zup=args.file_meshes_are_zup,
                visual_file_meshes_are_zup=args.file_meshes_are_zup,
                tet_resolution=args.tet_resolution,
                minratio=args.minratio,
                mindihedral=int(args.mindihedral),
                maxvolume=float(args.maxvolume),
                convexify=False,
                decimate=False,
            ),
            material=gs.materials.FEM.Elastic(E=1.0e5, nu=0.35, rho=1000.0, model="stable_neohookean"),
            surface=gs.surfaces.Default(vis_mode="visual", smooth=True),
        )
        result.update(
            status="ok",
            vertex_count=int(entity.n_vertices),
            tet_count=int(entity.n_elements),
            surface_vertex_count=int(entity.n_surface_vertices),
            surface_face_count=int(entity.n_surfaces),
            surface_uv_shape=(
                None if entity.surface_visual_uvs is None else list(entity.surface_visual_uvs.shape)
            ),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
    except BaseException as exc:  # noqa: BLE001
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    result["duration_sec"] = time.perf_counter() - started
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] == "ok" else 1


def _last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _invoke_worker(
    args: argparse.Namespace,
    *,
    mesh: Path,
    scale: float,
    scheme: str,
    maxvolume: float | None,
    minratio: float = 1.1,
    mindihedral: float = 15.0,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--mesh",
        str(mesh),
        "--scale",
        str(scale),
        "--scheme",
        scheme,
        "--minratio",
        str(minratio),
        "--mindihedral",
        str(mindihedral),
        "--worker-memory-gb",
        str(args.worker_memory_gb),
        "--quality-sample-size",
        str(args.quality_sample_size),
    ]
    if maxvolume is not None:
        command.extend(("--maxvolume", str(maxvolume)))
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout_sec, check=False)
    except subprocess.TimeoutExpired:
        return {"scheme": scheme, "maxvolume": maxvolume, "status": "timeout", "duration_sec": args.timeout_sec}
    result = _last_json(completed.stdout)
    if result is None:
        result = {
            "scheme": scheme,
            "maxvolume": maxvolume,
            "status": "crashed",
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
        }
    result["returncode"] = completed.returncode
    return result


def _invoke_genesis_worker(
    args: argparse.Namespace,
    entry: dict[str, Any],
    maxvolume: float,
    *,
    minratio: float = 1.1,
    mindihedral: float = 15.0,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--genesis-worker",
        "--mesh",
        str(entry["runtime_path"]),
        "--scale",
        str(entry.get("scale", 1.0)),
        "--scheme",
        "feedback_target_genesis",
        "--maxvolume",
        str(maxvolume),
        "--tet-resolution",
        str(args.tet_resolution),
        "--worker-memory-gb",
        str(args.worker_memory_gb),
        "--minratio",
        str(minratio),
        "--mindihedral",
        str(mindihedral),
    ]
    if entry.get("visual_path"):
        command.extend(("--visual-mesh", str(entry["visual_path"])))
    if entry.get("file_meshes_are_zup"):
        command.append("--file-meshes-are-zup")
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout_sec, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "duration_sec": args.timeout_sec, "maxvolume": maxvolume}
    result = _last_json(completed.stdout)
    if result is None:
        return {
            "status": "crashed",
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-2000:],
            "maxvolume": maxvolume,
        }
    result["returncode"] = completed.returncode
    return result


def _surface_stats(entry: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import trimesh

    mesh = trimesh.load_mesh(entry["runtime_path"], force="mesh", process=False)
    scale = float(entry.get("scale", 1.0))
    vertices = np.asarray(mesh.vertices) * scale
    edges = np.asarray(mesh.edges_unique_length) * scale
    volume = abs(float(mesh.volume)) * scale**3
    area = float(mesh.area) * scale**2
    return {
        "logical_name": entry["logical_name"],
        "runtime_path": entry["runtime_path"],
        "scale": scale,
        "surface_vertices": len(mesh.vertices),
        "surface_faces": len(mesh.faces),
        "bbox_extents": np.ptp(vertices, axis=0).tolist(),
        "volume": volume,
        "surface_area": area,
        "surface_edge_length": _percentiles(edges),
        "surface_area_equivalent_edge": math.sqrt(4.0 * area / (math.sqrt(3.0) * len(mesh.faces))),
    }


def _scheme_maxvolume(stats: dict[str, Any], scheme: str, args: argparse.Namespace) -> float | None:
    if scheme in {"quality_only_no_a", "quality_only_relaxed"}:
        return None
    if scheme == "bbox_current":
        edge = min(stats["bbox_extents"]) / (args.tet_resolution + 1)
        return REGULAR_TET_VOLUME_FACTOR * edge**3
    if scheme == "bbox_resolution1":
        edge = min(stats["bbox_extents"]) / 2.0
        return REGULAR_TET_VOLUME_FACTOR * edge**3
    if scheme == "surface_edge_p50":
        return REGULAR_TET_VOLUME_FACTOR * stats["surface_edge_length"]["p50"] ** 3
    if scheme == "surface_area_equivalent":
        return REGULAR_TET_VOLUME_FACTOR * stats["surface_area_equivalent_edge"] ** 3
    if scheme == "volume_budget":
        return stats["volume"] / args.target_tets
    raise ValueError(f"Unknown scheme: {scheme}")


def _feedback_search(args: argparse.Namespace, stats: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    scheme = "feedback_target"
    if baseline.get("status") != "ok":
        return {"scheme": scheme, "status": "blocked", "reason": "quality-only baseline failed"}
    if int(baseline["tet_count"]) > args.target_tets:
        return {
            "scheme": scheme,
            "status": "unreachable",
            "reason": "quality-only boundary/quality floor exceeds target",
            "tet_count": baseline["tet_count"],
            "attempts": [],
        }

    maxvolume = stats["volume"] / args.target_tets
    attempts = []
    final = None
    for attempt in range(1, 5):
        result = _invoke_worker(
            args,
            mesh=Path(stats["runtime_path"]),
            scale=stats["scale"],
            scheme=f"{scheme}_{attempt}",
            maxvolume=maxvolume,
        )
        attempts.append(result)
        final = result
        if result.get("status") != "ok":
            break
        actual = int(result["tet_count"])
        if abs(actual - args.target_tets) / args.target_tets <= 0.10:
            break
        maxvolume *= actual / args.target_tets

    output = dict(final or {"status": "error"})
    output["scheme"] = scheme
    output["attempts"] = attempts
    output["target_tets"] = args.target_tets
    return output


def _instance_counts(case_root: Path) -> Counter[str]:
    body_text = (case_root / "src/body.py").read_text()
    word = "SIGGRAPH"
    rows = 3
    for line in body_text.splitlines():
        if line.startswith("WORD = "):
            word = str(json.loads(line.split("=", 1)[1].strip()))
        elif line.startswith("ROW_COUNT = "):
            rows = int(line.split("=", 1)[1].strip())
    return Counter({f"soft_letter_{glyph}": count * rows for glyph, count in Counter(word).items()})


def _write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for asset in report["assets"]:
        for result in asset["results"]:
            quality = result.get("quality", {})
            rows.append(
                {
                    "asset": asset["logical_name"],
                    "instances": asset["instance_count"],
                    "scheme": result["scheme"],
                    "status": result.get("status"),
                    "maxvolume": result.get("maxvolume"),
                    "vertices": result.get("vertex_count"),
                    "tets": result.get("tet_count"),
                    "scene_tets": (
                        int(result["tet_count"]) * asset["instance_count"] if result.get("tet_count") else None
                    ),
                    "duration_sec": result.get("duration_sec"),
                    "peak_rss_kib": result.get("peak_rss_kib"),
                    "quality_p01": quality.get("volume_edge_quality", {}).get("p01"),
                    "quality_p50": quality.get("volume_edge_quality", {}).get("p50"),
                    "zero_volume_count": quality.get("zero_volume_count"),
                }
            )
    with (output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    schemes = list(FIXED_SCHEMES) + ["feedback_target"]
    lines = ["# TetGen sizing comparison", "", f"Target tetrahedra per body: {report['target_tets']}", ""]
    lines.append("| Asset | Instances | " + " | ".join(schemes) + " |")
    lines.append("|---|---:|" + "---:|" * len(schemes))
    for asset in report["assets"]:
        by_scheme = {result["scheme"]: result for result in asset["results"]}
        values = []
        for scheme in schemes:
            result = by_scheme[scheme]
            values.append(f"{int(result['tet_count']):,}" if result.get("status") == "ok" else result.get("status", "?"))
        lines.append(f"| {asset['logical_name']} | {asset['instance_count']} | " + " | ".join(values) + " |")
    lines.extend(("", "## Estimated full-scene tetrahedra", "", "| Scheme | Tetrahedra |", "|---|---:|"))
    for scheme in schemes:
        total = report["scene_totals"].get(scheme)
        lines.append(f"| {scheme} | {total:,} |" if total is not None else f"| {scheme} | unavailable |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _run_controller(args: argparse.Namespace) -> int:
    if args.case_root is None:
        raise ValueError("--case-root is required")
    case_root = args.case_root.resolve()
    output_dir = (args.output_dir or case_root / "reports/tetgen_sizing_experiment").resolve()
    manifest = json.loads((case_root / "assets/asset_manifest.json").read_text())
    counts = _instance_counts(case_root)
    report: dict[str, Any] = {
        "case_root": str(case_root),
        "target_tets": args.target_tets,
        "tet_resolution": args.tet_resolution,
        "quality_default": {"minratio": 1.1, "mindihedral": 15.0, "nobisect": True},
        "assets": [],
    }

    for entry in manifest["assets"]:
        stats = _surface_stats(entry)
        print(f"[tetgen-sizing] {stats['logical_name']}: {stats['surface_faces']} surface faces", flush=True)
        results = []
        for scheme in FIXED_SCHEMES:
            result = _invoke_worker(
                args,
                mesh=Path(stats["runtime_path"]),
                scale=stats["scale"],
                scheme=scheme,
                maxvolume=_scheme_maxvolume(stats, scheme, args),
                minratio=1.2 if scheme == "quality_only_relaxed" else 1.1,
                mindihedral=10.0 if scheme == "quality_only_relaxed" else 15.0,
            )
            results.append(result)
            print(f"  {scheme}: {result.get('status')} tets={result.get('tet_count')}", flush=True)
        baseline = next(result for result in results if result["scheme"] == "quality_only_no_a")
        feedback = _feedback_search(args, stats, baseline)
        results.append(feedback)
        print(f"  feedback_target: {feedback.get('status')} tets={feedback.get('tet_count')}", flush=True)
        report["assets"].append({**stats, "instance_count": counts[stats["logical_name"]], "results": results})

    scene_totals = {}
    for scheme in (*FIXED_SCHEMES, "feedback_target"):
        selected = [next(result for result in asset["results"] if result["scheme"] == scheme) for asset in report["assets"]]
        scene_totals[scheme] = (
            sum(int(result["tet_count"]) * asset["instance_count"] for asset, result in zip(report["assets"], selected))
            if all(result.get("status") == "ok" for result in selected)
            else None
        )
    report["scene_totals"] = scene_totals
    report["completed_at_unix"] = time.time()
    _write_reports(output_dir, report)
    print(f"[tetgen-sizing] reports: {output_dir}", flush=True)
    return 0


def _validate_results(args: argparse.Namespace) -> int:
    if args.case_root is None or args.validate_results is None:
        raise ValueError("--case-root and --validate-results are required")
    case_root = args.case_root.resolve()
    report = json.loads(args.validate_results.resolve().read_text())
    manifest = json.loads((case_root / "assets/asset_manifest.json").read_text())
    entries = {entry["logical_name"]: entry for entry in manifest["assets"]}
    counts = _instance_counts(case_root)
    output_path = args.validate_results.resolve().parent / "genesis_all_schemes_validation.json"
    previous = json.loads(output_path.read_text()) if output_path.is_file() else {"assets": []}
    previous_by_asset = {
        asset["logical_name"]: {result["scheme"]: result for result in asset["results"]}
        for asset in previous["assets"]
    }
    validations = []
    for asset in report["assets"]:
        asset_results = []
        for direct in asset["results"]:
            scheme = direct["scheme"]
            existing = previous_by_asset.get(asset["logical_name"], {}).get(scheme)
            if existing and existing.get("status") == "ok":
                result = existing
            elif direct.get("status") != "ok":
                result = {"scheme": scheme, "status": "blocked", "reason": "direct TetGen run failed"}
            else:
                maxvolume = direct.get("maxvolume")
                inactive_volume_cap = maxvolume is None
                result = _invoke_genesis_worker(
                    args,
                    entries[asset["logical_name"]],
                    float(asset["volume"] if inactive_volume_cap else maxvolume),
                    minratio=1.2 if scheme == "quality_only_relaxed" else 1.1,
                    mindihedral=10.0 if scheme == "quality_only_relaxed" else 15.0,
                )
                result.update(scheme=scheme, inactive_volume_cap=inactive_volume_cap)
            asset_results.append(result)
            source = "cached" if result is existing else "ran"
            print(
                f"[genesis-sizing] {asset['logical_name']} {scheme}: "
                f"{result.get('status')} tets={result.get('tet_count')} ({source})",
                flush=True,
            )
        validations.append(
            {
                "logical_name": asset["logical_name"],
                "instance_count": counts[asset["logical_name"]],
                "results": asset_results,
            }
        )
    schemes = [result["scheme"] for result in validations[0]["results"]]
    scene_totals = {}
    for scheme in schemes:
        selected = [next(result for result in asset["results"] if result["scheme"] == scheme) for asset in validations]
        scene_totals[scheme] = (
            sum(int(result["tet_count"]) * asset["instance_count"] for asset, result in zip(validations, selected))
            if all(result.get("status") == "ok" for result in selected)
            else None
        )
    output_path.write_text(json.dumps({"assets": validations, "scene_totals": scene_totals}, indent=2) + "\n")
    print(f"[genesis-sizing] report: {output_path}", flush=True)
    return 0 if all(
        result["status"] == "ok" for asset in validations for result in asset["results"]
    ) else 1


def main() -> int:
    args = _parse_args()
    if args.worker:
        return _run_worker(args)
    if args.genesis_worker:
        return _run_genesis_worker(args)
    if args.validate_results:
        return _validate_results(args)
    return _run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
