from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from code_agent.assets.mesh.repair.sanity import run_mesh_manifold_check
from code_agent.assets.mesh.workflow.summary import load_mesh_asset_summary
from code_agent.io_utils import dump_json


_GITHUB_BLOB_RE = re.compile(r"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<ref>[^/]+)/(?P<path>.+)$")
_GITHUB_RAW_RE = re.compile(r"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/raw/(?P<ref>[^/]+)/(?P<path>.+)$")
_MESH_SUFFIXES = {".obj", ".ply", ".stl", ".glb", ".gltf", ".dae"}


def prepare_layout_reusable_assets(*, case_dir: Path, layout_path: Path) -> dict[str, Any] | None:
    """Copy/download reusable meshes declared by a layout and write a partial asset manifest.

    Layout assets are intentionally not repaired, remeshed, resized, or texture-transferred. The only geometry
    validation is the same manifold/tetgen sanity check used by generated mesh assets.
    """

    layout_payload = _load_layout_json(layout_path)
    if layout_payload is None:
        return None

    specs = _layout_asset_specs(layout_payload)
    if not specs:
        return None

    assets_dir = case_dir / "assets"
    report_path = case_dir / "reports" / "layout_asset_report.json"
    manifest_path = assets_dir / "layout_asset_manifest.json"
    entries: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for index, spec in enumerate(specs):
        result = _prepare_one_layout_asset(
            spec=spec,
            index=index,
            case_dir=case_dir,
            layout_path=layout_path,
        )
        results.append(result)
        entry = result.get("manifest_entry")
        if isinstance(entry, dict):
            entries.append(entry)

    unresolved = [
        str(entry.get("logical_name"))
        for entry in entries
        if entry.get("status") != "ready" and entry.get("logical_name")
    ]
    manifest = {
        "assets": entries,
        "assumptions": [
            "Reusable layout mesh assets are copied or downloaded verbatim from layout-declared sources.",
            "No layout mesh asset is repaired, remeshed, resized, texture-transferred, or otherwise post-processed.",
            "Validation uses one read-only manifold/tetgen sanity check per runtime mesh.",
        ],
        "unresolved_risks": unresolved,
    }
    report = {
        "ok": not unresolved,
        "status": "layout_assets_ready" if not unresolved else "layout_asset_validation_failed",
        "layout_path": str(layout_path),
        "asset_manifest_path": str(manifest_path),
        "asset_generation_report_path": str(report_path),
        "num_assets": len(entries),
        "assets": results,
    }
    dump_json(manifest, manifest_path)
    dump_json(report, report_path)
    return report


def _prepare_one_layout_asset(
    *,
    spec: dict[str, Any],
    index: int,
    case_dir: Path,
    layout_path: Path,
) -> dict[str, Any]:
    logical_name = _safe_name(str(spec.get("logical_name") or spec.get("name") or f"layout_asset_{index:02d}"))
    asset_dir = case_dir / "assets" / "layout" / logical_name
    asset_dir.mkdir(parents=True, exist_ok=True)
    source_records: list[dict[str, Any]] = []
    texture_paths: dict[str, str] = {}
    notes = [
        "Reusable mesh asset declared by the layout file.",
        "Source file was copied/downloaded verbatim; no repair or post-processing was applied.",
    ]

    try:
        mesh_ref = spec.get("mesh") or spec.get("runtime_path") or spec.get("source")
        if mesh_ref is None:
            raise ValueError("Reusable layout asset is missing `mesh`, `runtime_path`, or `source`.")
        runtime_path, mesh_source = _materialize_ref(
            mesh_ref,
            asset_dir=asset_dir,
            layout_path=layout_path,
            default_repo=spec.get("repo") or spec.get("repository"),
            default_ref=spec.get("ref") or spec.get("revision"),
            purpose="mesh",
        )
        source_records.append(mesh_source)

        visual_path = runtime_path
        visual_ref = spec.get("visual") or spec.get("visual_path") or spec.get("visual_mesh")
        if visual_ref is not None:
            visual_path, visual_source = _materialize_ref(
                visual_ref,
                asset_dir=asset_dir,
                layout_path=layout_path,
                default_repo=spec.get("repo") or spec.get("repository"),
                default_ref=spec.get("ref") or spec.get("revision"),
                purpose="visual",
            )
            source_records.append(visual_source)

        for texture_name, texture_ref in _texture_refs(spec):
            texture_path, texture_source = _materialize_ref(
                texture_ref,
                asset_dir=asset_dir,
                layout_path=layout_path,
                default_repo=spec.get("repo") or spec.get("repository"),
                default_ref=spec.get("ref") or spec.get("revision"),
                purpose=f"texture_{texture_name}",
            )
            texture_paths[texture_name] = str(texture_path.resolve())
            source_records.append(texture_source)

        summary = load_mesh_asset_summary(runtime_path)
        bbox = _vector3(spec.get("bbox")) or _vector3(summary.get("bbox_size"))
        scale = _vector3(spec.get("scale"))
        file_meshes_are_zup = spec.get("file_meshes_are_zup")
        if not isinstance(file_meshes_are_zup, bool):
            file_meshes_are_zup = None
        manifold = run_mesh_manifold_check(runtime_path)
        texture_error = _texture_error(texture_paths)
        status = "ready" if _mesh_file_is_loadable(manifold.to_dict()) and texture_error is None else "failed"
        if texture_paths:
            notes.append("Texture/material files were copied/downloaded verbatim from layout-declared sources.")
        if texture_error:
            notes.append(texture_error)
        if manifold.error:
            notes.append(f"Read-only mesh sanity check reported a warning: {manifold.error}")

        entry = {
            "logical_name": logical_name,
            "source_type": "repo_asset",
            "runtime_path": str(runtime_path.resolve()),
            "visual_path": str(visual_path.resolve()) if visual_path is not None else None,
            "scale": scale,
            "bbox": bbox,
            "file_meshes_are_zup": file_meshes_are_zup,
            "texture_path": next(iter(texture_paths.values()), None),
            "texture_paths": texture_paths,
            "validation": {
                "manifold": manifold.to_dict(),
                "source_asset": {
                    "layout_path": str(layout_path),
                    "source_records": source_records,
                    "mesh_summary": summary,
                    "post_processing": "none",
                },
            },
            "simulation_role": str(spec.get("simulation_role") or spec.get("role") or "layout reusable mesh asset"),
            "status": status,
            "notes": notes,
        }
        return {
            "ok": status == "ready",
            "status": status,
            "request": spec,
            "manifest_entry": entry,
        }
    except Exception as exc:  # noqa: BLE001 - keep one bad layout asset from hiding the rest.
        error = f"{type(exc).__name__}: {exc}"
        entry = {
            "logical_name": logical_name,
            "source_type": "repo_asset",
            "runtime_path": "unavailable",
            "visual_path": None,
            "scale": _vector3(spec.get("scale")),
            "bbox": _vector3(spec.get("bbox")),
            "file_meshes_are_zup": (
                spec.get("file_meshes_are_zup") if isinstance(spec.get("file_meshes_are_zup"), bool) else None
            ),
            "texture_path": None,
            "texture_paths": {},
            "validation": {
                "source_asset": {
                    "layout_path": str(layout_path),
                    "post_processing": "none",
                },
                "error": error,
            },
            "simulation_role": str(spec.get("simulation_role") or spec.get("role") or "layout reusable mesh asset"),
            "status": "failed",
            "notes": notes + [error],
        }
        return {
            "ok": False,
            "status": "failed",
            "request": spec,
            "manifest_entry": entry,
            "error": error,
        }


def _load_layout_json(layout_path: Path) -> dict[str, Any] | None:
    if layout_path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _layout_asset_specs(layout_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = (
        layout_payload.get("reusable_assets")
        or layout_payload.get("layout_assets")
        or layout_payload.get("repo_assets")
    )
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _texture_refs(spec: dict[str, Any]) -> list[tuple[str, Any]]:
    refs: list[tuple[str, Any]] = []
    for key in ("material", "mtl"):
        if spec.get(key) is not None:
            refs.append((key, spec[key]))
    texture = spec.get("texture") or spec.get("texture_path")
    if texture is not None:
        refs.append(("primary", texture))
    textures = spec.get("textures") or spec.get("texture_paths")
    if isinstance(textures, dict):
        refs.extend((str(name), ref) for name, ref in textures.items())
    elif isinstance(textures, list):
        for idx, ref in enumerate(textures):
            refs.append((f"texture_{idx:02d}", ref))
    return refs


def _materialize_ref(
    ref: Any,
    *,
    asset_dir: Path,
    layout_path: Path,
    default_repo: Any,
    default_ref: Any,
    purpose: str,
) -> tuple[Path, dict[str, Any]]:
    resolved = _resolve_ref(ref, layout_path=layout_path, default_repo=default_repo, default_ref=default_ref)
    target = _target_path(asset_dir, resolved["filename"])
    if resolved["kind"] == "url":
        _download_file(str(resolved["uri"]), target)
    else:
        source_path = Path(str(resolved["path"]))
        if not source_path.is_file():
            raise FileNotFoundError(f"Layout asset file does not exist: {source_path}")
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
    record = {
        "purpose": purpose,
        "kind": resolved["kind"],
        "uri": resolved.get("uri"),
        "path": resolved.get("path"),
        "local_path": str(target.resolve()),
        "sha256": _sha256(target),
        "bytes": target.stat().st_size,
    }
    return target, record


def _resolve_ref(
    ref: Any,
    *,
    layout_path: Path,
    default_repo: Any,
    default_ref: Any,
) -> dict[str, Any]:
    repo = _repo_url(default_repo)
    revision = _revision(default_repo) or _revision(default_ref) or "main"
    if isinstance(ref, dict):
        repo = _repo_url(ref.get("repo") or ref.get("repository")) or repo
        revision = _revision(ref.get("ref") or ref.get("revision")) or revision
        uri = ref.get("url") or ref.get("uri")
        path = ref.get("path") or ref.get("file")
        if isinstance(uri, str) and uri.strip():
            return _resolve_string_ref(uri, layout_path=layout_path, repo=None, revision=revision)
        if isinstance(path, str) and path.strip():
            return _resolve_string_ref(path, layout_path=layout_path, repo=repo, revision=revision)
        raise ValueError(f"Invalid layout asset reference: {ref}")
    if isinstance(ref, str) and ref.strip():
        return _resolve_string_ref(ref, layout_path=layout_path, repo=repo, revision=revision)
    raise ValueError(f"Invalid layout asset reference: {ref!r}")


def _resolve_string_ref(ref: str, *, layout_path: Path, repo: str | None, revision: str) -> dict[str, Any]:
    ref = ref.strip()
    if _is_url(ref):
        uri = _normalize_url(ref)
        return {"kind": "url", "uri": uri, "filename": _filename_from_uri(uri)}
    if repo:
        uri = _github_raw_url(repo, revision, ref)
        return {"kind": "url", "uri": uri, "filename": Path(ref).name or "asset"}
    path = Path(ref)
    if not path.is_absolute():
        path = layout_path.parent / path
    return {"kind": "file", "path": str(path.resolve()), "filename": path.name}


def _repo_url(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        raw = value.get("url") or value.get("repo") or value.get("repository")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _revision(value: Any) -> str | None:
    if isinstance(value, str) and value.strip() and not value.strip().startswith("http"):
        return value.strip()
    if isinstance(value, dict):
        raw = value.get("ref") or value.get("revision") or value.get("branch") or value.get("commit")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _github_raw_url(repo: str, revision: str, path: str) -> str:
    parsed = urlparse(repo)
    if parsed.netloc not in {"github.com", "www.github.com"}:
        raise ValueError(f"Only GitHub repository shorthand is supported for relative remote paths: {repo}")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub repository URL: {repo}")
    owner, repo_name = parts[0], parts[1]
    encoded_path = quote(path.strip("/"), safe="/")
    encoded_ref = quote(revision, safe="")
    return f"https://raw.githubusercontent.com/{owner}/{repo_name}/{encoded_ref}/{encoded_path}"


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc in {"github.com", "www.github.com"}:
        raw_match = _GITHUB_RAW_RE.match(parsed.path)
        blob_match = _GITHUB_BLOB_RE.match(parsed.path)
        match = raw_match or blob_match
        if match is not None:
            groups = match.groupdict()
            encoded_path = quote(groups["path"], safe="/")
            encoded_ref = quote(groups["ref"], safe="")
            return (
                f"https://raw.githubusercontent.com/{groups['owner']}/{groups['repo']}/"
                f"{encoded_ref}/{encoded_path}"
            )
    return url


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _filename_from_uri(uri: str) -> str:
    name = Path(urlparse(uri).path).name
    return name or "asset"


def _target_path(asset_dir: Path, filename: str) -> Path:
    cleaned = _safe_filename(filename)
    target = asset_dir / cleaned
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    for idx in range(1, 1000):
        candidate = asset_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique target path for {target}")


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Genesis-code-agent-layout-assets/1.0"})
    with urlopen(request, timeout=120) as response, target.open("wb") as file:  # noqa: S310 - explicit user layout URL.
        shutil.copyfileobj(response, file)


def _texture_error(texture_paths: dict[str, str]) -> str | None:
    for name, path_text in texture_paths.items():
        path = Path(path_text)
        if not path.is_file():
            return f"Texture `{name}` was not materialized: {path}"
        if path.stat().st_size <= 0:
            return f"Texture `{name}` is empty: {path}"
    return None


def _mesh_file_is_loadable(manifold: dict[str, Any]) -> bool:
    return int(manifold.get("vertex_count") or 0) > 0 and int(manifold.get("face_count") or 0) > 0


def _vector3(value: Any) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return cleaned or "layout_asset"


def _safe_filename(value: str) -> str:
    cleaned = Path(value).name
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", cleaned).strip("._")
    if not cleaned:
        return "asset"
    if Path(cleaned).suffix.lower() not in _MESH_SUFFIXES and "." not in cleaned:
        return cleaned
    return cleaned
