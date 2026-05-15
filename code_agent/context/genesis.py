from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from shutil import copyfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


USER_GUIDE_INDEX_URL = "https://genesis-world.readthedocs.io/en/latest/user_guide/index.html"
API_REFERENCE_INDEX_URL = "https://genesis-world.readthedocs.io/en/latest/api_reference/index.html"
OUT_OF_SCOPE_PATTERN = re.compile(r"\b(?:MPM|PBD|SPH|Drone|Terrain|Hybrid|Emitter)\b", re.IGNORECASE)


@dataclass(slots=True, frozen=True)
class OfficialDoc:
    title: str
    url: str
    scopes: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class GenesisContextPack:
    markdown_path: Path
    json_path: Path
    docs_dir: Path
    catalog_path: Path
    markdown: str


OFFICIAL_DOCS: tuple[OfficialDoc, ...] = (
    OfficialDoc(
        "Hello Genesis",
        "https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html",
        ("planner", "scene", "body"),
    ),
    OfficialDoc(
        "Visualization and Rendering",
        "https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/visualization.html",
        ("rendering", "critic"),
    ),
    OfficialDoc(
        "Control Your Robot",
        "https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/control_your_robot.html",
        ("action", "articulated"),
    ),
    OfficialDoc(
        "Inverse Kinematics and Motion Planning",
        "https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/"
        "inverse_kinematics_motion_planning.html",
        ("action", "articulated"),
    ),
    OfficialDoc(
        "Constraints",
        "https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/constraints.html",
        ("action", "physics_runtime"),
    ),
    OfficialDoc(
        "Surfaces and Textures",
        "https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/surfaces_textures.html",
        ("rendering", "mesh", "texture", "body"),
    ),
    OfficialDoc(
        "Conventions",
        "https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/conventions.html",
        ("planner", "scene", "body", "mesh"),
    ),
    OfficialDoc(
        "Rigid Collision Detection",
        "https://genesis-world.readthedocs.io/en/latest/user_guide/advanced_topics/collision_contacts_forces.html",
        ("rigid", "action", "critic"),
    ),
    OfficialDoc(
        "IPC Coupler",
        "https://genesis-world.readthedocs.io/en/latest/user_guide/advanced_topics/ipc_coupler.html",
        ("fem", "ipc", "coupling"),
    ),
    OfficialDoc(
        "Rigid Collision Resolution",
        "https://genesis-world.readthedocs.io/en/latest/user_guide/advanced_topics/rigid_constraint_model.html",
        ("rigid", "physics_runtime"),
    ),
    OfficialDoc(
        "Mesh Processing",
        "https://genesis-world.readthedocs.io/en/latest/user_guide/advanced_topics/mesh_processing.html",
        ("mesh", "fem", "texture"),
    ),
    OfficialDoc(
        "Scene",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/scene/scene.html",
        ("scene", "planner"),
    ),
    OfficialDoc(
        "Simulator",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/scene/simulator.html",
        ("physics_runtime", "suite_ops"),
    ),
    OfficialDoc(
        "RigidEntity",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/entity/rigid_entity/rigid_entity.html",
        ("rigid", "body", "action"),
    ),
    OfficialDoc(
        "FEMEntity",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/entity/fem_entity.html",
        ("fem", "body"),
    ),
    OfficialDoc(
        "Camera",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/visualization/cameras/camera.html",
        ("rendering", "critic"),
    ),
    OfficialDoc(
        "Lights",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/visualization/lights.html",
        ("rendering",),
    ),
    OfficialDoc(
        "RigidSolver",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/engine/solvers/rigid_solver.html",
        ("rigid", "action", "physics_runtime"),
    ),
    OfficialDoc(
        "FEMSolver",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/engine/solvers/fem_solver.html",
        ("fem", "physics_runtime"),
    ),
    OfficialDoc(
        "IPCCoupler",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/engine/couplers/ipc_coupler.html",
        ("fem", "ipc", "coupling"),
    ),
    OfficialDoc(
        "Rigid Material",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/material/rigid.html",
        ("rigid", "materials", "body"),
    ),
    OfficialDoc(
        "FEM Elastic Material",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/material/fem/elastic.html",
        ("fem", "materials"),
    ),
    OfficialDoc(
        "SimOptions",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/options/"
        "simulator_coupler_and_solver_options/sim_options.html",
        ("scene", "physics_runtime"),
    ),
    OfficialDoc(
        "RigidOptions",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/options/"
        "simulator_coupler_and_solver_options/rigid_options.html",
        ("rigid", "physics_runtime"),
    ),
    OfficialDoc(
        "FEMOptions",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/options/"
        "simulator_coupler_and_solver_options/fem_options.html",
        ("fem",),
    ),
    OfficialDoc(
        "Mesh Morph",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/options/morph/file_morph/mesh.html",
        ("mesh", "body", "fem", "rigid"),
    ),
    OfficialDoc(
        "URDF Morph",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/options/morph/file_morph/urdf.html",
        ("articulated", "body"),
    ),
    OfficialDoc(
        "MJCF Morph",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/options/morph/file_morph/mjcf.html",
        ("articulated", "body"),
    ),
    OfficialDoc(
        "Rasterizer",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/options/renderer/rasterizer.html",
        ("rendering",),
    ),
    OfficialDoc(
        "RayTracer",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/options/renderer/raytracer.html",
        ("rendering",),
    ),
    OfficialDoc(
        "Surface",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/options/surface/surface.html",
        ("rendering", "materials", "texture"),
    ),
    OfficialDoc(
        "ImageTexture",
        "https://genesis-world.readthedocs.io/en/latest/api_reference/options/texture/image_texture.html",
        ("rendering", "texture", "mesh"),
    ),
)


LOCAL_CONTEXT_ANCHORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("genesis/engine/entities/rigid_entity/rigid_entity.py", ("local RigidEntity implementation",)),
    ("genesis/engine/entities/fem_entity.py", ("local FEMEntity implementation",)),
    ("genesis/engine/materials/rigid.py", ("rigid IPC coupling material behavior")),
    ("genesis/options/solvers.py", ("IPC coupler options and solver defaults")),
    ("genesis/utils/mesh.py", ("local mesh processing behavior",)),
    ("genesis/utils/element.py", ("local mesh/element conversion behavior",)),
    ("examples/rigid", ("rigid examples",)),
    ("examples/coupling/fem_cube_linked_with_arm.py", ("FEM coupled with articulated rigid example")),
    ("examples/coupling/grasp_soft_cube.py", ("FEM grasp/coupling example")),
    ("examples/coupling/cut_dragon.py", ("mesh FEM interaction example")),
    ("examples/IPC_Solver", ("IPC examples",)),
    ("examples/fem_hard_and_soft_constraint.py", ("FEM constraint example")),
    ("examples/elastic_dragon.py", ("FEM mesh example")),
)


def build_genesis_context_pack(root_dir: Path, *, refresh: bool = False) -> GenesisContextPack:
    """Build or refresh a suite-level Genesis context pack from official docs and local scope anchors."""

    context_dir = root_dir / "context" / "genesis"
    docs_dir = context_dir / "official_docs"
    context_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    documents = [_ensure_document(doc, docs_dir, refresh=refresh) for doc in OFFICIAL_DOCS]
    catalog = _selected_catalog()
    catalog_path = context_dir / "official_catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = {
        "schema_version": 1,
        "generated_at_unix": time.time(),
        "repo_git_sha": _repo_git_sha(),
        "official_indexes": [USER_GUIDE_INDEX_URL, API_REFERENCE_INDEX_URL],
        "documents": documents,
        "catalog_path": str(catalog_path),
        "docs_dir": str(docs_dir),
        "local_context_anchors": [
            {"path": path, "topics": list(topics)}
            for path, topics in LOCAL_CONTEXT_ANCHORS
        ],
        "scope_policy": _scope_policy(),
    }
    json_path = context_dir / "genesis_context.json"
    markdown = _render_markdown(payload)
    markdown_path = context_dir / "genesis_context.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return GenesisContextPack(
        markdown_path=markdown_path,
        json_path=json_path,
        docs_dir=docs_dir,
        catalog_path=catalog_path,
        markdown=markdown,
    )


def install_genesis_context_pack(case_dir: Path, pack: GenesisContextPack) -> None:
    contracts_dir = case_dir / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    copyfile(pack.markdown_path, contracts_dir / "genesis_context.md")
    copyfile(pack.json_path, contracts_dir / "genesis_context.json")


def _ensure_document(doc: OfficialDoc, docs_dir: Path, *, refresh: bool) -> dict[str, Any]:
    slug = _slugify(doc.title)
    text_path = docs_dir / f"{slug}.txt"
    meta_path = docs_dir / f"{slug}.json"
    if text_path.exists() and meta_path.exists() and not refresh:
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        metadata.update({"path": str(text_path), "cached": True})
        return metadata

    source_url = None
    status = "fetched"
    error = None
    try:
        source_url, text = _fetch_doc_text(doc.url)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        text = f"Fetch failed for {doc.url}: {type(exc).__name__}: {exc}\n"
        status = "fetch_failed"
        error = f"{type(exc).__name__}: {exc}"

    text = _sanitize_doc_text(text)
    text_path.write_text(text, encoding="utf-8")
    metadata = {
        "title": doc.title,
        "url": doc.url,
        "source_url": source_url,
        "scopes": list(doc.scopes),
        "path": str(text_path),
        "status": status,
        "error": error,
        "cached": False,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _fetch_doc_text(url: str) -> tuple[str, str]:
    errors: list[str] = []
    for candidate in _source_url_candidates(url):
        try:
            text = _read_url(candidate)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
        if text.strip():
            return candidate, text
    html = _read_url(url)
    text = _html_to_text(html)
    if not text.strip():
        raise ValueError(f"empty document text; source attempts: {'; '.join(errors)}")
    return url, text


def _source_url_candidates(url: str) -> list[str]:
    parsed = urlparse(url)
    if not parsed.path.endswith(".html"):
        return []
    marker = "/en/latest/"
    if marker not in parsed.path:
        return []
    rel = parsed.path.split(marker, 1)[1][:-5]
    base = f"{parsed.scheme}://{parsed.netloc}/en/latest/_sources/{rel}"
    return [f"{base}.md.txt", f"{base}.rst.txt"]


def _selected_catalog() -> dict[str, Any]:
    return {
        "indexes": [USER_GUIDE_INDEX_URL, API_REFERENCE_INDEX_URL],
        "scope": "selected FEM+IPC, rigid interaction, articulated interaction, mesh/texture, and rendering docs only",
        "links": [
            {"title": doc.title, "url": doc.url, "scopes": list(doc.scopes)}
            for doc in OFFICIAL_DOCS
        ],
        "status": "ok",
        "errors": [],
    }


def _read_url(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Genesis-Code-Agent/1.0"})
    with urlopen(request, timeout=20.0) as response:
        return response.read().decode("utf-8", errors="replace")


def _html_to_text(html: str) -> str:
    parser = _TextParser()
    parser.feed(html)
    return parser.text()


def _sanitize_doc_text(text: str) -> str:
    lines = [
        line
        for line in text.splitlines()
        if OUT_OF_SCOPE_PATTERN.search(line) is None
    ]
    return "\n".join(lines).strip() + "\n"


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Genesis Context Pack",
        "",
        "This pack is generated once per suite and copied into each case contract directory.",
        "Use it as the stable first stop for Genesis rigid, IPC, FEM, asset, rendering, and evaluation context.",
        "",
        "## Source Priority",
        "",
        "1. Prefer local repository source and examples when they conflict with online documentation.",
        "2. Use official Genesis documentation for API intent, parameter names, and IPC/FEM behavior.",
        "3. Ignore legacy `agent/` code unless the user explicitly asks about it.",
        "",
        "## Required Scope",
        "",
    ]
    lines.extend(f"- {item}" for item in _scope_policy())
    lines.extend(
        [
            "",
            "## Role Guidance",
            "",
            "- Planner: use this pack to design complete rigid, rigid+IPC, or FEM+IPC plans.",
            "- Scene and Body writers: consult scene, morph, material, solver, mesh, FEM, IPC, and articulated docs.",
            "- Action writer: consult control, solver, force, actuator, IPC, and contact documentation.",
            "- Rendering writer: consult camera, renderer, light, surface, texture, and visual evidence documentation.",
            "- Critic: compare prompt, generated source, artifacts, visual evidence, official docs, and local source.",
            "",
            "## Cached Official Documentation",
            "",
            "The files below contain the fetched full text for selected Genesis documentation pages.",
            "Read the cached text file that matches your role before guessing API behavior.",
            "",
        ]
    )
    for doc in payload["documents"]:
        scopes = ", ".join(doc["scopes"])
        lines.extend(
            [
                f"### {doc['title']}",
                "",
                f"- Scopes: {scopes}",
                f"- Official URL: {doc['url']}",
                f"- Fetched source: {doc.get('source_url') or doc['url']}",
                f"- Cached text: {doc['path']}",
                f"- Status: {doc['status']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Full Official Catalog",
            "",
            f"- Discovered catalog JSON: {payload['catalog_path']}",
            "- The catalog includes only selected current-pipeline Genesis documentation links.",
            "",
            "## Local Repository Anchors",
            "",
        ]
    )
    for anchor in payload["local_context_anchors"]:
        topics = ", ".join(anchor["topics"])
        lines.append(f"- `{anchor['path']}`: {topics}")
    lines.extend(
        [
            "",
            "## Generated Metadata",
            "",
            f"- Repo git SHA: {payload['repo_git_sha']}",
            f"- Official indexes: {', '.join(payload['official_indexes'])}",
            f"- Docs directory: {payload['docs_dir']}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _scope_policy() -> list[str]:
    return [
        "FEM deformable simulation with IPC coupling is the only non-rigid target family.",
        "Rigid and articulated scenes may run without IPC or with IPC contact/coupling.",
        "Articulated MJCF/URDF robots are in scope as controlled mechanisms and IPC contact participants.",
        "Generated mesh lifecycle, Meshy assets, repair, manifold readiness, texture transfer, and FEM meshes.",
        "Rendering, cameras, lights, surfaces, textures, recorders, visual evidence, and texture/orientation review.",
        "Generator, critic, optimization feedback, suite operations, artifact layout, and reproducible GPU execution.",
    ]


def _repo_git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = completed.stdout.strip()
    return sha or None


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug or "document"


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg"}:
            self._skip_depth += 1
            return
        if tag in {"h1", "h2", "h3", "h4", "p", "li", "pre", "dt", "dd", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in {"h1", "h2", "h3", "h4", "p", "li", "pre", "dt", "dd", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text)
            self.parts.append(" ")

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip() + "\n"
