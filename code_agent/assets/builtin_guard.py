from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from code_agent.configs import CONFIGS

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


BUILTIN_ASSET_POLICY = """
Built-in Genesis asset policy:
- Do not inspect, copy, import, or reference prepackaged files under `genesis/assets`.
- Do not use `gs.utils.get_assets_dir()` or Genesis helper APIs to resolve built-in XML, URDF, mesh, texture, or hfield
  assets.
- Use primitives, generated XML/MJCF assets written into the case workspace, generated Meshy assets, or explicit
  user-provided layout assets copied into the case workspace. XML/MJCF mesh references are allowed only when the mesh
  file is generated into the case workspace and passes XML asset validation.
""".strip()

_BUILTIN_ASSET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"genesis[/\\]assets", re.IGNORECASE), "direct genesis/assets path"),
    (re.compile(r"get_assets_dir\s*\(", re.IGNORECASE), "Genesis get_assets_dir() asset lookup"),
    (re.compile(r"gs\.utils\.get_assets_dir", re.IGNORECASE), "Genesis gs.utils.get_assets_dir asset lookup"),
    (re.compile(r"(?:gs|genesis)\.__file__", re.IGNORECASE), "Genesis package-path asset derivation"),
    (
        re.compile(r"importlib\.resources[^\n]*(?:genesis|assets)", re.IGNORECASE),
        "Genesis package-resource asset lookup",
    ),
    (
        re.compile(
            r"(?P<quote>['\"])(?:xml|urdf|meshes)[/\\][^'\"]+\."
            r"(?:xml|mjcf|urdf|obj|stl|glb|dae|png|jpe?g)(?P=quote)",
            re.IGNORECASE,
        ),
        "relative Genesis built-in asset path",
    ),
)


def builtin_asset_denied_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for raw in CONFIGS.codex.builtin_asset_denied_roots:
        path = Path(raw)
        if not path.is_absolute():
            path = DEFAULT_REPO_ROOT / path
        roots.append(path.resolve())
    return tuple(roots)


def builtin_asset_violations(payload: Any, *, label: str) -> list[str]:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, sort_keys=True)
    violations: list[str] = []
    for pattern, description in _BUILTIN_ASSET_PATTERNS:
        for match in pattern.finditer(text):
            snippet = _snippet(text, match.start(), match.end())
            violations.append(f"{label}: {description}: {snippet}")
    return violations


def source_file_builtin_asset_violations(path: Path, *, case_dir: Path | None = None) -> list[str]:
    if not path.is_file():
        return []
    label = _relative_label(path, case_dir=case_dir)
    return builtin_asset_violations(path.read_text(encoding="utf-8", errors="replace"), label=label)


def case_source_builtin_asset_violations(case_dir: Path) -> list[str]:
    src_dir = case_dir / "src"
    if not src_dir.is_dir():
        return []
    violations: list[str] = []
    for path in sorted(src_dir.glob("*.py")):
        violations.extend(source_file_builtin_asset_violations(path, case_dir=case_dir))
    return violations


def _snippet(text: str, start: int, end: int, radius: int = 80) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right].replace("\n", "\\n")
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."
    return snippet


def _relative_label(path: Path, *, case_dir: Path | None) -> str:
    if case_dir is not None:
        try:
            return str(path.resolve().relative_to(case_dir.resolve()))
        except ValueError:
            pass
    try:
        return str(path.resolve().relative_to(DEFAULT_REPO_ROOT))
    except ValueError:
        return str(path)
