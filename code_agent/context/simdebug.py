from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable
from typing import Any

from code_agent.utils.codex import DEFAULT_REPO_ROOT


SIMDEBUG_SCHEMA_VERSION = 1
SIMDEBUG_CONTEXT_DIR = Path("code_agent/context/simdebug")
SIMDEBUG_CARD_REQUIRED_FIELDS = ("id", "title", "summary", "scopes", "physics_modes")
SIMDEBUG_ALL_PHYSICS = "any"


def repo_simdebug_root(repo_root: Path | None = None) -> Path:
    root = repo_root or DEFAULT_REPO_ROOT
    return root / SIMDEBUG_CONTEXT_DIR


def repo_simdebug_library_dir(repo_root: Path | None = None) -> Path:
    return repo_simdebug_root(repo_root)


def repo_simdebug_catalog_path(repo_root: Path | None = None) -> Path:
    return repo_simdebug_root(repo_root) / "catalog.json"


def load_simdebug_cards(library_dir: Path | None = None) -> list[dict[str, Any]]:
    root = library_dir or repo_simdebug_library_dir()
    if not root.is_dir():
        return []
    cards_root = root / "cards"
    if not cards_root.is_dir():
        return []
    search_roots = [cards_root]
    paths = sorted(
        path
        for search_root in search_roots
        for suffix in ("*.json", "*.yaml", "*.yml")
        for path in search_root.rglob(suffix)
        if path.is_file()
    )
    return [_normalize_card(_load_card_file(path), path) for path in paths]


def build_simdebug_catalog(library_dir: Path | None = None) -> dict[str, Any]:
    cards = load_simdebug_cards(library_dir)
    return {
        "schema_version": SIMDEBUG_SCHEMA_VERSION,
        "library": "simdebug",
        "description": "Planner-selected human debugging experience cards for Genesis simulation generation.",
        "library_dir": _repo_display_path(library_dir or repo_simdebug_library_dir()),
        "card_count": len(cards),
        "cards": [_catalog_card_entry(card) for card in cards],
    }


def load_simdebug_catalog(
    catalog_path: Path | None = None,
    library_dir: Path | None = None,
) -> dict[str, Any]:
    path = catalog_path or repo_simdebug_catalog_path()
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return build_simdebug_catalog(library_dir)


def write_simdebug_catalog(
    output_path: Path | None = None,
    library_dir: Path | None = None,
) -> dict[str, Any]:
    catalog = build_simdebug_catalog(library_dir)
    path = output_path or repo_simdebug_catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return catalog


def infer_simdebug_physics_modes(case_state: dict[str, Any] | None) -> tuple[str, ...]:
    state = case_state or {}
    explicit = state.get("physics_modes")
    if isinstance(explicit, str) and explicit.strip():
        return (_normalize_token(explicit),)
    if isinstance(explicit, list):
        modes = tuple(_normalize_token(item) for item in explicit if isinstance(item, str) and item.strip())
        if modes:
            return modes

    capabilities = state.get("capabilities") if isinstance(state.get("capabilities"), dict) else {}
    deformable_enabled = bool(state.get("deformable_enabled") or capabilities.get("deformable_enabled"))
    ipc_enabled = bool(state.get("ipc_enabled") or capabilities.get("ipc_enabled") or deformable_enabled)
    if deformable_enabled:
        return ("fem_ipc",)
    if ipc_enabled:
        return ("rigid_ipc",)
    return ("rigid",)


def select_simdebug_cards(
    case_state: dict[str, Any] | None,
    *,
    target_role: str = "planner",
    requested_card_ids: Iterable[str] | None = None,
    catalog: dict[str, Any] | None = None,
    library_dir: Path | None = None,
) -> dict[str, Any]:
    state = case_state or {}
    role = _normalize_token(target_role or "planner")
    physics_modes = infer_simdebug_physics_modes(state)
    requested_ids = (
        tuple(_normalize_token(item) for item in requested_card_ids if str(item).strip())
        if requested_card_ids is not None
        else None
    )
    requested_id_set = set(requested_ids) if requested_ids is not None else None

    source_catalog = catalog or build_simdebug_catalog(library_dir)
    selected: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    for raw_card in source_catalog.get("cards", []):
        if not isinstance(raw_card, dict):
            continue
        card = _normalize_card(raw_card, Path(str(raw_card.get("source_path", "<catalog>"))))
        known_ids.add(card["id"])
        if requested_id_set is not None and card["id"] not in requested_id_set:
            continue
        if not _role_matches(card, role):
            continue
        if not _physics_matches(card, physics_modes):
            continue
        selected.append(
            {
                "id": card["id"],
                "title": card["title"],
                "summary": card["summary"],
                "reason": _selection_reason(card, role, physics_modes),
                "card": card,
            }
        )

    selected.sort(key=lambda item: str(item["id"]))
    selection_policy = (
        "planner_requested_ids_filtered_by_declared_role_scope_and_active_physics_mode"
        if requested_ids is not None
        else "all_role_and_physics_compatible_candidates_for_planner_relevance_judgment"
    )
    unmatched_requested_ids = (
        [
            card_id
            for card_id in requested_ids or ()
            if card_id not in {str(item["id"]) for item in selected}
        ]
        if requested_ids is not None
        else []
    )
    unknown_requested_ids = (
        [card_id for card_id in requested_ids or () if card_id not in known_ids] if requested_ids is not None else []
    )
    return {
        "schema_version": SIMDEBUG_SCHEMA_VERSION,
        "target_role": role,
        "physics_modes": list(physics_modes),
        "selection_policy": selection_policy,
        "requested_card_ids": list(requested_ids or ()),
        "unmatched_requested_card_ids": unmatched_requested_ids,
        "unknown_requested_card_ids": unknown_requested_ids,
        "selected_count": len(selected),
        "selected_cards": selected,
    }


def format_simdebug_cards_for_prompt(selection: dict[str, Any]) -> str:
    cards = selection.get("selected_cards") if isinstance(selection, dict) else []
    header = [
        "Planner-dispatched human debugging experience cards:",
        "- The Planner owns card selection and dispatch. Downstream agents should use only the cards sent to them.",
        "- Cards may be updated between turns as evidence changes; do not assume this list is fixed for the episode.",
        (
            "- Candidate policy: Python only filters by declared role scope and active physics mode. The Planner must "
            "judge task/evidence relevance and dispatch every card it considers useful; there is no fixed top-k cap."
        ),
    ]
    if not cards:
        return "\n".join(header + ["- No task-specific cards selected for this turn."])

    lines = header + [f"- Target role: {selection.get('target_role', 'planner')}."]
    lines.append(f"- Inferred physics modes: {', '.join(selection.get('physics_modes', [])) or 'unknown'}.")
    for item in cards:
        card = item.get("card") if isinstance(item, dict) else None
        if not isinstance(card, dict):
            continue
        lines.extend(_format_card_block(card, item))
    return "\n".join(lines)


def _load_card_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        payload = json.loads(text)
    else:
        payload = _load_yaml_or_json(text)
    if not isinstance(payload, dict):
        raise ValueError(f"SimDebug card must be a JSON/YAML object: {path}")
    return payload


def _load_yaml_or_json(text: str) -> Any:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return json.loads(text)
    return yaml.safe_load(text)


def _normalize_card(raw: dict[str, Any], source_path: Path) -> dict[str, Any]:
    card = dict(raw)
    card.pop("kind", None)
    for field in SIMDEBUG_CARD_REQUIRED_FIELDS:
        if field not in card:
            raise ValueError(f"SimDebug card missing required field `{field}`: {source_path}")
    card["id"] = _normalize_token(card["id"])
    card["title"] = str(card["title"]).strip()
    card["summary"] = str(card["summary"]).strip()
    card["scopes"] = _normalize_string_list(card.get("scopes"))
    card["physics_modes"] = _normalize_string_list(card.get("physics_modes"))
    card["task_tags"] = _normalize_string_list(card.get("task_tags"))
    card["failure_tags"] = _normalize_string_list(card.get("failure_tags"))
    card["source_path"] = str(card.get("source_path") or source_path)
    return card


def _catalog_card_entry(card: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "id",
        "title",
        "summary",
        "scopes",
        "physics_modes",
        "task_tags",
        "failure_tags",
        "guidance",
        "restrictions",
        "checks",
        "dispatch_hints",
        "provenance",
        "source_path",
    )
    entry = {key: card[key] for key in keys if key in card}
    if "source_path" in entry:
        entry["source_path"] = _repo_display_path(Path(str(entry["source_path"])))
    return entry


def _selection_reason(
    card: dict[str, Any],
    role: str,
    physics_modes: tuple[str, ...],
) -> str:
    return (
        f"Candidate card because its declared scope includes `{role}` and its physics modes match "
        f"{', '.join(physics_modes)}. Planner is responsible for deciding task/evidence relevance."
    )


def _role_matches(card: dict[str, Any], role: str) -> bool:
    scopes = set(card.get("scopes", ()))
    return "all" in scopes or role in scopes


def _physics_matches(card: dict[str, Any], physics_modes: tuple[str, ...]) -> bool:
    card_modes = set(card.get("physics_modes", ()))
    return SIMDEBUG_ALL_PHYSICS in card_modes or bool(card_modes.intersection(physics_modes))


def _format_card_block(card: dict[str, Any], selection_item: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"[card] {card['id']}: {card['title']}",
        f"Summary: {card['summary']}",
        f"Selection reason: {selection_item.get('reason', '<unspecified>')}",
    ]
    for field, label in (
        ("guidance", "Guidance"),
        ("restrictions", "Restrictions"),
        ("checks", "Self-checks"),
        ("dispatch_hints", "Dispatch hints"),
    ):
        if field in card:
            lines.append(f"{label}: {_compact_value(card[field])}")
    if "provenance" in card:
        lines.append(f"Provenance: {_compact_value(card['provenance'])}")
    return lines


def _compact_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(f"- {str(item).strip()}" for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _repo_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(DEFAULT_REPO_ROOT))
    except ValueError:
        return str(path)


def _normalize_string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_normalize_token(value),)
    if isinstance(value, (list, tuple, set)):
        return tuple(_normalize_token(item) for item in value if isinstance(item, str) and item.strip())
    raise ValueError(f"Expected a string list, got {type(value).__name__}")


def _normalize_token(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")
