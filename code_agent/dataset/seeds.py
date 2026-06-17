from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_agent.dataset.models import BuildConfig, SimilaritySeed
from code_agent.dataset.utils import short_hash, slugify


def collect_similarity_seeds(config: BuildConfig, manifest: dict[str, Any]) -> list[SimilaritySeed]:
    """Collect semantic targets from explicit inputs and previously accepted/refined prompts."""

    seeds: list[SimilaritySeed] = []
    for index, text in enumerate(config.similar_to, start=1):
        seed = _seed_from_text(text, source="cli", fallback_id=f"cli_{index:03d}")
        if seed is not None:
            seeds.append(seed)
    if config.similar_to_file is not None:
        seeds.extend(_seeds_from_file(config.similar_to_file))
    seeds.extend(_seeds_from_manifest(manifest))
    return _dedupe_and_limit(seeds, limit=max(0, config.similarity_seed_limit))


def _seeds_from_file(path: Path) -> list[SimilaritySeed]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return list(_seeds_from_json(data, source=str(path)))
    seeds = []
    for index, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        seed = _seed_from_text(line, source=str(path), fallback_id=f"{path.stem}_{index:03d}")
        if seed is not None:
            seeds.append(seed)
    return seeds


def _seeds_from_manifest(manifest: dict[str, Any]) -> list[SimilaritySeed]:
    seeds = []
    for item in manifest.get("style_memory", []):
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            continue
        case_id = _optional_str(item.get("case_id") or item.get("clip_id"))
        seeds.append(
            SimilaritySeed(
                seed_id=f"style_{slugify(case_id or prompt[:32], fallback='style')}",
                case_id=case_id,
                prompt=prompt,
                source="manifest.style_memory",
                notes=_optional_str(item.get("reason")),
            )
        )
    for clip in manifest.get("clips", []):
        if not isinstance(clip, dict) or clip.get("status") != "accepted":
            continue
        prompt = str(clip.get("prompt") or "").strip()
        if not prompt:
            continue
        case_id = _optional_str(clip.get("case_id") or clip.get("id"))
        seeds.append(
            SimilaritySeed(
                seed_id=f"accepted_{slugify(case_id or prompt[:32], fallback='accepted')}",
                case_id=case_id,
                prompt=prompt,
                source="manifest.accepted_clip",
                notes=_optional_str(clip.get("visual_summary")),
            )
        )
    return seeds


def _seeds_from_json(data: Any, *, source: str) -> list[SimilaritySeed]:
    seeds: list[SimilaritySeed] = []
    if isinstance(data, dict):
        if isinstance(data.get("clips"), list):
            seeds.extend(_seeds_from_manifest(data))
        prompt = _optional_str(data.get("prompt") or data.get("original_prompt") or data.get("task"))
        if prompt:
            case_id = _optional_str(data.get("case_id") or data.get("id") or data.get("case"))
            seeds.append(_make_seed(prompt=prompt, case_id=case_id, source=source))
        for value in data.values():
            if isinstance(value, (dict, list)):
                seeds.extend(_seeds_from_json(value, source=source))
    elif isinstance(data, list):
        for value in data:
            seeds.extend(_seeds_from_json(value, source=source))
    return seeds


def _seed_from_text(text: str, *, source: str, fallback_id: str) -> SimilaritySeed | None:
    text = text.strip()
    if not text:
        return None
    case_id = None
    prompt = text
    if "|" in text:
        raw_case_id, raw_prompt = text.split("|", 1)
        case_id = slugify(raw_case_id, fallback=fallback_id)
        prompt = raw_prompt.strip()
    if not prompt:
        return None
    return _make_seed(prompt=prompt, case_id=case_id, source=source, fallback_id=fallback_id)


def _make_seed(
    *,
    prompt: str,
    case_id: str | None,
    source: str,
    fallback_id: str | None = None,
    notes: str | None = None,
) -> SimilaritySeed:
    base = case_id or fallback_id or prompt[:48]
    return SimilaritySeed(
        seed_id=f"{slugify(base, fallback='seed')}_{short_hash(prompt, length=8)}",
        case_id=case_id,
        prompt=prompt,
        source=source,
        notes=notes,
    )


def _dedupe_and_limit(seeds: list[SimilaritySeed], *, limit: int) -> list[SimilaritySeed]:
    if limit == 0:
        return []
    seen: set[str] = set()
    deduped: list[SimilaritySeed] = []
    # Later seeds are more likely to include fresh human edits, so keep the tail first.
    for seed in reversed(seeds):
        key = seed.prompt.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(seed)
        if len(deduped) >= limit:
            break
    return list(reversed(deduped))


def _optional_str(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
