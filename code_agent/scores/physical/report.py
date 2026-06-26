from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from code_agent.io_utils import load_json_object

CAP_LIMITS = {"invalid_no_run_or_no_video": 20.0, "not_real_simulation": 40.0}
CAP_LIMITS["severe_forbidden_violation"] = 60.0
CATEGORY_WEIGHTS = {"scene": 0.20, "body": 0.275, "action": 0.275, "render": 0.25}


def normalize_report(
    report: dict[str, Any],
    *,
    prompt: str,
    run_dir: Path,
    code_root: Path,
    case_id: str | None,
    evidence: dict[str, Any],
    codex_result: dict[str, Any],
) -> dict[str, Any]:
    rubric = report.get("rubric") if isinstance(report.get("rubric"), dict) else {}
    scene_score = _rubric_score(rubric.get("scene"))
    body_score = _rubric_score(rubric.get("body"))
    action_score = _rubric_score(rubric.get("action"))
    render_faithfulness = _clamp_score(report.get("render_faithfulness"))
    render_aesthetic = _clamp_score(report.get("render_aesthetic_quality"))
    render_score = _clamp_score(0.7 * render_faithfulness + 0.3 * render_aesthetic)
    violation_penalty = _forbidden_penalty(rubric.get("forbidden"))
    base_score = (
        CATEGORY_WEIGHTS["scene"] * scene_score
        + CATEGORY_WEIGHTS["body"] * body_score
        + CATEGORY_WEIGHTS["action"] * action_score
        + CATEGORY_WEIGHTS["render"] * render_score
        - violation_penalty
    )
    caps_applied = _normalized_string_list(report.get("caps_applied"))
    if _has_no_visual_evidence(evidence) and "invalid_no_run_or_no_video" not in caps_applied:
        caps_applied.append("invalid_no_run_or_no_video")
    overall_score = _apply_caps(_clamp_score(base_score), caps_applied)

    normalized = dict(report)
    normalized.update(
        {
            "metric": "SBAR-v1",
            "schema_version": 1,
            "case_id": case_id,
            "prompt": prompt,
            "run_dir": str(run_dir),
            "code_root": str(code_root),
            "overall_score": overall_score,
            "scene_score": scene_score,
            "body_score": body_score,
            "action_score": action_score,
            "render_score": render_score,
            "render_faithfulness": render_faithfulness,
            "render_aesthetic_quality": render_aesthetic,
            "violation_penalty": violation_penalty,
            "caps_applied": caps_applied,
            "fatal_violations": _normalized_string_list(report.get("fatal_violations")),
            "missing_evidence": _normalized_string_list(report.get("missing_evidence")),
            "confidence": _clamp_unit(report.get("confidence")),
            "summary": str(report.get("summary") or ""),
            "evidence_index_path": evidence.get("index_path"),
            "scorer_status": "completed" if codex_result.get("success") else "codex_exec_failed",
            "codex_result": codex_result,
            "normalization": {
                "formula": (
                    "overall = cap(0.20 * scene + 0.275 * body + 0.275 * action + "
                    "0.25 * render - violation_penalty)"
                ),
                "category_weights": CATEGORY_WEIGHTS,
                "render_formula": "render = 0.70 * render_faithfulness + 0.30 * render_aesthetic_quality",
                "category_scores_recomputed_from_rubric": True,
                "violation_penalty_recomputed_from_forbidden_rubric": True,
            },
        }
    )
    return normalized


def failed_report(
    *,
    prompt: str,
    run_dir: Path,
    code_root: Path,
    case_id: str | None,
    evidence: dict[str, Any],
    result: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "metric": "SBAR-v1",
        "schema_version": 1,
        "case_id": case_id,
        "prompt": prompt,
        "run_dir": str(run_dir),
        "code_root": str(code_root),
        "overall_score": 0.0,
        "scene_score": 0.0,
        "body_score": 0.0,
        "action_score": 0.0,
        "render_score": 0.0,
        "render_faithfulness": 0.0,
        "render_aesthetic_quality": 0.0,
        "violation_penalty": 0.0,
        "caps_applied": ["invalid_no_run_or_no_video"],
        "fatal_violations": [],
        "rubric": {"scene": [], "body": [], "action": [], "render": [], "forbidden": []},
        "category_summaries": {
            "scene": {"strengths": [], "weaknesses": [reason]},
            "body": {"strengths": [], "weaknesses": [reason]},
            "action": {"strengths": [], "weaknesses": [reason]},
            "render": {"strengths": [], "weaknesses": [reason]},
        },
        "missing_evidence": [reason],
        "confidence": 0.0,
        "summary": f"SBAR scoring failed: {reason}",
        "evidence_index_path": evidence.get("index_path"),
        "scorer_status": "failed",
        "codex_result": result,
    }


def _rubric_score(items: Any) -> float:
    if not isinstance(items, list) or not items:
        return 0.0
    total = 0.0
    weighted = 0.0
    for item in items:
        if not isinstance(item, dict):
            continue
        weight = _positive_float(item.get("weight"))
        score = _valid_item_score(item.get("score"))
        total += weight
        weighted += weight * score
    return 0.0 if total <= 0 else _clamp_score(100.0 * weighted / total)


def _forbidden_penalty(items: Any) -> float:
    if not isinstance(items, list) or not items:
        return 0.0
    total = 0.0
    violated = 0.0
    for item in items:
        if not isinstance(item, dict):
            continue
        weight = _positive_float(item.get("weight"))
        total += weight
        if bool(item.get("violation")):
            violated += weight
    return 0.0 if total <= 0 else _clamp_score(40.0 * violated / total)


def _valid_item_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number >= 0.75:
        return 1.0
    if number >= 0.25:
        return 0.5
    return 0.0


def _positive_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(number) or number <= 0:
        return 1.0
    return number


def _clamp_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(max(0.0, min(100.0, number)), 4)


def _clamp_unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(max(0.0, min(1.0, number)), 4)


def _apply_caps(score: float, caps_applied: list[str]) -> float:
    capped = score
    for label in caps_applied:
        limit = CAP_LIMITS.get(label)
        if limit is not None:
            capped = min(capped, limit)
    return _clamp_score(capped)


def _has_no_visual_evidence(evidence: dict[str, Any]) -> bool:
    return (
        not evidence.get("video_paths")
        and not evidence.get("frame_paths")
        and not evidence.get("image_paths_sent_to_scorer")
    )


def _normalized_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def file_digest(path: Path, *, max_chars: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "suffix": path.suffix,
        "size_bytes": file_size(path),
        "content": read_text(path, max_chars=max_chars),
    }


def compact_file(path: Path) -> Any:
    payload = load_json_object(path)
    if isinstance(payload, dict):
        return compact_json(payload)
    return read_text(path, max_chars=24000)


def compact_json(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"frames", "sampled_frames", "frame_summaries", "artifacts"}:
            if isinstance(value, list):
                compact[key] = {"count": len(value), "head": value[:2], "tail": value[-2:]}
            elif isinstance(value, dict):
                compact[key] = {"count": len(value), "sample_keys": list(value)[:20]}
            else:
                compact[key] = value
            continue
        compact[key] = value
    text = json.dumps(compact, ensure_ascii=False, indent=2)
    if len(text) <= 50000:
        return compact
    return {
        "truncated_digest": True,
        "top_level_keys": list(payload)[:80],
        "head_tail": clip_middle(text, 50000),
    }


def read_text(path: Path, *, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<unreadable {path}: {type(exc).__name__}: {exc}>"
    return clip_middle(text, max_chars)


def clip_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = f"\n<truncated {len(text) - max_chars} chars from middle>\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:]


def file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def unique_paths(paths: list[Path], *, limit: int) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        unique.append(resolved)
        if len(unique) >= limit:
            break
    return unique
