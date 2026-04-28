from __future__ import annotations

from collections import Counter
import json
from typing import Any

from ..mesh.workflow.summary import estimate_scaled_bbox_size, load_mesh_asset_summary
from ..tool_library.capabilities import (
    build_compact_generator_tool_context,
    build_generator_tool_context,
)


def load_optional_texts_by_body(
    paths_by_body: dict[str, Any] | None,
    *,
    max_chars: int = 50_000,
) -> dict[str, dict[str, Any]]:
    if not paths_by_body:
        return {}

    loaded: dict[str, dict[str, Any]] = {}
    for body_name, path in sorted(paths_by_body.items()):
        if not isinstance(body_name, str) or not body_name:
            continue
        if path is None:
            continue
        if not path.exists():
            raise ValueError(f"XML file not found for body `{body_name}`: {path}")
        text = path.read_text(encoding="utf-8")
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]
        loaded[body_name] = {
            "provided": True,
            "path": str(path),
            "text": text,
            "truncated": truncated,
        }
    return loaded


def extract_first_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty LLM response.")
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM response does not contain a valid JSON object.")
    snippet = stripped[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from LLM response: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response root is not an object.")
    return parsed


def ensure_sectioned_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    by_section = analysis.get("by_section")
    if not isinstance(by_section, dict):
        raise ValueError("Critic output missing `by_section` object.")
    for key in ("scene", "actions"):
        if key not in by_section or not isinstance(by_section.get(key), dict):
            raise ValueError(f"Critic output missing `by_section.{key}` object.")
    by_body = analysis.get("by_body")
    if not isinstance(by_body, dict):
        raise ValueError("Critic output missing `by_body` object.")
    return analysis


def build_input_digest(
    *,
    task: str,
    ir: dict[str, Any],
    event_pack: dict[str, Any],
    xml_infos_by_body: dict[str, dict[str, Any]],
    video_duration_sec: float | None,
    sample_every_sec: float,
    max_frames: int,
) -> dict[str, Any]:
    metrics = _build_event_supporting_metrics(event_pack)

    return {
        "task_prompt": task,
        "generator_tool_context": build_generator_tool_context(
            xml_generation_enabled=True,
        ),
        "video_meta": {
            "video_path": str(event_pack.get("render", {}).get("video_path", "")),
            "video_duration_sec": video_duration_sec,
            "sample_every_sec": sample_every_sec,
            "max_frames": max_frames,
        },
        "xml_meta_by_body": {
            body_name: {
                "provided": xml_info["provided"],
                "path": xml_info["path"],
                "truncated": xml_info["truncated"],
            }
            for body_name, xml_info in xml_infos_by_body.items()
        },
        "supporting_metrics": {
            "sim_duration_sec": metrics["sim_duration_sec"],
            "observation_count": metrics["observation_count"],
            "estimated_displacement_by_entity_m": metrics["estimated_displacement_by_entity_m"],
        },
        "ir_digest": _build_ir_digest(ir),
        "event_pack": event_pack,
    }


def build_compact_input_digest(
    *,
    task: str,
    ir: dict[str, Any],
    event_pack: dict[str, Any],
    xml_infos_by_body: dict[str, dict[str, Any]],
    video_duration_sec: float | None,
    sample_every_sec: float,
    max_frames: int,
) -> dict[str, Any]:
    metrics = _build_event_supporting_metrics(event_pack)

    return {
        "task_prompt": task,
        "generator_tool_context": build_compact_generator_tool_context(
            xml_generation_enabled=True,
        ),
        "video_meta": {
            "video_duration_sec": video_duration_sec,
            "sample_every_sec": sample_every_sec,
            "max_frames": max_frames,
        },
        "xml_meta_by_body": {
            body_name: {
                "path": xml_info["path"],
                "truncated": xml_info["truncated"],
            }
            for body_name, xml_info in xml_infos_by_body.items()
        },
        "supporting_metrics": {
            "sim_duration_sec": metrics["sim_duration_sec"],
            "observation_count": metrics["observation_count"],
            "observed_entities": metrics["observed_entities"],
            "estimated_displacement_by_entity_m": metrics["estimated_displacement_by_entity_m"],
        },
        "ir_digest": _build_ir_digest(ir),
    }


def _build_event_supporting_metrics(event_pack: dict[str, Any]) -> dict[str, Any]:
    timeline = _observation_timeline(event_pack)
    observed_entities = sorted(
        {
            item.get("entity")
            for item in timeline
            if isinstance(item, dict) and isinstance(item.get("entity"), str)
        }
    )
    return {
        "sim_duration_sec": _final_time_sec(event_pack),
        "observation_count": len(timeline) if timeline else None,
        "observed_entities": observed_entities,
        "estimated_displacement_by_entity_m": _estimate_displacement_by_entity(event_pack),
    }


def _final_time_sec(event_pack: dict[str, Any]) -> float | None:
    execution = event_pack.get("execution")
    if not isinstance(execution, dict):
        return None
    value = execution.get("final_time_sec")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _observation_timeline(event_pack: dict[str, Any]) -> list[dict[str, Any]]:
    observations = event_pack.get("observations")
    if not isinstance(observations, dict):
        return []
    timeline = observations.get("timeline")
    return [item for item in timeline if isinstance(item, dict)] if isinstance(timeline, list) else []


def _estimate_displacement_by_entity(event_pack: dict[str, Any]) -> dict[str, float]:
    observations = event_pack.get("observations")
    if not isinstance(observations, dict):
        return {}
    timeline_any = observations.get("timeline")
    if not isinstance(timeline_any, list) or len(timeline_any) < 2:
        return {}

    first_by_entity: dict[str, dict[str, Any]] = {}
    last_by_entity: dict[str, dict[str, Any]] = {}
    for item in timeline_any:
        if not isinstance(item, dict):
            continue
        entity = item.get("entity")
        if not isinstance(entity, str) or not entity:
            continue
        first_by_entity.setdefault(entity, item)
        last_by_entity[entity] = item

    displacement_by_entity: dict[str, float] = {}
    for entity, first in first_by_entity.items():
        last = last_by_entity.get(entity)
        if not isinstance(last, dict):
            continue
        first_state = first.get("state")
        last_state = last.get("state")
        if not isinstance(first_state, dict) or not isinstance(last_state, dict):
            continue
        first_pos = first_state.get("pos")
        last_pos = last_state.get("pos")
        if not isinstance(first_pos, list) or not isinstance(last_pos, list):
            continue
        if len(first_pos) < 3 or len(last_pos) < 3:
            continue
        try:
            dx = float(last_pos[0]) - float(first_pos[0])
            dy = float(last_pos[1]) - float(first_pos[1])
            dz = float(last_pos[2]) - float(first_pos[2])
        except Exception:  # noqa: BLE001
            continue
        displacement_by_entity[entity] = float((dx * dx + dy * dy + dz * dz) ** 0.5)
    return displacement_by_entity


def _build_ir_digest(ir: dict[str, Any]) -> dict[str, Any]:
    scene = ir.get("scene", {})
    bodies_any = ir.get("bodies", [])
    bodies = [body for body in bodies_any if isinstance(body, dict)] if isinstance(bodies_any, list) else []
    actions_any = ir.get("actions", [])
    actions = [action for action in actions_any if isinstance(action, dict)] if isinstance(actions_any, list) else []
    articulated_body_names = [body.get("name") for body in bodies if _is_articulated_body(body)]

    op_counter = Counter()
    total_step_count = 0
    for action in actions:
        op = action.get("op")
        if isinstance(op, str):
            op_counter[op] += 1
        if action.get("op") == "step":
            steps = action.get("steps")
            if isinstance(steps, int | float) and not isinstance(steps, bool):
                total_step_count += int(steps)

    return {
        "scene": scene,
        "bodies": bodies,
        "body_count": len(bodies),
        "mesh_body_summaries": _build_mesh_body_summaries(bodies),
        "articulated_body_names": articulated_body_names,
        "articulated_body_count": len(articulated_body_names),
        "actions_summary": {
            "count": len(actions),
            "op_counts": dict(sorted(op_counter.items())),
            "total_step_count": total_step_count,
        },
        "raw_ir": ir,
    }


def _build_mesh_body_summaries(bodies: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for body in bodies:
        body_name = body.get("name")
        shape = body.get("shape")
        if not isinstance(body_name, str) or not body_name:
            continue
        if not isinstance(shape, dict) or shape.get("kind") != "mesh":
            continue
        mesh_path = shape.get("file")
        if not isinstance(mesh_path, str) or not mesh_path.strip():
            continue
        summary = load_mesh_asset_summary(mesh_path)
        applied_scale = shape.get("scale")
        if isinstance(applied_scale, int | float) and not isinstance(applied_scale, bool):
            summary["applied_scale"] = float(applied_scale)
            summary["estimated_bbox_size_after_scale"] = estimate_scaled_bbox_size(
                summary.get("bbox_size"),
                float(applied_scale),
            )
        summaries[body_name] = summary
    return summaries


def _is_articulated_body(body: dict[str, Any]) -> bool:
    shape = body.get("shape")
    return isinstance(shape, dict) and shape.get("kind") in {"mjcf", "urdf"}
