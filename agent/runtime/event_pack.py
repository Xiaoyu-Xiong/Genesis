from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..ir_schema import RigidIR, parse_ir_payload

LLM_EVENT_PACK_VERSION = "genesis.rigid.event_pack.v2"


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _as_float_list(value: Any) -> list[float] | None:
    if isinstance(value, list | tuple):
        output: list[float] = []
        for item in value:
            if not _is_number(item):
                return None
            output.append(float(item))
        return output
    return None


def _vec_norm(vec: list[float] | None) -> float | None:
    if vec is None:
        return None
    return math.sqrt(sum(component * component for component in vec))


def _clean_none_fields(data: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _state_delta(prev_state: Mapping[str, Any], curr_state: Mapping[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for field, curr_value in curr_state.items():
        if field not in prev_state:
            continue
        prev_value = prev_state[field]

        curr_vec = _as_float_list(curr_value)
        prev_vec = _as_float_list(prev_value)
        if curr_vec is not None and prev_vec is not None and len(curr_vec) == len(prev_vec):
            delta[field] = [curr_vec[i] - prev_vec[i] for i in range(len(curr_vec))]
            continue

        if _is_number(curr_value) and _is_number(prev_value):
            delta[field] = float(curr_value) - float(prev_value)
    return delta


def _event_derived(event: Mapping[str, Any]) -> dict[str, Any]:
    state = event.get("state", {})
    if not isinstance(state, Mapping):
        state = {}

    pos = _as_float_list(state.get("pos"))
    vel = _as_float_list(state.get("vel"))
    ang = _as_float_list(state.get("ang"))
    quat = _as_float_list(state.get("quat"))
    qpos = _as_float_list(state.get("qpos"))
    dofs_position = _as_float_list(state.get("dofs_position"))
    dofs_velocity = _as_float_list(state.get("dofs_velocity"))
    bbox_min = _as_float_list(state.get("bbox_min"))
    bbox_max = _as_float_list(state.get("bbox_max"))
    bbox_size = _as_float_list(state.get("bbox_size"))
    deformation_mean = float(state["vertex_disp_mean"]) if _is_number(state.get("vertex_disp_mean")) else None
    deformation_max = float(state["vertex_disp_max"]) if _is_number(state.get("vertex_disp_max")) else None

    contacts = event.get("contacts", {})
    contact_count = None
    if isinstance(contacts, Mapping) and _is_number(contacts.get("count")):
        contact_count = int(contacts["count"])

    derived: dict[str, Any] = {
        "height_z": pos[2] if pos is not None and len(pos) >= 3 else None,
        "speed": _vec_norm(vel),
        "angular_speed": _vec_norm(ang),
        "position_norm": _vec_norm(pos),
        "dofs_position_l2_norm": _vec_norm(dofs_position),
        "dofs_velocity_l2_norm": _vec_norm(dofs_velocity),
        "contact_count": contact_count,
        "bbox_height": bbox_size[2] if bbox_size is not None and len(bbox_size) >= 3 else None,
        "deformation_mean": deformation_mean,
        "deformation_max": deformation_max,
    }
    if quat is not None and len(quat) >= 4:
        derived["quat_w"] = quat[0]
    if qpos is not None:
        derived["qpos_l2_norm"] = _vec_norm(qpos)
        derived["qpos_dim"] = len(qpos)
    if bbox_min is not None:
        derived["bbox_min_norm"] = _vec_norm(bbox_min)
    if bbox_max is not None:
        derived["bbox_max_norm"] = _vec_norm(bbox_max)
    return _clean_none_fields(derived)


def _planned_step_after_each_action(program: RigidIR) -> tuple[list[dict[str, int]], int]:
    step_cursor = 0
    timeline: list[dict[str, int]] = []

    for action in program.actions:
        step_before = step_cursor
        steps_advanced = int(getattr(action, "steps", 0)) if getattr(action, "op", None) == "step" else 0
        step_cursor += steps_advanced
        timeline.append(
            {
                "step_before": step_before,
                "step_after": step_cursor,
                "steps_advanced": steps_advanced,
            }
        )

    return timeline, step_cursor


def _build_action_trace(
    program: RigidIR,
    dt: float,
    observation_indices_by_action: dict[int, list[int]],
) -> tuple[list[dict[str, Any]], int]:
    planned_steps_by_action, planned_final_step = _planned_step_after_each_action(program)
    trace: list[dict[str, Any]] = []

    for action_index, action in enumerate(program.actions):
        action_payload = action.model_dump(mode="json")
        op = action_payload.pop("op")
        step_info = planned_steps_by_action[action_index]
        step_before = step_info["step_before"]
        step_after = step_info["step_after"]
        steps_advanced = step_info["steps_advanced"]

        action_entry = {
            "action_index": action_index,
            "op": op,
            "entity": action_payload.get("entity"),
            "step_before": step_before,
            "step_after": step_after,
            "duration_steps": steps_advanced,
            "duration_sec": steps_advanced * dt,
            "parameters": _clean_none_fields(action_payload),
            "observation_indices": observation_indices_by_action.get(action_index, []),
        }
        trace.append(action_entry)

    return trace, planned_final_step


def build_llm_event_pack(
    program_or_payload: Mapping[str, Any] | RigidIR,
    run_result: Mapping[str, Any],
) -> dict[str, Any]:
    program = parse_ir_payload(program_or_payload)
    dt = float(program.scene.sim.dt)
    raw_events = run_result.get("events", [])
    crash_any = run_result.get("crash")
    crash = dict(crash_any) if isinstance(crash_any, Mapping) else None
    final_step_raw = run_result.get("final_step", 0)
    final_step = int(final_step_raw) if _is_number(final_step_raw) else 0
    status = run_result.get("status")
    run_status = status if isinstance(status, str) and status else "ok"

    observation_timeline: list[dict[str, Any]] = []
    observation_indices_by_action: dict[int, list[int]] = {}
    observation_indices_by_tag: dict[str, list[int]] = {}
    observation_indices_by_entity: dict[str, list[int]] = {}
    observation_indices_by_entity_and_tag: dict[str, dict[str, list[int]]] = {}

    prev_observation: dict[str, Any] | None = None

    max_speed: tuple[float, int] | None = None
    max_angular_speed: tuple[float, int] | None = None
    min_height: tuple[float, int] | None = None
    max_height: tuple[float, int] | None = None
    first_contact: dict[str, Any] | None = None
    max_contact_count: tuple[int, int] | None = None

    if isinstance(raw_events, list):
        for raw_event_index, event_any in enumerate(raw_events):
            if not isinstance(event_any, Mapping):
                continue
            if event_any.get("type") != "observation":
                continue

            step_raw = event_any.get("step", 0)
            step = int(step_raw) if _is_number(step_raw) else 0
            time_sec = step * dt
            action_index_raw = event_any.get("action_index", -1)
            action_index = int(action_index_raw) if _is_number(action_index_raw) else -1

            state = event_any.get("state", {})
            if not isinstance(state, Mapping):
                state = {}

            derived = _event_derived(event_any)

            timeline_item: dict[str, Any] = {
                "observation_index": len(observation_timeline),
                "raw_event_index": raw_event_index,
                "action_index": action_index,
                "step": step,
                "time_sec": time_sec,
                "entity": event_any.get("entity"),
                "tag": event_any.get("tag"),
                "state": dict(state),
                "derived": derived,
            }

            contacts = event_any.get("contacts")
            if isinstance(contacts, Mapping):
                timeline_item["contacts"] = dict(contacts)

            if prev_observation is not None:
                prev_step_raw = prev_observation.get("step", 0)
                prev_step = int(prev_step_raw) if _is_number(prev_step_raw) else 0
                prev_state = prev_observation.get("state", {})
                if not isinstance(prev_state, Mapping):
                    prev_state = {}
                timeline_item["delta_from_previous"] = {
                    "dt_steps": step - prev_step,
                    "dt_sec": (step - prev_step) * dt,
                    "state_delta": _state_delta(prev_state, state),
                }

            observation_timeline.append(timeline_item)
            prev_observation = dict(event_any)

            if action_index >= 0:
                observation_indices_by_action.setdefault(action_index, []).append(timeline_item["observation_index"])

            entity_name = event_any.get("entity")
            if isinstance(entity_name, str) and entity_name:
                observation_indices_by_entity.setdefault(entity_name, []).append(timeline_item["observation_index"])

            tag = event_any.get("tag")
            if isinstance(tag, str) and tag:
                observation_indices_by_tag.setdefault(tag, []).append(timeline_item["observation_index"])
                if isinstance(entity_name, str) and entity_name:
                    observation_indices_by_entity_and_tag.setdefault(entity_name, {}).setdefault(tag, []).append(
                        timeline_item["observation_index"]
                    )

            speed = derived.get("speed")
            if isinstance(speed, float):
                if max_speed is None or speed > max_speed[0]:
                    max_speed = (speed, step)

            angular_speed = derived.get("angular_speed")
            if isinstance(angular_speed, float):
                if max_angular_speed is None or angular_speed > max_angular_speed[0]:
                    max_angular_speed = (angular_speed, step)

            height = derived.get("height_z")
            if isinstance(height, float):
                if min_height is None or height < min_height[0]:
                    min_height = (height, step)
                if max_height is None or height > max_height[0]:
                    max_height = (height, step)

            contact_count = derived.get("contact_count")
            if isinstance(contact_count, int):
                if max_contact_count is None or contact_count > max_contact_count[0]:
                    max_contact_count = (contact_count, step)
                if contact_count > 0 and first_contact is None:
                    first_contact = {
                        "step": step,
                        "time_sec": step * dt,
                        "observation_index": timeline_item["observation_index"],
                    }

    action_trace, planned_final_step = _build_action_trace(program, dt, observation_indices_by_action)
    final_observation = observation_timeline[-1] if observation_timeline else None
    render_follow_entity = None
    if program.scene.render is not None and program.scene.render.follow_entity is not None:
        render_follow_entity = program.scene.render.follow_entity.model_dump(mode="json")

    tags = sorted(observation_indices_by_tag.keys())
    by_tag_last_observation: dict[str, Any] = {}
    for tag in tags:
        indices = observation_indices_by_tag[tag]
        if not indices:
            continue
        by_tag_last_observation[tag] = observation_timeline[indices[-1]]

    by_entity_last_observation: dict[str, Any] = {}
    for entity_name, indices in observation_indices_by_entity.items():
        if not indices:
            continue
        by_entity_last_observation[entity_name] = observation_timeline[indices[-1]]

    entities_summary = {
        body.name: {
            "shape_kind": body.shape.kind,
            "articulated": body.shape.kind in {"mjcf", "urdf"},
            "fixed": body.fixed,
            "actuators": [actuator.model_dump(mode="json") for actuator in body.actuators],
        }
        for body in program.bodies
    }
    highlights: dict[str, Any] = {
        "observation_count": len(observation_timeline),
        "tags": tags,
        "crash": crash,
        "first_contact": first_contact,
        "max_speed": {
            "value": max_speed[0],
            "step": max_speed[1],
            "time_sec": max_speed[1] * dt,
        }
        if max_speed is not None
        else None,
        "max_angular_speed": {
            "value": max_angular_speed[0],
            "step": max_angular_speed[1],
            "time_sec": max_angular_speed[1] * dt,
        }
        if max_angular_speed is not None
        else None,
        "max_contact_count": {
            "value": max_contact_count[0],
            "step": max_contact_count[1],
            "time_sec": max_contact_count[1] * dt,
        }
        if max_contact_count is not None
        else None,
        "height_range": {
            "min": {"value": min_height[0], "step": min_height[1], "time_sec": min_height[1] * dt} if min_height else None,
            "max": {"value": max_height[0], "step": max_height[1], "time_sec": max_height[1] * dt} if max_height else None,
        },
        "final_observation": final_observation,
    }

    result = {
        "format_version": LLM_EVENT_PACK_VERSION,
        "ir_version": program.ir_version,
        "units": {
            "length": "m",
            "time": "s",
            "mass": "kg",
            "velocity": "m/s",
            "angular_velocity": "rad/s",
            "force": "N",
            "torque": "N*m",
        },
        "scene": {
            "backend": program.scene.backend,
            "show_viewer": program.scene.show_viewer,
            "dt": dt,
            "gravity": list(program.scene.sim.gravity),
            "has_ground": program.scene.add_ground,
            "render_enabled": program.scene.render is not None,
            "render_follow_entity": render_follow_entity,
        },
        "entities": entities_summary,
        "execution": {
            "status": run_status,
            "crashed": crash is not None,
            "crash": crash,
            "planned_final_step": planned_final_step,
            "actual_final_step": final_step,
            "planned_matches_actual": planned_final_step == final_step,
            "final_time_sec": final_step * dt,
            "raw_event_count": len(raw_events) if isinstance(raw_events, list) else 0,
        },
        "action_trace": action_trace,
        "observations": {
            "timeline": observation_timeline,
            "by_entity_indices": observation_indices_by_entity,
            "by_entity_tag_indices": observation_indices_by_entity_and_tag,
            "by_entity_last_observation": by_entity_last_observation,
            "by_tag_indices": observation_indices_by_tag,
            "by_tag_last_observation": by_tag_last_observation,
        },
        "highlights": highlights,
    }
    return result
