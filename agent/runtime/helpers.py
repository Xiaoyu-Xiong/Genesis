from __future__ import annotations

import os
from typing import Any

from ..ir_schema import RenderIR


def to_serializable(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def count_true(mask: Any) -> int:
    data = to_serializable(mask)
    if not isinstance(data, list):
        return int(bool(data))

    count = 0
    stack: list[Any] = [data]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(current)
        else:
            count += int(bool(current))
    return count


def count_contacts(contact_data: dict[str, Any]) -> int:
    if "valid_mask" in contact_data:
        return count_true(contact_data["valid_mask"])

    geom_a = to_serializable(contact_data.get("geom_a", []))
    if isinstance(geom_a, list):
        if geom_a and isinstance(geom_a[0], list):
            return len(geom_a[0])
        return len(geom_a)
    return 0


def _iter_contact_geom_pairs(contact_data: dict[str, Any]) -> list[tuple[int, int]]:
    geom_a = to_serializable(contact_data.get("geom_a", []))
    geom_b = to_serializable(contact_data.get("geom_b", []))
    valid_mask = to_serializable(contact_data.get("valid_mask")) if "valid_mask" in contact_data else None

    pairs: list[tuple[int, int]] = []

    def _append_pairs(a_data: Any, b_data: Any, mask_data: Any = None) -> None:
        if isinstance(a_data, list) and isinstance(b_data, list) and a_data and isinstance(a_data[0], list):
            for index in range(min(len(a_data), len(b_data))):
                env_mask = None if mask_data is None else mask_data[index]
                _append_pairs(a_data[index], b_data[index], env_mask)
            return

        if not isinstance(a_data, list) or not isinstance(b_data, list):
            return

        for index in range(min(len(a_data), len(b_data))):
            if mask_data is not None and isinstance(mask_data, list) and index < len(mask_data) and not bool(mask_data[index]):
                continue
            a_item = a_data[index]
            b_item = b_data[index]
            if not isinstance(a_item, (int, float)) or isinstance(a_item, bool):
                continue
            if not isinstance(b_item, (int, float)) or isinstance(b_item, bool):
                continue
            pairs.append((int(a_item), int(b_item)))

    _append_pairs(geom_a, geom_b, valid_mask)
    return pairs


def contact_other_entities(
    contact_data: dict[str, Any],
    *,
    target_entity_name: str,
    scene_entities: dict[str, Any],
) -> list[str]:
    target_entity = scene_entities.get(target_entity_name)
    if target_entity is None:
        return []

    try:
        target_geom_start = int(target_entity.geom_start)
        target_geom_end = int(target_entity.geom_end)
    except Exception:  # noqa: BLE001
        return []

    def _entity_name_for_geom(geom_idx: int) -> str | None:
        for entity_name, entity in scene_entities.items():
            try:
                geom_start = int(entity.geom_start)
                geom_end = int(entity.geom_end)
            except Exception:  # noqa: BLE001
                continue
            if geom_start <= geom_idx < geom_end:
                return entity_name
        return None

    other_entities: list[str] = []
    seen: set[str] = set()
    for geom_a, geom_b in _iter_contact_geom_pairs(contact_data):
        a_in_target = target_geom_start <= geom_a < target_geom_end
        b_in_target = target_geom_start <= geom_b < target_geom_end
        if a_in_target == b_in_target:
            continue
        other_geom = geom_b if a_in_target else geom_a
        other_entity_name = _entity_name_for_geom(other_geom)
        if other_entity_name is None or other_entity_name == target_entity_name or other_entity_name in seen:
            continue
        seen.add(other_entity_name)
        other_entities.append(other_entity_name)
    return other_entities


def is_floating_base_entity(entity: Any) -> bool:
    try:
        return int(entity.n_qs) >= 7 and int(entity.n_dofs) >= 6
    except Exception:  # noqa: BLE001
        return False


def _as_float_vector(value: Any) -> list[float] | None:
    data = to_serializable(value)
    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, list):
        return None
    output: list[float] = []
    for component in data:
        if not isinstance(component, int | float) or isinstance(component, bool):
            return None
        output.append(float(component))
    return output


def get_floating_base_root_state(entity: Any) -> dict[str, list[float]] | None:
    if not is_floating_base_entity(entity):
        return None

    qpos = _as_float_vector(entity.get_qpos())
    dofs_velocity = _as_float_vector(entity.get_dofs_velocity())
    if qpos is None or dofs_velocity is None:
        return None
    if len(qpos) < 7 or len(dofs_velocity) < 6:
        return None

    return {
        "pos": qpos[:3],
        "quat": qpos[3:7],
        "vel": dofs_velocity[:3],
        "ang": dofs_velocity[3:6],
    }


def get_entity_root_pos(entity: Any, envs_idx: Any = None) -> Any:
    if not is_floating_base_entity(entity):
        return entity.get_pos(envs_idx)
    qpos = entity.get_qpos(envs_idx=envs_idx)
    return qpos[..., :3]


class FollowRootPositionProxy:
    def __init__(self, entity: Any) -> None:
        self._entity = entity

    def get_pos(self, envs_idx: Any = None) -> Any:
        return get_entity_root_pos(self._entity, envs_idx=envs_idx)


def get_follow_target_entity(entity: Any) -> Any:
    if is_floating_base_entity(entity):
        return FollowRootPositionProxy(entity)
    return entity


def capture_entity_snapshot(entity: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    floating_base_root_state = get_floating_base_root_state(entity)
    getter_by_field = {
        "pos": entity.get_pos,
        "quat": entity.get_quat,
        "vel": entity.get_vel,
        "ang": entity.get_ang,
        "qpos": entity.get_qpos,
        "dofs_position": entity.get_dofs_position,
        "dofs_velocity": entity.get_dofs_velocity,
    }

    for field, getter in getter_by_field.items():
        try:
            if floating_base_root_state is not None and field in floating_base_root_state:
                snapshot[field] = floating_base_root_state[field]
            else:
                snapshot[field] = to_serializable(getter())
        except Exception:  # noqa: BLE001
            continue

    try:
        snapshot["contacts"] = {"count": count_contacts(entity.get_contacts())}
    except Exception:  # noqa: BLE001
        pass

    return snapshot


def finalize_recording(camera: Any, render: RenderIR) -> None:
    output_dir = os.path.dirname(render.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    camera.stop_recording(save_to_filename=render.output_video, fps=render.fps)
