from __future__ import annotations

import os
from typing import Any

import genesis as gs

from ..ir_schema.scene import RenderIR
from .observation.contacts import (
    bbox_contact_other_entities as _bbox_contact_other_entities,
    contact_other_entities as _contact_other_entities,
    count_contacts as _count_contacts,
    entity_bbox as _entity_bbox,
    observed_contact_summary as _observed_contact_summary,
)
from .observation.deformable import (
    get_deformable_entity_com_pos,
    get_deformable_observation_state,
    get_entity_root_pos,
    get_floating_base_root_state,
    is_deformable_entity,
    is_floating_base_entity,
)


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
    return _count_contacts(contact_data, count_true=count_true, to_serializable=to_serializable)


def contact_other_entities(
    contact_data: dict[str, Any],
    *,
    target_entity_name: str,
    scene_entities: dict[str, Any],
) -> list[str]:
    return _contact_other_entities(
        contact_data,
        target_entity_name=target_entity_name,
        scene_entities=scene_entities,
        to_serializable=to_serializable,
    )


def entity_bbox(
    entity: Any,
    *,
    deformable_observation_state: dict[str, Any] | None = None,
) -> tuple[list[float], list[float]] | None:
    return _entity_bbox(
        entity,
        deformable_observation_state=deformable_observation_state,
        to_serializable=to_serializable,
    )


def bbox_contact_other_entities(
    *,
    target_entity_name: str,
    target_bbox: tuple[list[float], list[float]] | None,
    scene_entities: dict[str, Any],
) -> list[str]:
    return _bbox_contact_other_entities(
        target_entity_name=target_entity_name,
        target_bbox=target_bbox,
        scene_entities=scene_entities,
        to_serializable=to_serializable,
    )


def observed_contact_summary(
    *,
    target_entity_name: str,
    target_entity: Any,
    scene_entities: dict[str, Any],
    deformable_observation_state: dict[str, Any] | None = None,
    include_count: bool = False,
) -> dict[str, Any]:
    return _observed_contact_summary(
        target_entity_name=target_entity_name,
        target_entity=target_entity,
        scene_entities=scene_entities,
        deformable_observation_state=deformable_observation_state,
        include_count=include_count,
        to_serializable=to_serializable,
    )


class FollowRootPositionProxy:
    def __init__(self, entity: Any) -> None:
        self._entity = entity

    def get_pos(self, envs_idx: Any = None) -> Any:
        return get_entity_root_pos(self._entity, envs_idx=envs_idx)


class FollowDeformablePositionProxy:
    def __init__(self, entity: Any) -> None:
        self._entity = entity

    def get_pos(self, envs_idx: Any = None) -> Any:
        pos = get_deformable_entity_com_pos(self._entity, to_serializable=to_serializable)
        if pos is None:
            raise ValueError("Deformable follow target does not have a valid COM position.")
        pos_tensor = gs.torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)
        if envs_idx is None or envs_idx == ():
            return pos_tensor
        try:
            n_envs = len(envs_idx)
        except TypeError:
            return pos_tensor
        return pos_tensor.reshape((1, 3)).expand((n_envs, 3))


def get_follow_target_entity(entity: Any) -> Any:
    if is_floating_base_entity(entity):
        return FollowRootPositionProxy(entity)
    if is_deformable_entity(entity):
        return FollowDeformablePositionProxy(entity)
    return entity


def capture_entity_snapshot(entity: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    deformable_state = get_deformable_observation_state(entity, to_serializable=to_serializable)
    floating_base_root_state = get_floating_base_root_state(entity, to_serializable=to_serializable)
    getter_by_field = {
        "pos": "get_pos",
        "quat": "get_quat",
        "vel": "get_vel",
        "ang": "get_ang",
        "qpos": "get_qpos",
        "dofs_position": "get_dofs_position",
        "dofs_velocity": "get_dofs_velocity",
    }

    for field, getter_name in getter_by_field.items():
        try:
            if deformable_state is not None and field in deformable_state:
                snapshot[field] = deformable_state[field]
            elif floating_base_root_state is not None and field in floating_base_root_state:
                snapshot[field] = floating_base_root_state[field]
            else:
                getter = getattr(entity, getter_name, None)
                if getter is None:
                    continue
                snapshot[field] = to_serializable(getter())
        except Exception:  # noqa: BLE001
            continue
    if deformable_state is not None:
        for field in ("bbox_min", "bbox_max", "bbox_size", "vertex_disp_mean", "vertex_disp_max"):
            if field in deformable_state:
                snapshot[field] = deformable_state[field]

    try:
        scene_entities = {
            other_entity.name: other_entity
            for other_entity in getattr(getattr(entity, "scene", None), "entities", [])
            if getattr(other_entity, "name", None) is not None
        }
    except Exception:  # noqa: BLE001
        scene_entities = {getattr(entity, "name", "<unknown_entity>"): entity}
    snapshot["contacts"] = observed_contact_summary(
        target_entity_name=getattr(entity, "name", "<unknown_entity>"),
        target_entity=entity,
        scene_entities=scene_entities,
        deformable_observation_state=deformable_state,
        include_count=True,
    )

    return snapshot


def finalize_recording(camera: Any, render: RenderIR) -> None:
    output_dir = os.path.dirname(render.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    camera.stop_recording(save_to_filename=render.output_video, fps=render.fps)

__all__ = [
    "FollowDeformablePositionProxy",
    "FollowRootPositionProxy",
    "bbox_contact_other_entities",
    "capture_entity_snapshot",
    "contact_other_entities",
    "count_contacts",
    "count_true",
    "entity_bbox",
    "finalize_recording",
    "get_deformable_entity_com_pos",
    "get_deformable_observation_state",
    "get_entity_root_pos",
    "get_floating_base_root_state",
    "get_follow_target_entity",
    "is_deformable_entity",
    "is_floating_base_entity",
    "observed_contact_summary",
    "to_serializable",
]
