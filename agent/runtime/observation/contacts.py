from __future__ import annotations

from typing import Any

from .deformable import get_deformable_observation_state, is_deformable_entity


def count_contacts(contact_data: dict[str, Any], *, count_true: Any, to_serializable: Any) -> int:
    if "valid_mask" in contact_data:
        return count_true(contact_data["valid_mask"])

    geom_a = to_serializable(contact_data.get("geom_a", []))
    if isinstance(geom_a, list):
        if geom_a and isinstance(geom_a[0], list):
            return len(geom_a[0])
        return len(geom_a)
    return 0


def _iter_contact_geom_pairs(contact_data: dict[str, Any], *, to_serializable: Any) -> list[tuple[int, int]]:
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
    to_serializable: Any,
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
    for geom_a, geom_b in _iter_contact_geom_pairs(contact_data, to_serializable=to_serializable):
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


def _as_float_vector(value: Any, *, to_serializable: Any) -> list[float] | None:
    data = to_serializable(value)
    if isinstance(data, list) and data and isinstance(data[0], list):
        data = data[0]
    if not isinstance(data, list):
        return None
    output: list[float] = []
    for component in data:
        if not isinstance(component, (int, float)) or isinstance(component, bool):
            return None
        output.append(float(component))
    return output


def _first_aabb(aabb_value: Any, *, to_serializable: Any) -> tuple[list[float], list[float]] | None:
    data = to_serializable(aabb_value)
    if not isinstance(data, list):
        return None
    if len(data) == 2 and all(isinstance(item, list) for item in data):
        mins = _as_float_vector(data[0], to_serializable=to_serializable)
        maxs = _as_float_vector(data[1], to_serializable=to_serializable)
        if mins is None or maxs is None or len(mins) != 3 or len(maxs) != 3:
            return None
        return mins, maxs
    if len(data) >= 1 and isinstance(data[0], list):
        first = data[0]
        if isinstance(first, list) and len(first) == 2 and all(isinstance(item, list) for item in first):
            mins = _as_float_vector(first[0], to_serializable=to_serializable)
            maxs = _as_float_vector(first[1], to_serializable=to_serializable)
            if mins is None or maxs is None or len(mins) != 3 or len(maxs) != 3:
                return None
            return mins, maxs
    return None


def entity_bbox(
    entity: Any,
    *,
    deformable_observation_state: dict[str, Any] | None = None,
    to_serializable: Any,
) -> tuple[list[float], list[float]] | None:
    if deformable_observation_state is not None:
        bbox_min = _as_float_vector(deformable_observation_state.get("bbox_min"), to_serializable=to_serializable)
        bbox_max = _as_float_vector(deformable_observation_state.get("bbox_max"), to_serializable=to_serializable)
        if bbox_min is not None and bbox_max is not None and len(bbox_min) == 3 and len(bbox_max) == 3:
            return bbox_min, bbox_max

    if is_deformable_entity(entity):
        deformable_state = get_deformable_observation_state(entity, to_serializable=to_serializable)
        if deformable_state is None:
            return None
        bbox_min = _as_float_vector(deformable_state.get("bbox_min"), to_serializable=to_serializable)
        bbox_max = _as_float_vector(deformable_state.get("bbox_max"), to_serializable=to_serializable)
        if bbox_min is None or bbox_max is None or len(bbox_min) != 3 or len(bbox_max) != 3:
            return None
        return bbox_min, bbox_max

    get_aabb = getattr(entity, "get_AABB", None)
    if get_aabb is None:
        return None
    try:
        return _first_aabb(get_aabb(allow_fast_approx=True), to_serializable=to_serializable)
    except TypeError:
        try:
            return _first_aabb(get_aabb(), to_serializable=to_serializable)
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None


def _aabb_overlaps(
    lhs_min: list[float],
    lhs_max: list[float],
    rhs_min: list[float],
    rhs_max: list[float],
    *,
    eps: float = 1e-6,
) -> bool:
    return all(lhs_min[i] <= rhs_max[i] + eps and rhs_min[i] <= lhs_max[i] + eps for i in range(3))


def bbox_contact_other_entities(
    *,
    target_entity_name: str,
    target_bbox: tuple[list[float], list[float]] | None,
    scene_entities: dict[str, Any],
    to_serializable: Any,
) -> list[str]:
    if target_bbox is None:
        return []
    target_min, target_max = target_bbox
    overlaps: list[str] = []
    for other_entity_name, other_entity in scene_entities.items():
        if other_entity_name == target_entity_name:
            continue
        other_bbox = entity_bbox(other_entity, to_serializable=to_serializable)
        if other_bbox is None:
            continue
        other_min, other_max = other_bbox
        if _aabb_overlaps(target_min, target_max, other_min, other_max):
            overlaps.append(other_entity_name)
    return overlaps


def observed_contact_summary(
    *,
    target_entity_name: str,
    target_entity: Any,
    scene_entities: dict[str, Any],
    deformable_observation_state: dict[str, Any] | None = None,
    include_count: bool = False,
    to_serializable: Any,
) -> dict[str, Any]:
    target_bbox = entity_bbox(
        target_entity,
        deformable_observation_state=deformable_observation_state,
        to_serializable=to_serializable,
    )
    other_entities = bbox_contact_other_entities(
        target_entity_name=target_entity_name,
        target_bbox=target_bbox,
        scene_entities=scene_entities,
        to_serializable=to_serializable,
    )
    summary = {
        "other_entities": other_entities,
        "exact": False,
        "source": "aabb_overlap",
    }
    if include_count:
        summary["count"] = len(other_entities)
    return summary
