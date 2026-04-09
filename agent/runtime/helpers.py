from __future__ import annotations

import os
from typing import Any

import genesis as gs

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


def is_deformable_entity(entity: Any) -> bool:
    return getattr(entity.__class__, "__name__", "") in {
        "PBD2DEntity",
        "PBD3DEntity",
        "PBDParticleEntity",
        "PBDFreeParticleEntity",
        "FEMEntity",
    }


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


def _as_vec3_rows(value: Any) -> list[list[float]] | None:
    data = to_serializable(value)
    if (
        isinstance(data, list)
        and data
        and isinstance(data[0], list)
        and data[0]
        and isinstance(data[0][0], list)
    ):
        data = data[0]
    if not isinstance(data, list):
        return None
    rows: list[list[float]] = []
    for row in data:
        if not isinstance(row, list) or len(row) != 3:
            return None
        converted: list[float] = []
        for component in row:
            if not isinstance(component, int | float) or isinstance(component, bool):
                return None
            converted.append(float(component))
        rows.append(converted)
    return rows


def _mean_vec3(rows: list[list[float]]) -> list[float]:
    if not rows:
        return [0.0, 0.0, 0.0]
    count = float(len(rows))
    return [
        sum(row[0] for row in rows) / count,
        sum(row[1] for row in rows) / count,
        sum(row[2] for row in rows) / count,
    ]


def _bbox_from_rows(rows: list[list[float]]) -> tuple[list[float], list[float], list[float]]:
    mins = [min(row[i] for row in rows) for i in range(3)]
    maxs = [max(row[i] for row in rows) for i in range(3)]
    return mins, maxs, [maxs[i] - mins[i] for i in range(3)]


def _displacement_stats(current_rows: list[list[float]], initial_rows: list[list[float]] | None) -> tuple[float | None, float | None]:
    if initial_rows is None or len(current_rows) != len(initial_rows):
        return None, None
    magnitudes: list[float] = []
    for current, initial in zip(current_rows, initial_rows, strict=True):
        dx = current[0] - initial[0]
        dy = current[1] - initial[1]
        dz = current[2] - initial[2]
        magnitudes.append((dx * dx + dy * dy + dz * dz) ** 0.5)
    if not magnitudes:
        return 0.0, 0.0
    return sum(magnitudes) / len(magnitudes), max(magnitudes)


def get_deformable_observation_state(entity: Any) -> dict[str, Any] | None:
    if not is_deformable_entity(entity):
        return None
    try:
        if getattr(entity.__class__, "__name__", "") == "FEMEntity":
            fem_state = entity.get_state()
            pos_tensor = fem_state.pos
            vel_tensor = fem_state.vel
            initial_source = getattr(entity, "init_positions", None)
        else:
            pos_tensor = entity.get_particles_pos()
            vel_tensor = entity.get_particles_vel()
            initial_source = getattr(entity, "_particles", None)
    except Exception:  # noqa: BLE001
        return None
    pos_rows = _as_vec3_rows(to_serializable(pos_tensor))
    vel_rows = _as_vec3_rows(to_serializable(vel_tensor))
    if pos_rows is None or vel_rows is None:
        return None
    bbox_min, bbox_max, bbox_size = _bbox_from_rows(pos_rows)
    initial_rows = _as_vec3_rows(to_serializable(initial_source))
    disp_mean, disp_max = _displacement_stats(pos_rows, initial_rows)
    return {
        "pos": _mean_vec3(pos_rows),
        "vel": _mean_vec3(vel_rows),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_size": bbox_size,
        "vertex_disp_mean": disp_mean,
        "vertex_disp_max": disp_max,
    }


def get_entity_root_pos(entity: Any, envs_idx: Any = None) -> Any:
    if not is_floating_base_entity(entity):
        return entity.get_pos(envs_idx)
    qpos = entity.get_qpos(envs_idx=envs_idx)
    return qpos[..., :3]


def get_deformable_entity_com_pos(entity: Any) -> list[float] | None:
    deformable_state = get_deformable_observation_state(entity)
    if deformable_state is None:
        return None
    pos = deformable_state.get("pos")
    if isinstance(pos, list) and len(pos) == 3:
        return pos
    return None


class FollowRootPositionProxy:
    def __init__(self, entity: Any) -> None:
        self._entity = entity

    def get_pos(self, envs_idx: Any = None) -> Any:
        return get_entity_root_pos(self._entity, envs_idx=envs_idx)


class FollowDeformablePositionProxy:
    def __init__(self, entity: Any) -> None:
        self._entity = entity

    def get_pos(self, envs_idx: Any = None) -> Any:
        pos = get_deformable_entity_com_pos(self._entity)
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
    deformable_state = get_deformable_observation_state(entity)
    floating_base_root_state = get_floating_base_root_state(entity)
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
        snapshot["contacts"] = {"count": count_contacts(entity.get_contacts())}
    except Exception:  # noqa: BLE001
        pass

    return snapshot


def finalize_recording(camera: Any, render: RenderIR) -> None:
    output_dir = os.path.dirname(render.output_video)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    camera.stop_recording(save_to_filename=render.output_video, fps=render.fps)
