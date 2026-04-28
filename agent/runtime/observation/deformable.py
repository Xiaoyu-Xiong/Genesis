from __future__ import annotations

from typing import Any


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


def _as_vec3_rows(value: Any, *, to_serializable: Any) -> list[list[float]] | None:
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
            if not isinstance(component, (int, float)) or isinstance(component, bool):
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


def _displacement_stats(
    current_rows: list[list[float]],
    initial_rows: list[list[float]] | None,
) -> tuple[float | None, float | None]:
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


def get_floating_base_root_state(entity: Any, *, to_serializable: Any) -> dict[str, list[float]] | None:
    if not is_floating_base_entity(entity):
        return None

    qpos = _as_float_vector(entity.get_qpos(), to_serializable=to_serializable)
    dofs_velocity = _as_float_vector(entity.get_dofs_velocity(), to_serializable=to_serializable)
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


def get_deformable_observation_state(entity: Any, *, to_serializable: Any) -> dict[str, Any] | None:
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
    pos_rows = _as_vec3_rows(pos_tensor, to_serializable=to_serializable)
    vel_rows = _as_vec3_rows(vel_tensor, to_serializable=to_serializable)
    if pos_rows is None or vel_rows is None:
        return None
    bbox_min, bbox_max, bbox_size = _bbox_from_rows(pos_rows)
    initial_rows = _as_vec3_rows(initial_source, to_serializable=to_serializable)
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


def get_entity_root_pos(entity: Any, *, envs_idx: Any = None) -> Any:
    if not is_floating_base_entity(entity):
        return entity.get_pos(envs_idx)
    qpos = entity.get_qpos(envs_idx=envs_idx)
    return qpos[..., :3]


def get_deformable_entity_com_pos(entity: Any, *, to_serializable: Any) -> list[float] | None:
    deformable_state = get_deformable_observation_state(entity, to_serializable=to_serializable)
    if deformable_state is None:
        return None
    pos = deformable_state.get("pos")
    if isinstance(pos, list) and len(pos) == 3:
        return pos
    return None
