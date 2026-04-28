from __future__ import annotations

from typing import Any

from ..ir_schema.body import ActuatorIR
from .models import ActuatorBinding
from .selectors import resolve_dofs_idx_local, to_scalar_or_tuple


def configure_actuators(entity: Any, actuators: tuple[ActuatorIR, ...]) -> dict[str, ActuatorBinding]:
    resolved_by_name: dict[str, ActuatorBinding] = {}

    for actuator in actuators:
        dofs_idx_local = resolve_dofs_idx_local(
            entity,
            dofs_idx_local=actuator.dofs_idx_local,
            joint_names=actuator.joint_names,
        )
        if dofs_idx_local is None:
            raise ValueError(f"Actuator `{actuator.name}` failed to resolve DoF indices.")
        if len(dofs_idx_local) == 0:
            raise ValueError(f"Actuator `{actuator.name}` resolved an empty DoF set.")

        dof_count = len(dofs_idx_local)
        if actuator.kind == "position":
            if actuator.kp is not None:
                entity.set_dofs_kp(
                    to_scalar_or_tuple(actuator.kp, expected_size=dof_count, field_name="kp"),
                    dofs_idx_local=dofs_idx_local,
                )
            if actuator.kv is not None:
                entity.set_dofs_kv(
                    to_scalar_or_tuple(actuator.kv, expected_size=dof_count, field_name="kv"),
                    dofs_idx_local=dofs_idx_local,
                )
        if actuator.force_range is not None:
            lower = to_scalar_or_tuple(
                actuator.force_range.lower,
                expected_size=dof_count,
                field_name="force_range.lower",
            )
            upper = to_scalar_or_tuple(
                actuator.force_range.upper,
                expected_size=dof_count,
                field_name="force_range.upper",
            )
            entity.set_dofs_force_range(lower, upper, dofs_idx_local=dofs_idx_local)
        if actuator.armature is not None:
            entity.set_dofs_armature(
                to_scalar_or_tuple(actuator.armature, expected_size=dof_count, field_name="armature"),
                dofs_idx_local=dofs_idx_local,
            )

        resolved_by_name[actuator.name] = ActuatorBinding(
            kind=actuator.kind,
            dofs_idx_local=dofs_idx_local,
        )

    return resolved_by_name
