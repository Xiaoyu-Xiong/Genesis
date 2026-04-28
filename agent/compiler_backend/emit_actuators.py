from __future__ import annotations

from typing import Callable

from ..ir_schema.program import RigidIR
from .formatting import fmt_int_tuple, fmt_scalar_or_tuple, fmt_str_tuple


def emit_actuator_setup(
    emit: Callable[[int, str], None],
    *,
    program: RigidIR,
    body_vars: dict[str, str],
) -> None:
    emit(1, "actuators = {}")
    for body in program.bodies:
        body_var = body_vars[body.name]
        emit(1, f"actuators[{body.name!r}] = {{}}")
        for actuator in body.actuators:
            if actuator.dofs_idx_local is not None:
                emit(1, f"_actuator_dofs = {fmt_int_tuple(actuator.dofs_idx_local)}")
            else:
                emit(
                    1,
                    "_actuator_dofs = _resolve_dofs_idx_local("
                    f"{body_var}, dofs_idx_local=None, joint_names={fmt_str_tuple(actuator.joint_names or ())})",
                )
            emit(1, "if _actuator_dofs is None or len(_actuator_dofs) == 0:")
            emit(2, f"raise ValueError('Actuator `{actuator.name}` on body `{body.name}` resolved an empty DoF set.')")

            if actuator.kind == "position":
                if actuator.kp is not None:
                    emit(
                        1,
                        f"{body_var}.set_dofs_kp("
                        f"_to_scalar_or_tuple({fmt_scalar_or_tuple(actuator.kp)}, "
                        f"expected_size=len(_actuator_dofs), field_name='kp'), dofs_idx_local=_actuator_dofs)",
                    )
                if actuator.kv is not None:
                    emit(
                        1,
                        f"{body_var}.set_dofs_kv("
                        f"_to_scalar_or_tuple({fmt_scalar_or_tuple(actuator.kv)}, "
                        f"expected_size=len(_actuator_dofs), field_name='kv'), dofs_idx_local=_actuator_dofs)",
                    )
            if actuator.force_range is not None:
                emit(
                    1,
                    "_force_lower = _to_scalar_or_tuple("
                    f"{fmt_scalar_or_tuple(actuator.force_range.lower)}, "
                    "expected_size=len(_actuator_dofs), field_name='force_range.lower')",
                )
                emit(
                    1,
                    "_force_upper = _to_scalar_or_tuple("
                    f"{fmt_scalar_or_tuple(actuator.force_range.upper)}, "
                    "expected_size=len(_actuator_dofs), field_name='force_range.upper')",
                )
                emit(1, f"{body_var}.set_dofs_force_range(_force_lower, _force_upper, dofs_idx_local=_actuator_dofs)")
            if actuator.armature is not None:
                emit(
                    1,
                    f"{body_var}.set_dofs_armature("
                    f"_to_scalar_or_tuple({fmt_scalar_or_tuple(actuator.armature)}, "
                    f"expected_size=len(_actuator_dofs), field_name='armature'), dofs_idx_local=_actuator_dofs)",
                )
            emit(
                1,
                f"actuators[{body.name!r}][{actuator.name!r}] = "
                f"{{'kind': {actuator.kind!r}, 'dofs_idx_local': _actuator_dofs}}",
            )
