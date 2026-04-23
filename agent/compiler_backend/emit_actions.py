from __future__ import annotations

from typing import Callable

from ..ir_schema import (
    ApplyExternalWrenchActionIR,
    ObserveActionIR,
    RenderIR,
    SetDofsPositionActionIR,
    SetDofsVelocityActionIR,
    SetPoseActionIR,
    SetTargetPosActionIR,
    SetTorqueActionIR,
    RigidIR,
    StepActionIR,
)
from .formatting import fmt_int_tuple, fmt_str_tuple, fmt_tuple, fmt_vec3


def emit_action_loop(
    emit: Callable[[int, str], None],
    *,
    program: RigidIR,
    render: RenderIR | None,
    entity_vars: dict[str, str],
) -> None:
    def iter_entity_names(entity: str | tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(entity, str):
            return (entity,)
        return entity

    emit(1, "events = []")
    emit(1, "sim_step = 0")
    emit(1, "rendered_frames = 0")
    emit(1, "_contact_entities = {}")
    for entity_name, entity_var in entity_vars.items():
        emit(1, f"_contact_entities[{entity_name!r}] = {entity_var}")
    if render is not None:
        emit(1, "camera.start_recording()")
        if render.include_initial_frame:
            emit(1, f"camera.render(force_render={render.force_render})")
            emit(1, "rendered_frames += 1")
    emit(1)

    for action_index, action in enumerate(program.actions):
        if isinstance(action, StepActionIR):
            emit(1, f"# action {action_index}: step")
            emit(1, f"for _ in range({action.steps}):")
            emit(2, "scene.step()")
            emit(2, "sim_step += 1")
            if render is not None:
                emit(2, f"if sim_step % {render.render_every_n_steps} == 0:")
                emit(3, f"camera.render(force_render={render.force_render})")
                emit(3, "rendered_frames += 1")
            emit(1)
            continue

        if isinstance(action, SetPoseActionIR):
            emit(1, f"# action {action_index}: set_pose ({action.entity})")
            for entity_name in iter_entity_names(action.entity):
                entity_var = entity_vars[entity_name]
                if action.pos is not None:
                    emit(
                        1,
                        f"{entity_var}.set_pos("
                        f"{fmt_tuple(action.pos)}, zero_velocity={action.zero_velocity}, relative={action.relative})",
                    )
                if action.quat is not None:
                    emit(
                        1,
                        f"{entity_var}.set_quat("
                        f"{fmt_tuple(action.quat)}, zero_velocity={action.zero_velocity}, relative={action.relative})",
                    )
            emit(1)
            continue

        if isinstance(action, SetDofsPositionActionIR):
            entity_var = entity_vars[action.entity]
            emit(1, f"# action {action_index}: set_dofs_position ({action.entity})")
            if action.dofs_idx_local is not None:
                emit(1, f"_dofs_idx_local = {fmt_int_tuple(action.dofs_idx_local)}")
                emit(1, "_joint_names = None")
            elif action.joint_names is not None:
                emit(1, "_dofs_idx_local = None")
                emit(1, f"_joint_names = {fmt_str_tuple(action.joint_names)}")
            else:
                emit(1, "_dofs_idx_local = None")
                emit(1, "_joint_names = None")
            emit(
                1,
                f"_resolved = _resolve_dofs_idx_local("
                f"{entity_var}, dofs_idx_local=_dofs_idx_local, joint_names=_joint_names, values_length={len(action.values)})",
            )
            emit(
                1,
                f"{entity_var}.set_dofs_position("
                f"{fmt_tuple(action.values)}, dofs_idx_local=_resolved, zero_velocity={action.zero_velocity})",
            )
            emit(1)
            continue

        if isinstance(action, SetDofsVelocityActionIR):
            entity_var = entity_vars[action.entity]
            emit(1, f"# action {action_index}: set_dofs_velocity ({action.entity})")
            if action.dofs_idx_local is not None:
                emit(1, f"_dofs_idx_local = {fmt_int_tuple(action.dofs_idx_local)}")
                emit(1, "_joint_names = None")
            elif action.joint_names is not None:
                emit(1, "_dofs_idx_local = None")
                emit(1, f"_joint_names = {fmt_str_tuple(action.joint_names)}")
            else:
                emit(1, "_dofs_idx_local = None")
                emit(1, "_joint_names = None")
            emit(
                1,
                f"_resolved = _resolve_dofs_idx_local("
                f"{entity_var}, dofs_idx_local=_dofs_idx_local, joint_names=_joint_names, values_length={len(action.values)})",
            )
            emit(1, f"{entity_var}.set_dofs_velocity({fmt_tuple(action.values)}, dofs_idx_local=_resolved)")
            emit(1)
            continue

        if isinstance(action, ApplyExternalWrenchActionIR):
            emit(1, f"# action {action_index}: apply_external_wrench ({action.entity})")
            if action.links_idx_local is not None:
                emit(1, f"_links_idx_local = {fmt_int_tuple(action.links_idx_local)}")
                emit(1, "_link_names = None")
            elif action.link_names is not None:
                emit(1, "_links_idx_local = None")
                emit(1, f"_link_names = {fmt_str_tuple(action.link_names)}")
            else:
                emit(1, "_links_idx_local = None")
                emit(1, "_link_names = None")
            for entity_name in iter_entity_names(action.entity):
                entity_var = entity_vars[entity_name]
                emit(
                    1,
                    f"_links_idx = _resolve_links_idx("
                    f"{entity_var}, links_idx_local=_links_idx_local, link_names=_link_names)",
                )
                if action.force is not None:
                    emit(
                        1,
                        f"{entity_var}._solver.apply_links_external_force("
                        f"{fmt_vec3(action.force)}, links_idx=_links_idx, ref={action.ref!r}, local={action.local})",
                    )
                if action.torque is not None:
                    emit(
                        1,
                        f"{entity_var}._solver.apply_links_external_torque("
                        f"{fmt_vec3(action.torque)}, links_idx=_links_idx, ref={action.ref!r}, local={action.local})",
                    )
            emit(1)
            continue

        if isinstance(action, SetTargetPosActionIR):
            entity_var = entity_vars[action.entity]
            emit(1, f"# action {action_index}: set_target_pos ({action.entity}:{action.actuator})")
            emit(1, f"_actuator = actuators[{action.entity!r}][{action.actuator!r}]")
            emit(1, "_actuator_kind = _actuator['kind']")
            emit(1, "_actuator_dofs = _actuator['dofs_idx_local']")
            emit(1, "if _actuator_kind != 'position':")
            emit(
                2,
                f"raise ValueError('Action `set_target_pos` requires position actuator, "
                f"but `{action.actuator}` is ' + _actuator_kind + '.')",
            )
            emit(1, f"if len(_actuator_dofs) != {len(action.values)}:")
            emit(
                2,
                f"raise ValueError('Actuator `{action.actuator}` expects ' + str(len(_actuator_dofs)) "
                f"+ ' values, but got {len(action.values)}.')",
            )
            emit(1, f"{entity_var}.control_dofs_position({fmt_tuple(action.values)}, dofs_idx_local=_actuator_dofs)")
            emit(1)
            continue

        if isinstance(action, SetTorqueActionIR):
            entity_var = entity_vars[action.entity]
            emit(1, f"# action {action_index}: set_torque ({action.entity}:{action.actuator})")
            emit(1, f"_actuator = actuators[{action.entity!r}][{action.actuator!r}]")
            emit(1, "_actuator_kind = _actuator['kind']")
            emit(1, "_actuator_dofs = _actuator['dofs_idx_local']")
            emit(1, "if _actuator_kind != 'motor':")
            emit(
                2,
                f"raise ValueError('Action `set_torque` requires motor actuator, "
                f"but `{action.actuator}` is ' + _actuator_kind + '.')",
            )
            emit(1, f"if len(_actuator_dofs) != {len(action.values)}:")
            emit(
                2,
                f"raise ValueError('Actuator `{action.actuator}` expects ' + str(len(_actuator_dofs)) "
                f"+ ' values, but got {len(action.values)}.')",
            )
            emit(1, f"{entity_var}.control_dofs_force({fmt_tuple(action.values)}, dofs_idx_local=_actuator_dofs)")
            emit(1)
            continue

        if isinstance(action, ObserveActionIR):
            emit(1, f"# action {action_index}: observe ({action.entity})")
            getter_by_field = {
                "pos": "get_pos",
                "quat": "get_quat",
                "vel": "get_vel",
                "ang": "get_ang",
                "qpos": "get_qpos",
                "dofs_position": "get_dofs_position",
                "dofs_velocity": "get_dofs_velocity",
            }
            for entity_name in iter_entity_names(action.entity):
                entity_var = entity_vars[entity_name]
                emit(1, "_state = {}")
                emit(1, f"_deformable_observation_state = _get_deformable_observation_state({entity_var})")
                emit(1, f"_floating_base_root_state = _get_floating_base_root_state({entity_var})")
                for field in action.fields:
                    if field in {"bbox_min", "bbox_max", "bbox_size", "vertex_disp_mean", "vertex_disp_max"}:
                        emit(1, f"if _deformable_observation_state is not None and {field!r} in _deformable_observation_state:")
                        emit(2, f"_state[{field!r}] = _deformable_observation_state[{field!r}]")
                        emit(1, "else:")
                        emit(2, f"raise ValueError('Field `{field}` is not available for entity `{entity_name}`.')")
                    else:
                        getter = getter_by_field[field]
                        if field in {"pos", "vel"}:
                            emit(1, f"if _deformable_observation_state is not None and {field!r} in _deformable_observation_state:")
                            emit(2, f"_state[{field!r}] = _deformable_observation_state[{field!r}]")
                            emit(1, "elif _floating_base_root_state is not None and "
                                 f"{field!r} in _floating_base_root_state:")
                            emit(2, f"_state[{field!r}] = _floating_base_root_state[{field!r}]")
                            emit(1, "else:")
                            emit(2, f"_state[{field!r}] = _to_serializable({entity_var}.{getter}())")
                        elif field in {"quat", "ang"}:
                            emit(1, f"if _floating_base_root_state is not None and {field!r} in _floating_base_root_state:")
                            emit(2, f"_state[{field!r}] = _floating_base_root_state[{field!r}]")
                            emit(1, "else:")
                            emit(2, f"_state[{field!r}] = _to_serializable({entity_var}.{getter}())")
                        else:
                            emit(1, f"_state[{field!r}] = _to_serializable({entity_var}.{getter}())")

                emit(1, "_event = {")
                emit(2, "'type': 'observation',")
                emit(2, f"'action_index': {action_index},")
                emit(2, "'step': sim_step,")
                emit(2, f"'entity': {entity_name!r},")
                emit(2, f"'tag': {action.tag!r},")
                emit(2, "'state': _state,")
                emit(1, "}")
                emit(1, "_event['contacts'] = _observed_contact_summary(")
                emit(2, f"target_entity_name={entity_name!r},")
                emit(2, f"target_entity={entity_var},")
                emit(2, "scene_entities=_contact_entities,")
                emit(2, "deformable_observation_state=_deformable_observation_state,")
                emit(2, f"include_count={action.include_contacts!r},")
                emit(1, ")")
                emit(1, "events.append(_event)")
            emit(1)
            continue

        raise TypeError(f"Unsupported action IR: {type(action).__name__}")
