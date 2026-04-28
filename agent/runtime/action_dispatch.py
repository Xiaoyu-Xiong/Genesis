from __future__ import annotations

from typing import Any

from ..ir_schema.actions import (
    ApplyExternalWrenchActionIR,
    ObserveActionIR,
    SetDofsPositionActionIR,
    SetDofsVelocityActionIR,
    SetPoseActionIR,
    SetTargetPosActionIR,
    SetTorqueActionIR,
    StepActionIR,
)
from .helpers import (
    get_deformable_observation_state,
    get_floating_base_root_state,
    observed_contact_summary,
    to_serializable,
)
from .models import ActuatorBinding, RuntimeContext, RuntimeState
from .selectors import resolve_dofs_idx_local, resolve_links_idx


def _validate_actuator_values_length(
    action: Any,
    dofs_idx_local: tuple[int, ...],
    *,
    field_name: str,
    values: tuple[float, ...],
) -> None:
    if len(values) != len(dofs_idx_local):
        raise ValueError(
            f"Action actuator `{action.actuator}` expects {len(dofs_idx_local)} {field_name} values, "
            f"but got {len(values)}."
        )


def _require_actuator_kind(
    action: Any,
    binding: ActuatorBinding,
    *,
    expected_kind: str,
) -> None:
    if binding.kind != expected_kind:
        raise ValueError(
            f"Action `{action.op}` requires actuator kind `{expected_kind}`, "
            f"but actuator `{action.actuator}` is `{binding.kind}`."
        )


def _target_entities(runtime: RuntimeContext, entity: str | tuple[str, ...]) -> list[tuple[str, Any]]:
    if isinstance(entity, str):
        return [(entity, runtime.entities[entity])]
    return [(entity_name, runtime.entities[entity_name]) for entity_name in entity]


def dispatch_action(action_index: int, action: Any, runtime: RuntimeContext, state: RuntimeState) -> None:
    if isinstance(action, StepActionIR):
        for _ in range(action.steps):
            runtime.scene.step()
            state.sim_step += 1
            if (
                runtime.camera is not None
                and runtime.render is not None
                and (state.sim_step % runtime.render.render_every_n_steps == 0)
            ):
                runtime.camera.render(force_render=runtime.render.force_render)
                state.rendered_frames += 1
        return

    if isinstance(action, SetPoseActionIR):
        for _, entity in _target_entities(runtime, action.entity):
            if action.pos is not None:
                entity.set_pos(tuple(action.pos), zero_velocity=action.zero_velocity, relative=action.relative)
            if action.quat is not None:
                entity.set_quat(tuple(action.quat), zero_velocity=action.zero_velocity, relative=action.relative)
        return

    if isinstance(action, SetDofsPositionActionIR):
        entity = runtime.entities[action.entity]
        dofs_idx_local = resolve_dofs_idx_local(
            entity,
            dofs_idx_local=action.dofs_idx_local,
            joint_names=action.joint_names,
            values_length=len(action.values),
        )
        entity.set_dofs_position(
            tuple(action.values),
            dofs_idx_local=dofs_idx_local,
            zero_velocity=action.zero_velocity,
        )
        return

    if isinstance(action, SetDofsVelocityActionIR):
        entity = runtime.entities[action.entity]
        dofs_idx_local = resolve_dofs_idx_local(
            entity,
            dofs_idx_local=action.dofs_idx_local,
            joint_names=action.joint_names,
            values_length=len(action.values),
        )
        entity.set_dofs_velocity(tuple(action.values), dofs_idx_local=dofs_idx_local)
        return

    if isinstance(action, ApplyExternalWrenchActionIR):
        for _, target_entity in _target_entities(runtime, action.entity):
            links_idx = resolve_links_idx(
                target_entity,
                links_idx_local=action.links_idx_local,
                link_names=action.link_names,
            )
            if action.force is not None:
                target_entity._solver.apply_links_external_force(
                    tuple(action.force),
                    links_idx=links_idx,
                    ref=action.ref,
                    local=action.local,
                )
            if action.torque is not None:
                target_entity._solver.apply_links_external_torque(
                    tuple(action.torque),
                    links_idx=links_idx,
                    ref=action.ref,
                    local=action.local,
                )
        return

    if isinstance(action, SetTargetPosActionIR):
        entity = runtime.entities[action.entity]
        entity_actuators = state.actuators_by_entity[action.entity]
        binding = entity_actuators[action.actuator]
        _require_actuator_kind(action, binding, expected_kind="position")
        dofs_idx_local = binding.dofs_idx_local
        _validate_actuator_values_length(action, dofs_idx_local, field_name="command", values=action.values)
        entity.control_dofs_position(tuple(action.values), dofs_idx_local=dofs_idx_local)
        return

    if isinstance(action, SetTorqueActionIR):
        entity = runtime.entities[action.entity]
        entity_actuators = state.actuators_by_entity[action.entity]
        binding = entity_actuators[action.actuator]
        _require_actuator_kind(action, binding, expected_kind="motor")
        dofs_idx_local = binding.dofs_idx_local
        _validate_actuator_values_length(action, dofs_idx_local, field_name="command", values=action.values)
        entity.control_dofs_force(tuple(action.values), dofs_idx_local=dofs_idx_local)
        return

    if isinstance(action, ObserveActionIR):
        for entity_name, target_entity in _target_entities(runtime, action.entity):
            state_dict: dict[str, Any] = {}
            deformable_observation_state = get_deformable_observation_state(target_entity)
            floating_base_root_state = get_floating_base_root_state(target_entity)
            getter_by_field = {
                "pos": "get_pos",
                "quat": "get_quat",
                "vel": "get_vel",
                "ang": "get_ang",
                "qpos": "get_qpos",
                "dofs_position": "get_dofs_position",
                "dofs_velocity": "get_dofs_velocity",
            }
            for field in action.fields:
                if deformable_observation_state is not None and field in deformable_observation_state:
                    state_dict[field] = deformable_observation_state[field]
                elif floating_base_root_state is not None and field in floating_base_root_state:
                    state_dict[field] = floating_base_root_state[field]
                else:
                    if field not in getter_by_field:
                        raise ValueError(f"Field `{field}` is not available for entity `{entity_name}`.")
                    getter = getattr(target_entity, getter_by_field[field], None)
                    if getter is None:
                        raise ValueError(f"Field `{field}` is not available for entity `{entity_name}`.")
                    state_dict[field] = to_serializable(getter())

            event: dict[str, Any] = {
                "type": "observation",
                "action_index": action_index,
                "step": state.sim_step,
                "entity": entity_name,
                "tag": action.tag,
                "state": state_dict,
            }

            event["contacts"] = observed_contact_summary(
                target_entity_name=entity_name,
                target_entity=target_entity,
                scene_entities=runtime.entities,
                deformable_observation_state=deformable_observation_state,
                include_count=action.include_contacts,
            )

            state.events.append(event)
        return

    raise TypeError(f"Unsupported action IR: {type(action).__name__}")
