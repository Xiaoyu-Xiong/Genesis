from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, model_validator

from .actions import (
    ActionIR,
    ApplyExternalWrenchActionIR,
    ObserveActionIR,
    SetDofsPositionActionIR,
    SetDofsVelocityActionIR,
    SetPoseActionIR,
    SetTargetPosActionIR,
    SetTorqueActionIR,
)
from .body import BodyIR, MJCFShapeIR, URDFShapeIR
from .common import IR_VERSION, StrictModel, normalize_quat
from .scene import SceneIR


def _selected_entities(entity: str | tuple[str, ...] | None) -> tuple[str, ...]:
    if entity is None:
        return ()
    if isinstance(entity, str):
        return (entity,)
    return entity


class RigidIR(StrictModel):
    ir_version: Literal[IR_VERSION] = IR_VERSION
    scene: SceneIR = Field(default_factory=SceneIR)
    bodies: list[BodyIR] = Field(min_length=1)
    actions: list[ActionIR] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_references(self) -> "RigidIR":
        body_names: set[str] = set()
        bodies_by_name: dict[str, BodyIR] = {}
        actuator_kind_by_entity: dict[str, dict[str, str]] = {}
        deformable_body_names: set[str] = set()

        for body in self.bodies:
            if body.name in body_names:
                raise ValueError(f"Duplicate body name: `{body.name}`.")
            body_names.add(body.name)
            bodies_by_name[body.name] = body
            if body.is_deformable:
                deformable_body_names.add(body.name)

            is_articulated_shape = body.is_articulated
            if len(body.actuators) > 0 and not is_articulated_shape:
                raise ValueError(f"`bodies[{body.name}].actuators` requires an articulated shape (`mjcf` or `urdf`).")

            actuator_names: set[str] = set()
            actuator_kind_by_name: dict[str, str] = {}
            for actuator in body.actuators:
                if actuator.name in actuator_names:
                    raise ValueError(f"Duplicate actuator name `{actuator.name}` within body `{body.name}`.")
                actuator_names.add(actuator.name)
                actuator_kind_by_name[actuator.name] = actuator.kind
                if actuator.joint_names is not None and not is_articulated_shape:
                    raise ValueError(
                        f"Actuator `{actuator.name}` on body `{body.name}` uses `joint_names`, but shape "
                        f"`{body.shape.kind}` does not expose named articulated joints."
                    )
            actuator_kind_by_entity[body.name] = actuator_kind_by_name

        if "ground" in body_names and self.scene.add_ground:
            raise ValueError("Body name `ground` is reserved when `scene.add_ground=true`.")

        if not self.scene.add_ground and self.scene.ground_collision is not None:
            raise ValueError("`scene.ground_collision` requires `scene.add_ground=true`.")
        allowed_entities = set(body_names)
        if self.scene.add_ground:
            allowed_entities.add("ground")
        rigid_observe_fields = {"pos", "quat", "vel", "ang", "qpos", "dofs_position", "dofs_velocity"}
        if self.scene.render is not None and self.scene.render.follow_entity is not None:
            follow_entity = self.scene.render.follow_entity.entity
            if follow_entity not in allowed_entities:
                raise ValueError(
                    "`scene.render.follow_entity.entity` references unknown entity "
                    f"`{follow_entity}`. Allowed entities: {sorted(allowed_entities)}."
                )

        current_step = 0
        for index, action in enumerate(self.actions):
            entity_selector = getattr(action, "entity", None)
            selected_entities = _selected_entities(entity_selector)
            for entity in _selected_entities(entity_selector):
                if entity not in allowed_entities:
                    raise ValueError(
                        f"Action[{index}] references unknown entity `{entity}`. "
                        f"Allowed entities: {sorted(allowed_entities)}."
                    )

            if isinstance(action, (SetPoseActionIR, SetDofsPositionActionIR, SetDofsVelocityActionIR)) and current_step > 0:
                raise ValueError(
                    f"Action[{index}] `{action.op}` is only allowed before simulation starts "
                    "(i.e. before any `step` action advances time)."
                )

            if isinstance(action, (SetDofsPositionActionIR, SetDofsVelocityActionIR)):
                target_body = bodies_by_name.get(action.entity)
                if target_body is None:
                    raise ValueError(
                        f"Action[{index}] `{action.op}` requires a non-ground body entity, got `{action.entity}`."
                    )
                if action.joint_names is not None and not isinstance(target_body.shape, (MJCFShapeIR, URDFShapeIR)):
                    raise ValueError(
                        f"Action[{index}] uses `joint_names`, but body `{target_body.name}` with shape "
                        f"`{target_body.shape.kind}` does not expose named articulated joints. Use `dofs_idx_local` instead."
                    )

            if isinstance(
                action,
                (
                    SetTargetPosActionIR,
                    SetTorqueActionIR,
                ),
            ):
                if action.entity not in bodies_by_name:
                    raise ValueError(
                        f"Action[{index}] `{action.op}` requires a non-ground body entity, got `{action.entity}`."
                    )
                entity_actuator_kinds = actuator_kind_by_entity.get(action.entity, {})
                if action.actuator not in entity_actuator_kinds:
                    raise ValueError(
                        f"Action[{index}] references unknown actuator `{action.actuator}` on entity `{action.entity}`. "
                        f"Available actuators: {sorted(entity_actuator_kinds)}."
                    )
                actuator_kind = entity_actuator_kinds[action.actuator]
                if isinstance(action, SetTargetPosActionIR) and actuator_kind != "position":
                    raise ValueError(
                        f"Action[{index}] `{action.op}` requires a `position` actuator, "
                        f"but actuator `{action.actuator}` on `{action.entity}` is `{actuator_kind}`."
                    )
                if isinstance(action, SetTorqueActionIR) and actuator_kind != "motor":
                    raise ValueError(
                        f"Action[{index}] `{action.op}` requires a `motor` actuator, "
                        f"but actuator `{action.actuator}` on `{action.entity}` is `{actuator_kind}`."
                    )
            if any(entity in deformable_body_names for entity in selected_entities):
                if isinstance(
                    action,
                    (
                        SetPoseActionIR,
                        SetDofsPositionActionIR,
                        SetDofsVelocityActionIR,
                        ApplyExternalWrenchActionIR,
                        SetTargetPosActionIR,
                        SetTorqueActionIR,
                    ),
                ):
                    raise ValueError(
                        f"Action[{index}] `{action.op}` is not supported for deformable bodies in v1."
                    )
                if isinstance(action, ObserveActionIR):
                    if action.include_contacts:
                        raise ValueError(
                            f"Action[{index}] observe on deformable bodies does not support `include_contacts=true` in v1."
                        )
                    if any(entity not in deformable_body_names for entity in selected_entities):
                        raise ValueError(
                            f"Action[{index}] observe cannot mix deformable bodies with rigid bodies in one "
                            "multi-entity observation because their supported fields differ."
                        )
                    allowed_deformable_fields = {"pos", "vel", "bbox_min", "bbox_max", "bbox_size", "vertex_disp_mean", "vertex_disp_max"}
                    invalid_fields = [field for field in action.fields if field not in allowed_deformable_fields]
                    if invalid_fields:
                        raise ValueError(
                            f"Action[{index}] observe on deformable bodies cannot use fields {invalid_fields}. "
                            f"Allowed fields: {sorted(allowed_deformable_fields)}."
                        )
            elif isinstance(action, ObserveActionIR):
                invalid_fields = [field for field in action.fields if field not in rigid_observe_fields]
                if invalid_fields:
                    raise ValueError(
                        f"Action[{index}] observe on rigid bodies cannot use fields {invalid_fields}. "
                        f"These fields are deformable-only."
                    )
            if hasattr(action, "steps"):
                current_step += int(action.steps)
        return self


def parse_ir_payload(payload: Mapping[str, Any] | RigidIR) -> RigidIR:
    if isinstance(payload, RigidIR):
        return payload
    return RigidIR.model_validate(payload)


def normalize_ir(program_or_payload: Mapping[str, Any] | RigidIR) -> RigidIR:
    program = parse_ir_payload(program_or_payload).model_copy(deep=True)
    for body in program.bodies:
        body.initial_pose.quat = normalize_quat(body.initial_pose.quat)

    for action in program.actions:
        if isinstance(action, SetPoseActionIR) and action.quat is not None:
            action.quat = normalize_quat(action.quat)

    return program
