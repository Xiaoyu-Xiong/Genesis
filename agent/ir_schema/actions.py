from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .common import QuatWXYZ, StrictModel, Vec3, dedupe_non_empty_names, validate_non_negative_indices


EntitySelector = str | tuple[str, ...]


def _normalize_entity_selector(value: EntitySelector, *, field_name: str = "entity") -> EntitySelector:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError(f"`{field_name}` cannot be empty.")
        return normalized
    return dedupe_non_empty_names(value, field_name=field_name)


class StepActionIR(StrictModel):
    op: Literal["step"] = "step"
    steps: int = Field(ge=1, le=100_000)


class SetPoseActionIR(StrictModel):
    op: Literal["set_pose"] = "set_pose"
    entity: EntitySelector = Field(
        min_length=1,
        description=(
            "Target body name or list of body names. The same pose update is broadcast to all selected bodies. "
            "Prefer the list form when the same update should be applied to multiple bodies in one step."
        ),
    )
    pos: Vec3 | None = None
    quat: QuatWXYZ | None = None
    zero_velocity: bool = True
    relative: bool = False

    @field_validator("entity")
    @classmethod
    def _check_entity(cls, value: EntitySelector) -> EntitySelector:
        return _normalize_entity_selector(value)

    @model_validator(mode="after")
    def _check_pose_input(self) -> "SetPoseActionIR":
        if self.pos is None and self.quat is None:
            raise ValueError("`set_pose` requires at least one of `pos` or `quat`.")
        return self


class SetDofsPositionActionIR(StrictModel):
    op: Literal["set_dofs_position"] = "set_dofs_position"
    entity: str = Field(min_length=1)
    values: tuple[float, ...] = Field(min_length=1)
    dofs_idx_local: tuple[int, ...] | None = None
    joint_names: tuple[str, ...] | None = None
    zero_velocity: bool = True

    @field_validator("dofs_idx_local")
    @classmethod
    def _check_dofs_idx_local(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is None:
            return value
        return validate_non_negative_indices(value, field_name="dofs_idx_local")

    @field_validator("joint_names")
    @classmethod
    def _check_joint_names(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return value
        return dedupe_non_empty_names(value, field_name="joint_names")

    @model_validator(mode="after")
    def _check_selector(self) -> "SetDofsPositionActionIR":
        if self.dofs_idx_local is not None and self.joint_names is not None:
            raise ValueError("Only one of `dofs_idx_local` or `joint_names` can be provided.")
        if self.dofs_idx_local is not None and len(self.values) != len(self.dofs_idx_local):
            raise ValueError("Length mismatch: `values` and `dofs_idx_local` must have same length.")
        return self


class SetDofsVelocityActionIR(StrictModel):
    op: Literal["set_dofs_velocity"] = "set_dofs_velocity"
    entity: str = Field(min_length=1)
    values: tuple[float, ...] = Field(min_length=1)
    dofs_idx_local: tuple[int, ...] | None = None
    joint_names: tuple[str, ...] | None = None

    @field_validator("dofs_idx_local")
    @classmethod
    def _check_dofs_idx_local(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is None:
            return value
        return validate_non_negative_indices(value, field_name="dofs_idx_local")

    @field_validator("joint_names")
    @classmethod
    def _check_joint_names(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return value
        return dedupe_non_empty_names(value, field_name="joint_names")

    @model_validator(mode="after")
    def _check_selector(self) -> "SetDofsVelocityActionIR":
        if self.dofs_idx_local is not None and self.joint_names is not None:
            raise ValueError("Only one of `dofs_idx_local` or `joint_names` can be provided.")
        if self.dofs_idx_local is not None and len(self.values) != len(self.dofs_idx_local):
            raise ValueError("Length mismatch: `values` and `dofs_idx_local` must have same length.")
        return self


class ObserveActionIR(StrictModel):
    op: Literal["observe"] = "observe"
    entity: EntitySelector = Field(
        min_length=1,
        description=(
            "Target body name or list of body names. One observation event is emitted per selected body. "
            "Prefer the list form when observing multiple bodies with the same fields and tag at the same time."
        ),
    )
    fields: tuple[Literal["pos", "quat", "vel", "ang", "qpos", "dofs_position", "dofs_velocity"], ...] = Field(
        default=("pos", "quat", "vel", "ang"),
        min_length=1,
    )
    include_contacts: bool = False
    tag: str | None = None

    @field_validator("entity")
    @classmethod
    def _check_entity(cls, value: EntitySelector) -> EntitySelector:
        return _normalize_entity_selector(value)

    @field_validator("fields")
    @classmethod
    def _dedupe_fields(
        cls, value: tuple[Literal["pos", "quat", "vel", "ang", "qpos", "dofs_position", "dofs_velocity"], ...]
    ) -> tuple[str, ...]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in value:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return tuple(deduped)


class ApplyExternalWrenchActionIR(StrictModel):
    op: Literal["apply_external_wrench"] = "apply_external_wrench"
    entity: EntitySelector = Field(
        min_length=1,
        description=(
            "Target body name or list of body names. The same external wrench update is broadcast to each selected body. "
            "Prefer the list form when the same disturbance should be applied to multiple bodies."
        ),
    )
    force: Vec3 | None = Field(
        default=None,
        description=(
            "External force applied to the selected body or links. This is an external disturbance, not an actuator "
            "command. The effect persists across subsequent simulation steps until another wrench update changes it. "
            "If the disturbance is too weak or too strong, prefer adjusting force magnitude first before changing how "
            "long the wrench is applied."
        ),
    )
    torque: Vec3 | None = Field(
        default=None,
        description=(
            "External torque applied to the selected body or links. This is an external disturbance, not an actuator "
            "command. The effect persists across subsequent simulation steps until another wrench update changes it."
        ),
    )
    links_idx_local: tuple[int, ...] | None = None
    link_names: tuple[str, ...] | None = None
    ref: Literal["link_origin", "link_com", "root_com"] = Field(
        default="link_origin",
        description=(
            "Reference point used when applying the external wrench. Changing `ref` changes the induced rotational "
            "effect even if the force vector is the same."
        ),
    )
    local: bool = Field(
        default=False,
        description=(
            "If false, force and torque are interpreted in world coordinates. If true, they are interpreted in the "
            "target link's local frame."
        ),
    )

    @field_validator("entity")
    @classmethod
    def _check_entity(cls, value: EntitySelector) -> EntitySelector:
        return _normalize_entity_selector(value)

    @field_validator("links_idx_local")
    @classmethod
    def _check_links_idx_local(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is None:
            return value
        return validate_non_negative_indices(value, field_name="links_idx_local")

    @field_validator("link_names")
    @classmethod
    def _check_link_names(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return value
        return dedupe_non_empty_names(value, field_name="link_names")

    @model_validator(mode="after")
    def _check_input(self) -> "ApplyExternalWrenchActionIR":
        if self.force is None and self.torque is None:
            raise ValueError("`apply_external_wrench` requires at least one of `force` or `torque`.")
        if self.links_idx_local is not None and self.link_names is not None:
            raise ValueError("Only one of `links_idx_local` or `link_names` can be provided.")
        if self.local and self.ref == "root_com":
            raise ValueError("`local=true` is incompatible with `ref='root_com'`.")
        return self


class SetTargetPosActionIR(StrictModel):
    op: Literal["set_target_pos"] = "set_target_pos"
    entity: str = Field(min_length=1)
    actuator: str = Field(min_length=1)
    values: tuple[float, ...] = Field(
        min_length=1,
        description=(
            "Target positions for a position actuator. These are control targets, not direct state writes. The "
            "actual response depends on actuator kp, kv, and force_range."
        ),
    )


class SetTorqueActionIR(StrictModel):
    op: Literal["set_torque"] = "set_torque"
    entity: str = Field(min_length=1)
    actuator: str = Field(min_length=1)
    values: tuple[float, ...] = Field(
        min_length=1,
        description=(
            "Direct force/torque commands for a motor actuator."
        ),
    )


ActionIR = Annotated[
    StepActionIR
    | SetPoseActionIR
    | SetDofsPositionActionIR
    | SetDofsVelocityActionIR
    | ApplyExternalWrenchActionIR
    | SetTargetPosActionIR
    | SetTorqueActionIR
    | ObserveActionIR,
    Field(discriminator="op"),
]
