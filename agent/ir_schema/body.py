from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from .common import ScalarOrSequence, StrictModel, Vec3, dedupe_non_empty_names, length_if_sequence, validate_non_negative_indices
from .scene import CollisionIR, PoseIR


class SphereShapeIR(StrictModel):
    kind: Literal["sphere"] = "sphere"
    radius: float = Field(default=0.5, gt=0.0)


class BoxShapeIR(StrictModel):
    kind: Literal["box"] = "box"
    size: Vec3 = (0.5, 0.5, 0.5)

    @field_validator("size")
    @classmethod
    def _check_positive_size(cls, value: Vec3) -> Vec3:
        if any(component <= 0.0 for component in value):
            raise ValueError("`size` components must be > 0.")
        return value


class CylinderShapeIR(StrictModel):
    kind: Literal["cylinder"] = "cylinder"
    radius: float = Field(default=0.5, gt=0.0)
    height: float = Field(default=1.0, gt=0.0)


class MeshShapeIR(StrictModel):
    kind: Literal["mesh"] = "mesh"
    file: str = Field(min_length=1)
    scale: float = Field(default=1.0, gt=0.0)


class MJCFShapeIR(StrictModel):
    kind: Literal["mjcf"] = "mjcf"
    file: str = Field(min_length=1)
    scale: float = Field(default=1.0, gt=0.0)
    requires_jac_and_IK: bool = True
    default_armature: float | None = Field(
        default=0.1,
        gt=0.0,
        description=(
            "Additional joint armature for imported articulated models. This is mainly a dynamics-stability "
            "parameter, not a task-level motion parameter."
        ),
    )


class URDFShapeIR(StrictModel):
    kind: Literal["urdf"] = "urdf"
    file: str = Field(min_length=1)
    scale: float = Field(default=1.0, gt=0.0)
    requires_jac_and_IK: bool = True
    fixed: bool = False
    merge_fixed_links: bool = True
    default_armature: float | None = Field(
        default=0.1,
        gt=0.0,
        description=(
            "Additional joint armature for imported articulated models. This is mainly a dynamics-stability "
            "parameter, not a task-level motion parameter."
        ),
    )


ShapeIR = Annotated[
    SphereShapeIR | BoxShapeIR | CylinderShapeIR | MeshShapeIR | MJCFShapeIR | URDFShapeIR,
    Field(discriminator="kind"),
]


class PBDElasticMaterialIR(StrictModel):
    kind: Literal["elastic"] = "elastic"
    rho: float = Field(gt=0.0)
    stretch_compliance: float = Field(ge=0.0)
    volume_compliance: float = Field(ge=0.0)


DeformableMaterialIR = Annotated[
    PBDElasticMaterialIR,
    Field(discriminator="kind"),
]


class ActuatorForceRangeIR(StrictModel):
    lower: ScalarOrSequence
    upper: ScalarOrSequence

    @model_validator(mode="after")
    def _check_shape(self) -> "ActuatorForceRangeIR":
        lower_len = length_if_sequence(self.lower)
        upper_len = length_if_sequence(self.upper)
        if lower_len is not None and lower_len == 0:
            raise ValueError("`force_range.lower` sequence cannot be empty.")
        if upper_len is not None and upper_len == 0:
            raise ValueError("`force_range.upper` sequence cannot be empty.")
        if lower_len is not None and upper_len is not None and lower_len != upper_len:
            raise ValueError("`force_range.lower` and `force_range.upper` must have the same length.")
        return self


class _ActuatorBaseIR(StrictModel):
    name: str = Field(min_length=1)
    dofs_idx_local: tuple[int, ...] | None = None
    joint_names: tuple[str, ...] | None = None
    force_range: ActuatorForceRangeIR | None = Field(
        default=None,
        description=(
            "Actuator force/torque limit. This caps available output authority; if it is too small, the joint may "
            "still be weak even with large control gains."
        ),
    )
    armature: ScalarOrSequence | None = None

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
    def _check_selector_and_lengths(self) -> "_ActuatorBaseIR":
        if self.dofs_idx_local is None and self.joint_names is None:
            raise ValueError("Actuator requires one selector: `dofs_idx_local` or `joint_names`.")
        if self.dofs_idx_local is not None and self.joint_names is not None:
            raise ValueError("Only one of `dofs_idx_local` or `joint_names` can be provided.")

        if self.dofs_idx_local is not None:
            expected = len(self.dofs_idx_local)
            for field_name, value in (("armature", self.armature),):
                value_len = length_if_sequence(value)
                if value_len is not None and value_len != expected:
                    raise ValueError(f"Length mismatch: `{field_name}` must match `dofs_idx_local` length ({expected}).")

            if self.force_range is not None:
                lower_len = length_if_sequence(self.force_range.lower)
                upper_len = length_if_sequence(self.force_range.upper)
                if lower_len is not None and lower_len != expected:
                    raise ValueError("Length mismatch: `force_range.lower` must match `dofs_idx_local` length.")
                if upper_len is not None and upper_len != expected:
                    raise ValueError("Length mismatch: `force_range.upper` must match `dofs_idx_local` length.")
        return self


class PositionActuatorIR(_ActuatorBaseIR):
    kind: Literal["position"] = "position"
    kp: ScalarOrSequence = Field(
        default=80.0,
        description=(
            "Position-control proportional gain. Larger kp makes tracking stiffer, but too large a value can "
            "cause oscillation or instability."
        ),
    )
    kv: ScalarOrSequence | None = Field(
        default=None,
        description=(
            "Position-control damping gain. kv helps suppress oscillation; too little damping can be shaky, while "
            "too much can make motion sluggish."
        ),
    )

    @model_validator(mode="after")
    def _check_pd_lengths(self) -> "PositionActuatorIR":
        if self.dofs_idx_local is None:
            return self

        expected = len(self.dofs_idx_local)
        for field_name, value in (("kp", self.kp), ("kv", self.kv)):
            value_len = length_if_sequence(value)
            if value_len is not None and value_len != expected:
                raise ValueError(f"Length mismatch: `{field_name}` must match `dofs_idx_local` length ({expected}).")
        return self


class MotorActuatorIR(_ActuatorBaseIR):
    kind: Literal["motor"] = "motor"


ActuatorIR = Annotated[
    PositionActuatorIR | MotorActuatorIR,
    Field(discriminator="kind"),
]


class BodyIR(StrictModel):
    name: str = Field(default="body", min_length=1)
    shape: ShapeIR
    simulation_kind: Literal["rigid", "pbd"] = "rigid"
    deformable_material: DeformableMaterialIR | None = None
    initial_pose: PoseIR = Field(default_factory=PoseIR)
    fixed: bool = Field(
        default=False,
        description=(
            "Whether this rigid body should be fixed in the world. Use this for rigid obstacles, tables, platforms, "
            "or anchored articulated bodies."
        ),
    )
    visualize_contact: bool = False
    rho: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Material density. Higher rho makes the body heavier and increases inertia, but does not change geometric size. Adjust this parameter to make a body heavier or lighter without changing its shape, which is useful for tuning interactive behaviors between bodies"
        ),
    )
    collision: CollisionIR = Field(default_factory=CollisionIR)
    actuators: tuple[ActuatorIR, ...] = ()

    @property
    def is_deformable(self) -> bool:
        return self.simulation_kind == "pbd"

    @property
    def is_articulated(self) -> bool:
        return isinstance(self.shape, (MJCFShapeIR, URDFShapeIR))

    @model_validator(mode="after")
    def _check_fixed_support(self) -> "BodyIR":
        if self.fixed and isinstance(self.shape, MJCFShapeIR):
            raise ValueError(
                "`bodies[].fixed=true` is not supported for `mjcf` shapes. "
                "For MJCF bodies, encode a fixed base directly in the XML (for example by omitting the freejoint)."
            )
        if self.is_deformable:
            if self.is_articulated:
                raise ValueError("`simulation_kind='pbd'` does not support articulated shapes (`mjcf` or `urdf`).")
            if not isinstance(self.shape, (SphereShapeIR, BoxShapeIR, CylinderShapeIR, MeshShapeIR)):
                raise ValueError(
                    "`simulation_kind='pbd'` only supports `sphere`, `box`, `cylinder`, or `mesh` shapes in v1."
                )
            if self.deformable_material is None:
                raise ValueError("`simulation_kind='pbd'` requires `deformable_material`.")
            if self.rho is not None:
                raise ValueError("Use `deformable_material.rho` instead of `body.rho` for PBD bodies.")
            if self.fixed:
                raise ValueError("`bodies[].fixed=true` is not supported for PBD bodies in deformable v1.")
            if len(self.actuators) > 0:
                raise ValueError("PBD bodies do not support `actuators` in deformable v1.")
            if self.collision.coup_friction is not None or self.collision.coup_restitution is not None:
                raise ValueError("`coup_friction` and `coup_restitution` are not supported on PBD bodies.")
            if self.collision.sol_params is not None:
                raise ValueError("`collision.sol_params` is not supported on PBD bodies.")
            if self.collision.contact_resistance is not None:
                raise ValueError("`collision.contact_resistance` is not supported on PBD bodies in deformable v1.")
        elif self.deformable_material is not None:
            raise ValueError("`deformable_material` requires `simulation_kind='pbd'`.")
        return self
