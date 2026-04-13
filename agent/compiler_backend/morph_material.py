from __future__ import annotations

from collections.abc import Callable

from ..defaults import DEFAULTS
from ..ir_schema import (
    BodyIR,
    BoxShapeIR,
    CollisionIR,
    CylinderShapeIR,
    FEMElasticMaterialIR,
    MJCFShapeIR,
    MeshShapeIR,
    PBDElasticMaterialIR,
    SphereShapeIR,
    URDFShapeIR,
)
from .formatting import fmt_tuple


def body_morph_source(body: BodyIR) -> str:
    shape = body.shape
    pose = body.initial_pose
    pos_src = fmt_tuple(pose.pos)
    quat_src = fmt_tuple(pose.quat)
    fixed_src = body.fixed
    tet_kwarg = ""
    if body.is_deformable:
        tet_kwarg = f", tet_resolution={DEFAULTS.deformable.tet_resolution}"

    if isinstance(shape, SphereShapeIR):
        return f"gs.morphs.Sphere(radius={shape.radius}, pos={pos_src}, quat={quat_src}, fixed={fixed_src}{tet_kwarg})"
    if isinstance(shape, BoxShapeIR):
        return f"gs.morphs.Box(size={fmt_tuple(shape.size)}, pos={pos_src}, quat={quat_src}, fixed={fixed_src}{tet_kwarg})"
    if isinstance(shape, CylinderShapeIR):
        return (
            f"gs.morphs.Cylinder(radius={shape.radius}, height={shape.height}, "
            f"pos={pos_src}, quat={quat_src}, fixed={fixed_src}{tet_kwarg})"
        )
    if isinstance(shape, MeshShapeIR):
        return (
            "gs.morphs.Mesh("
            f"file={shape.file!r}, "
            f"scale={shape.scale}, "
            f"pos={pos_src}, "
            f"quat={quat_src}, "
            f"fixed={fixed_src}"
            f"{tet_kwarg}"
            ")"
        )
    if isinstance(shape, MJCFShapeIR):
        return (
            "gs.morphs.MJCF("
            f"file={shape.file!r}, "
            f"scale={shape.scale}, "
            f"pos={pos_src}, "
            f"quat={quat_src}, "
            f"requires_jac_and_IK={shape.requires_jac_and_IK}, "
            f"default_armature={shape.default_armature!r}"
            ")"
        )
    if isinstance(shape, URDFShapeIR):
        return (
            "gs.morphs.URDF("
            f"file={shape.file!r}, "
            f"scale={shape.scale}, "
            f"pos={pos_src}, "
            f"quat={quat_src}, "
            f"requires_jac_and_IK={shape.requires_jac_and_IK}, "
            f"fixed={fixed_src or shape.fixed}, "
            f"merge_fixed_links={shape.merge_fixed_links}, "
            f"default_armature={shape.default_armature!r}"
            ")"
        )

    raise TypeError(f"Unsupported shape IR: {type(shape).__name__}")


def body_material_source(body: BodyIR) -> str | None:
    if body.is_deformable:
        material = body.deformable_material
        if DEFAULTS.deformable.simulation_backend == "pbd":
            if not isinstance(material, PBDElasticMaterialIR):
                raise TypeError(f"Unsupported deformable material IR: {type(material).__name__}")
            friction = body.collision.friction if body.collision.friction is not None else DEFAULTS.deformable.friction
            kwargs = [
                f"rho={material.rho}",
                f"static_friction={friction}",
                f"kinetic_friction={friction}",
                f"stretch_compliance={material.stretch_compliance}",
                f"volume_compliance={material.volume_compliance}",
                f"stretch_relaxation={DEFAULTS.deformable.stretch_relaxation}",
                f"bending_relaxation={DEFAULTS.deformable.bending_relaxation}",
                f"volume_relaxation={DEFAULTS.deformable.volume_relaxation}",
            ]
            return f"gs.materials.PBD.Elastic({', '.join(kwargs)})"

        if not isinstance(material, FEMElasticMaterialIR):
            raise TypeError(f"Unsupported deformable material IR: {type(material).__name__}")
        kwargs = [
            f"E={material.E}",
            f"nu={material.nu}",
            f"rho={material.rho}",
            f"model={DEFAULTS.deformable.fem_model!r}",
            f"hydroelastic_modulus={DEFAULTS.deformable.fem_hydroelastic_modulus}",
            f"friction_mu={DEFAULTS.deformable.fem_friction_mu}",
            f"contact_resistance={DEFAULTS.deformable.fem_contact_resistance!r}",
            f"hessian_invariant={DEFAULTS.deformable.fem_hessian_invariant}",
        ]
        return f"gs.materials.FEM.Elastic({', '.join(kwargs)})"
    coup_type_override = None
    if DEFAULTS.deformable.simulation_backend == "fem_ipc" and not body.is_articulated:
        if body.fixed:
            coup_type_override = "ipc_only"
        else:
            coup_type_override = "two_way_soft_constraint"
    kwargs = material_kwargs_from_collision(rho=body.rho, collision=body.collision, coup_type_override=coup_type_override)
    if not kwargs:
        return None
    return f"gs.materials.Rigid({', '.join(kwargs)})"


def material_kwargs_from_collision(
    *,
    rho: float | None,
    collision: CollisionIR | None,
    coup_type_override: str | None = None,
) -> list[str]:
    kwargs: list[str] = []
    if rho is not None:
        kwargs.append(f"rho={rho}")
    if collision is not None:
        if collision.friction is not None:
            kwargs.append(f"friction={collision.friction}")
        if collision.coup_friction is not None:
            kwargs.append(f"coup_friction={collision.coup_friction}")
        if collision.coup_restitution is not None:
            kwargs.append(f"coup_restitution={collision.coup_restitution}")
        if collision.contact_resistance is not None:
            kwargs.append(f"contact_resistance={collision.contact_resistance}")
    if coup_type_override is not None:
        kwargs.append(f"coup_type={coup_type_override!r}")
    return kwargs


def emit_collision_overrides(
    emit: Callable[[int, str], None],
    *,
    entity_var: str,
    collision: CollisionIR | None,
) -> None:
    if collision is None:
        return
    if collision.friction is not None:
        emit(1, f"{entity_var}.set_friction({collision.friction})")
    if collision.sol_params is not None:
        emit(1, f"_sol_params = {fmt_tuple(collision.sol_params)}")
        emit(1, f"for _link in {entity_var}.links:")
        emit(2, "for _geom in _link.geoms:")
        emit(3, "_geom.set_sol_params(_sol_params)")
