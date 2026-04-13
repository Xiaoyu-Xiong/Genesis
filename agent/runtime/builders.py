from __future__ import annotations

from typing import Any

from ..configs import CONFIGS
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


def build_body_morph(gs: Any, body: BodyIR) -> Any:
    shape = body.shape
    pose = body.initial_pose
    pos = tuple(pose.pos)
    quat = tuple(pose.quat)
    fixed = body.fixed
    tet_kwargs: dict[str, Any] = {}
    if body.is_deformable:
        tet_kwargs["tet_resolution"] = CONFIGS.deformable.tet_resolution

    if isinstance(shape, SphereShapeIR):
        return gs.morphs.Sphere(radius=shape.radius, pos=pos, quat=quat, fixed=fixed, **tet_kwargs)
    if isinstance(shape, BoxShapeIR):
        return gs.morphs.Box(size=tuple(shape.size), pos=pos, quat=quat, fixed=fixed, **tet_kwargs)
    if isinstance(shape, CylinderShapeIR):
        return gs.morphs.Cylinder(
            radius=shape.radius,
            height=shape.height,
            pos=pos,
            quat=quat,
            fixed=fixed,
            **tet_kwargs,
        )
    if isinstance(shape, MeshShapeIR):
        return gs.morphs.Mesh(file=shape.file, scale=shape.scale, pos=pos, quat=quat, fixed=fixed, **tet_kwargs)
    if isinstance(shape, MJCFShapeIR):
        return gs.morphs.MJCF(
            file=shape.file,
            scale=shape.scale,
            pos=pos,
            quat=quat,
            requires_jac_and_IK=shape.requires_jac_and_IK,
            default_armature=shape.default_armature,
        )
    if isinstance(shape, URDFShapeIR):
        return gs.morphs.URDF(
            file=shape.file,
            scale=shape.scale,
            pos=pos,
            quat=quat,
            requires_jac_and_IK=shape.requires_jac_and_IK,
            fixed=(fixed or shape.fixed),
            merge_fixed_links=shape.merge_fixed_links,
            default_armature=shape.default_armature,
        )

    raise TypeError(f"Unsupported shape IR: {type(shape).__name__}")


def build_body_material(gs: Any, body: BodyIR) -> Any | None:
    if body.is_deformable:
        material = body.deformable_material
        if CONFIGS.deformable.simulation_backend == "pbd":
            if not isinstance(material, PBDElasticMaterialIR):
                raise TypeError(f"Unsupported deformable material IR: {type(material).__name__}")
            friction = body.collision.friction if body.collision.friction is not None else CONFIGS.deformable.friction
            kwargs: dict[str, Any] = {
                "rho": material.rho,
                "static_friction": friction,
                "kinetic_friction": friction,
                "stretch_compliance": material.stretch_compliance,
                "volume_compliance": material.volume_compliance,
                "stretch_relaxation": CONFIGS.deformable.stretch_relaxation,
                "bending_relaxation": CONFIGS.deformable.bending_relaxation,
                "volume_relaxation": CONFIGS.deformable.volume_relaxation,
            }
            return gs.materials.PBD.Elastic(**kwargs)

        if not isinstance(material, FEMElasticMaterialIR):
            raise TypeError(f"Unsupported deformable material IR: {type(material).__name__}")
        return gs.materials.FEM.Elastic(
            E=material.E,
            nu=material.nu,
            rho=material.rho,
            model=CONFIGS.deformable.fem_model,
            hydroelastic_modulus=CONFIGS.deformable.fem_hydroelastic_modulus,
            friction_mu=CONFIGS.deformable.fem_friction_mu,
            contact_resistance=CONFIGS.deformable.fem_contact_resistance,
            hessian_invariant=CONFIGS.deformable.fem_hessian_invariant,
        )
    coup_type_override = None
    if CONFIGS.deformable.simulation_backend == "fem_ipc" and not body.is_articulated:
        if body.fixed:
            coup_type_override = "ipc_only"
        else:
            coup_type_override = "two_way_soft_constraint"
    return build_rigid_material(gs, rho=body.rho, collision=body.collision, coup_type_override=coup_type_override)


def build_rigid_material(
    gs: Any,
    *,
    rho: float | None,
    collision: CollisionIR | None,
    coup_type_override: str | None = None,
) -> Any | None:
    material_kwargs: dict[str, Any] = {}
    if rho is not None:
        material_kwargs["rho"] = rho
    if collision is not None:
        if collision.friction is not None:
            material_kwargs["friction"] = collision.friction
        if collision.coup_friction is not None:
            material_kwargs["coup_friction"] = collision.coup_friction
        if collision.coup_restitution is not None:
            material_kwargs["coup_restitution"] = collision.coup_restitution
        if collision.contact_resistance is not None:
            material_kwargs["contact_resistance"] = collision.contact_resistance
    if coup_type_override is not None:
        material_kwargs["coup_type"] = coup_type_override
    if not material_kwargs:
        return None
    return gs.materials.Rigid(**material_kwargs)


def apply_collision_overrides(entity: Any, collision: CollisionIR | None) -> None:
    if collision is None:
        return

    if collision.friction is not None:
        entity.set_friction(collision.friction)

    if collision.sol_params is not None:
        for link in entity.links:
            for geom in link.geoms:
                geom.set_sol_params(collision.sol_params)
