from __future__ import annotations

from typing import Any

from ..defaults import DEFAULTS
from ..ir_schema import (
    BodyIR,
    BoxShapeIR,
    CollisionIR,
    CylinderShapeIR,
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
        tet_kwargs["tet_resolution"] = DEFAULTS.deformable.tet_resolution

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
        return gs.morphs.Mesh(file=shape.file, scale=shape.scale, pos=pos, quat=quat, fixed=fixed)
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
        if not isinstance(material, PBDElasticMaterialIR):
            raise TypeError(f"Unsupported deformable material IR: {type(material).__name__}")
        friction = body.collision.friction if body.collision.friction is not None else DEFAULTS.deformable.friction
        kwargs: dict[str, Any] = {
            "rho": material.rho,
            "static_friction": friction,
            "kinetic_friction": friction,
            "stretch_compliance": material.stretch_compliance,
            "volume_compliance": material.volume_compliance,
            "stretch_relaxation": DEFAULTS.deformable.stretch_relaxation,
            "bending_relaxation": DEFAULTS.deformable.bending_relaxation,
            "volume_relaxation": DEFAULTS.deformable.volume_relaxation,
        }
        return gs.materials.PBD.Elastic(**kwargs)
    return build_rigid_material(gs, rho=body.rho, collision=body.collision)


def build_rigid_material(
    gs: Any,
    *,
    rho: float | None,
    collision: CollisionIR | None,
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
