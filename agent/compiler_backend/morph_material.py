from __future__ import annotations

from collections.abc import Callable

from ..ir_schema import (
    BodyIR,
    BoxShapeIR,
    CollisionIR,
    CylinderShapeIR,
    MJCFShapeIR,
    MeshShapeIR,
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

    if isinstance(shape, SphereShapeIR):
        return f"gs.morphs.Sphere(radius={shape.radius}, pos={pos_src}, quat={quat_src}, fixed={fixed_src})"
    if isinstance(shape, BoxShapeIR):
        return f"gs.morphs.Box(size={fmt_tuple(shape.size)}, pos={pos_src}, quat={quat_src}, fixed={fixed_src})"
    if isinstance(shape, CylinderShapeIR):
        return (
            f"gs.morphs.Cylinder(radius={shape.radius}, height={shape.height}, "
            f"pos={pos_src}, quat={quat_src}, fixed={fixed_src})"
        )
    if isinstance(shape, MeshShapeIR):
        return (
            "gs.morphs.Mesh("
            f"file={shape.file!r}, "
            f"scale={shape.scale}, "
            f"pos={pos_src}, "
            f"quat={quat_src}, "
            f"fixed={fixed_src}"
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


def material_kwargs_from_collision(
    *,
    rho: float | None,
    collision: CollisionIR | None,
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
