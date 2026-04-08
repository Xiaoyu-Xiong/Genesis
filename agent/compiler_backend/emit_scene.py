from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..defaults import DEFAULTS
from ..ir_schema import RenderIR, RigidIR
from .formatting import fmt_tuple, safe_var_name
from .morph_material import body_material_source, body_morph_source, emit_collision_overrides, material_kwargs_from_collision


@dataclass(frozen=True)
class SceneEmitContext:
    render: RenderIR | None
    entity_vars: dict[str, str]
    body_vars: dict[str, str]


def emit_scene_setup(emit: Callable[[int, str], None], program: RigidIR) -> SceneEmitContext:
    backend_expr = "gs.cpu" if program.scene.backend == "cpu" else "gs.gpu"
    has_deformable_bodies = any(body.is_deformable for body in program.bodies)
    if not program.scene.show_viewer:
        emit(1, 'os.environ.setdefault("PYOPENGL_PLATFORM", "egl")')
        emit(1, 'os.environ.setdefault("MUJOCO_GL", "egl")')
        emit(1, 'os.environ.setdefault("PYGLET_HEADLESS", "1")')
        emit(1)
    if has_deformable_bodies:
        emit(1, f"gs.init(backend={backend_expr}, precision={DEFAULTS.deformable.genesis_precision!r})")
    else:
        emit(1, f"gs.init(backend={backend_expr})")
    emit(1, "scene = gs.Scene(")
    emit(2, "sim_options=gs.options.SimOptions(")
    emit(3, f"dt={program.scene.sim.dt},")
    emit(3, f"gravity={fmt_tuple(program.scene.sim.gravity)},")
    emit(2, "),")
    if has_deformable_bodies:
        emit(2, "pbd_options=gs.options.PBDOptions(")
        emit(3, f"particle_size={DEFAULTS.deformable.particle_size},")
        emit(3, f"max_stretch_solver_iterations={DEFAULTS.deformable.max_stretch_solver_iterations},")
        emit(3, f"max_bending_solver_iterations={DEFAULTS.deformable.max_bending_solver_iterations},")
        emit(3, f"max_volume_solver_iterations={DEFAULTS.deformable.max_volume_solver_iterations},")
        emit(3, f"max_density_solver_iterations={DEFAULTS.deformable.max_density_solver_iterations},")
        emit(3, f"max_viscosity_solver_iterations={DEFAULTS.deformable.max_viscosity_solver_iterations},")
        emit(3, f"lower_bound={fmt_tuple(DEFAULTS.deformable.lower_bound)},")
        emit(3, f"upper_bound={fmt_tuple(DEFAULTS.deformable.upper_bound)},")
        emit(2, "),")
    if program.scene.viewer is not None:
        viewer = program.scene.viewer
        emit(2, "viewer_options=gs.options.ViewerOptions(")
        emit(3, f"camera_pos={fmt_tuple(viewer.camera_pos)},")
        emit(3, f"camera_lookat={fmt_tuple(viewer.camera_lookat)},")
        emit(3, f"camera_fov={viewer.camera_fov},")
        emit(2, "),")
    emit(2, f"show_viewer={program.scene.show_viewer},")
    emit(1, ")")
    emit(1)

    render = program.scene.render
    entity_vars: dict[str, str] = {}
    if program.scene.add_ground:
        ground_var = safe_var_name("ground")
        entity_vars["ground"] = ground_var
        if has_deformable_bodies:
            friction = (
                program.scene.ground_collision.friction
                if program.scene.ground_collision is not None and program.scene.ground_collision.friction is not None
                else None
            )
            if friction is not None:
                emit(
                    1,
                    f"{ground_var} = scene.add_entity("
                    f"gs.morphs.Plane(), material=gs.materials.Rigid(friction={friction}, needs_coup=False), name='ground')",
                )
            else:
                emit(
                    1,
                    f"{ground_var} = scene.add_entity("
                    "gs.morphs.Plane(), material=gs.materials.Rigid(needs_coup=False), name='ground')",
                )
        else:
            ground_material_kwargs = material_kwargs_from_collision(
                rho=None,
                collision=program.scene.ground_collision,
            )
            if ground_material_kwargs:
                emit(
                    1,
                    f"{ground_var} = scene.add_entity("
                    f"gs.morphs.Plane(), material=gs.materials.Rigid({', '.join(ground_material_kwargs)}), name='ground')",
                )
            else:
                emit(1, f"{ground_var} = scene.add_entity(gs.morphs.Plane(), name='ground')")
    elif has_deformable_bodies and render is not None:
        emit(1, "_visual_ground = scene.add_entity(gs.morphs.Plane(collision=False), name='_visual_ground')")

    body_vars: dict[str, str] = {}
    for body in program.bodies:
        body_var = safe_var_name(body.name)
        body_vars[body.name] = body_var
        entity_vars[body.name] = body_var
        emit(1, f"{body_var} = scene.add_entity(")
        emit(2, f"morph={body_morph_source(body)},")
        body_material = body_material_source(body)
        if body_material is not None:
            emit(2, f"material={body_material},")
        if not body.is_deformable:
            emit(2, f"visualize_contact={body.visualize_contact},")
        emit(2, f"name={body.name!r},")
        emit(1, ")")
        emit(1)

    if render is not None:
        emit(1, "camera = scene.add_camera(")
        emit(2, f"res={fmt_tuple(render.res)},")
        emit(2, f"pos={fmt_tuple(render.camera_pos)},")
        emit(2, f"lookat={fmt_tuple(render.camera_lookat)},")
        emit(2, f"up={fmt_tuple(render.camera_up)},")
        emit(2, f"fov={render.camera_fov},")
        emit(2, f"near={render.near},")
        emit(2, f"far={render.far},")
        emit(2, f"GUI={render.gui},")
        emit(1, ")")
    else:
        emit(1, "camera = None")
    emit(1)

    emit(1, "entities = {")
    for entity_name, entity_var in entity_vars.items():
        emit(2, f"{entity_name!r}: {entity_var},")
    emit(1, "}")
    emit(1, "scene.build()")
    if render is not None and render.follow_entity is not None:
        follow = render.follow_entity
        emit(1, "camera.follow_entity(")
        emit(2, f"_follow_entity_target(entities[{follow.entity!r}]),")
        emit(2, f"fixed_axis={repr(tuple(follow.fixed_axis))},")
        emit(2, f"smoothing={follow.smoothing!r},")
        emit(2, f"fix_orientation={follow.fix_orientation},")
        emit(1, ")")
    for body in program.bodies:
        if not body.is_deformable:
            emit_collision_overrides(emit, entity_var=body_vars[body.name], collision=body.collision)
    if program.scene.add_ground:
        emit_collision_overrides(
            emit,
            entity_var=entity_vars["ground"],
            collision=program.scene.ground_collision,
        )

    return SceneEmitContext(render=render, entity_vars=entity_vars, body_vars=body_vars)
