from __future__ import annotations

import os
from typing import Any

from ..configs import CONFIGS
from ..ir_schema import RigidIR
from .actuators import configure_actuators
from .builders import apply_collision_overrides, build_body_material, build_body_morph, build_rigid_material
from .helpers import get_follow_target_entity
from .models import RuntimeContext, RuntimeState


def _configure_follow_camera(program: RigidIR, runtime: RuntimeContext) -> None:
    render = runtime.render
    camera = runtime.camera
    if render is None or camera is None or render.follow_entity is None:
        return

    follow_cfg = render.follow_entity
    target_entity = runtime.entities.get(follow_cfg.entity)
    if target_entity is None:
        raise ValueError(f"Cannot follow unknown entity `{follow_cfg.entity}`.")

    camera.follow_entity(
        get_follow_target_entity(target_entity),
        fixed_axis=tuple(follow_cfg.fixed_axis),
        smoothing=follow_cfg.smoothing,
        fix_orientation=follow_cfg.fix_orientation,
    )


def configure_headless_if_needed(program: RigidIR) -> None:
    if not program.scene.show_viewer:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYGLET_HEADLESS", "1")


def ensure_genesis_initialized(gs: Any, program: RigidIR) -> None:
    has_deformable_bodies = any(body.is_deformable for body in program.bodies)
    if CONFIGS.deformable.simulation_backend == "fem_ipc" and has_deformable_bodies:
        requested_backend = gs.cpu
    else:
        requested_backend = gs.cpu if program.scene.backend == "cpu" else gs.gpu
    if getattr(gs, "_initialized", False):
        active_backend = getattr(gs, "backend", None)
        # if active_backend != requested_backend:
        #     raise ValueError(
        #         "Genesis already initialized with a different backend. "
        #         f"Active backend={active_backend}, requested backend={requested_backend}."
        #     )
        return
    init_kwargs: dict[str, Any] = {"backend": requested_backend}
    if has_deformable_bodies:
        init_kwargs["precision"] = CONFIGS.deformable.genesis_precision
    gs.init(**init_kwargs)


def create_runtime_context(gs: Any, program: RigidIR) -> RuntimeContext:
    viewer_options = None
    if program.scene.viewer is not None:
        viewer = program.scene.viewer
        viewer_options = gs.options.ViewerOptions(
            camera_pos=tuple(viewer.camera_pos),
            camera_lookat=tuple(viewer.camera_lookat),
            camera_fov=viewer.camera_fov,
        )

    has_deformable_bodies = any(body.is_deformable for body in program.bodies)
    if CONFIGS.deformable.simulation_backend == "fem_ipc" and has_deformable_bodies:
        program = program.model_copy(deep=True)
        program.scene.backend = "cpu"
    scene_kwargs: dict[str, Any] = {
        "sim_options": gs.options.SimOptions(
            dt=program.scene.sim.dt,
            gravity=tuple(program.scene.sim.gravity),
        ),
        "viewer_options": viewer_options,
        "show_viewer": program.scene.show_viewer,
    }
    if has_deformable_bodies:
        if CONFIGS.deformable.simulation_backend == "pbd":
            boundary_friction = (
                program.scene.ground_collision.friction
                if program.scene.ground_collision is not None and program.scene.ground_collision.friction is not None
                else CONFIGS.deformable.friction
            )
            scene_kwargs["pbd_options"] = gs.options.PBDOptions(
                particle_size=CONFIGS.deformable.particle_size,
                max_stretch_solver_iterations=CONFIGS.deformable.max_stretch_solver_iterations,
                max_bending_solver_iterations=CONFIGS.deformable.max_bending_solver_iterations,
                max_volume_solver_iterations=CONFIGS.deformable.max_volume_solver_iterations,
                max_density_solver_iterations=CONFIGS.deformable.max_density_solver_iterations,
                max_viscosity_solver_iterations=CONFIGS.deformable.max_viscosity_solver_iterations,
                lower_bound=CONFIGS.deformable.lower_bound,
                upper_bound=CONFIGS.deformable.upper_bound,
                boundary_static_friction=boundary_friction,
                boundary_kinetic_friction=boundary_friction,
            )
        else:
            scene_kwargs["fem_options"] = gs.options.FEMOptions()
            scene_kwargs["coupler_options"] = gs.options.IPCCouplerOptions(
                contact_d_hat=CONFIGS.deformable.ipc_contact_d_hat,
                contact_friction_enable=CONFIGS.deformable.ipc_contact_friction_enable,
                contact_resistance=CONFIGS.deformable.ipc_contact_resistance,
                contact_eps_velocity=CONFIGS.deformable.ipc_contact_eps_velocity,
                contact_constitution=CONFIGS.deformable.ipc_contact_constitution,
                collision_detection_method=CONFIGS.deformable.ipc_collision_detection_method,
                constraint_strength_translation=CONFIGS.deformable.ipc_constraint_strength_translation,
                constraint_strength_rotation=CONFIGS.deformable.ipc_constraint_strength_rotation,
                enable_rigid_ground_contact=CONFIGS.deformable.ipc_enable_rigid_ground_contact,
                enable_rigid_rigid_contact=CONFIGS.deformable.ipc_enable_rigid_rigid_contact,
                two_way_coupling=CONFIGS.deformable.ipc_two_way_coupling,
                enable_rigid_dofs_sync=CONFIGS.deformable.ipc_enable_rigid_dofs_sync,
                free_base_driven_by_ipc=CONFIGS.deformable.ipc_free_base_driven_by_ipc,
            )
    scene = gs.Scene(**scene_kwargs)

    render = program.scene.render
    camera = None
    if render is not None:
        camera = scene.add_camera(
            res=tuple(render.res),
            pos=tuple(render.camera_pos),
            lookat=tuple(render.camera_lookat),
            up=tuple(render.camera_up),
            fov=render.camera_fov,
            near=render.near,
            far=render.far,
            GUI=render.gui,
        )

    entities: dict[str, Any] = {}
    body_entities: dict[str, Any] = {}
    if program.scene.add_ground:
        ground_kwargs: dict[str, Any] = {
            "morph": gs.morphs.Plane(),
            "name": "ground",
        }
        if has_deformable_bodies:
            if CONFIGS.deformable.simulation_backend == "pbd":
                friction = (
                    program.scene.ground_collision.friction
                    if program.scene.ground_collision is not None and program.scene.ground_collision.friction is not None
                    else None
                )
                ground_kwargs["material"] = gs.materials.Rigid(friction=friction, needs_coup=False)
            else:
                friction = (
                    program.scene.ground_collision.friction
                    if program.scene.ground_collision is not None and program.scene.ground_collision.friction is not None
                    else None
                )
                ground_kwargs["material"] = gs.materials.Rigid(
                    friction=friction,
                    needs_coup=False,
                )
        else:
            ground_material = build_rigid_material(gs, rho=None, collision=program.scene.ground_collision)
            if ground_material is not None:
                ground_kwargs["material"] = ground_material
        entities["ground"] = scene.add_entity(**ground_kwargs)
        if has_deformable_bodies and CONFIGS.deformable.simulation_backend == "fem_ipc":
            ipc_ground_coup_friction = (
                program.scene.ground_collision.coup_friction
                if program.scene.ground_collision is not None and program.scene.ground_collision.coup_friction is not None
                else (
                    program.scene.ground_collision.friction
                    if program.scene.ground_collision is not None and program.scene.ground_collision.friction is not None
                    else CONFIGS.deformable.friction
                )
            )
            ipc_ground_coup_restitution = (
                program.scene.ground_collision.coup_restitution
                if program.scene.ground_collision is not None and program.scene.ground_collision.coup_restitution is not None
                else 0.0
            )
            ipc_ground_contact_resistance = (
                program.scene.ground_collision.contact_resistance
                if program.scene.ground_collision is not None and program.scene.ground_collision.contact_resistance is not None
                else CONFIGS.deformable.ipc_contact_resistance
            )
            scene.add_entity(
                morph=gs.morphs.Plane(visualization=False),
                material=gs.materials.Rigid(
                    coup_friction=ipc_ground_coup_friction,
                    coup_restitution=ipc_ground_coup_restitution,
                    contact_resistance=ipc_ground_contact_resistance,
                    coup_type="ipc_only",
                ),
                name="_ipc_ground",
            )
    elif has_deformable_bodies and render is not None:
        scene.add_entity(
            morph=gs.morphs.Plane(collision=False),
            name="_visual_ground",
        )

    for body in program.bodies:
        add_entity_kwargs: dict[str, Any] = {
            "morph": build_body_morph(gs, body),
            "name": body.name,
        }
        if not body.is_deformable:
            add_entity_kwargs["visualize_contact"] = body.visualize_contact
        body_material = build_body_material(gs, body)
        if body_material is not None:
            add_entity_kwargs["material"] = body_material
        body_entity = scene.add_entity(**add_entity_kwargs)
        entities[body.name] = body_entity
        body_entities[body.name] = body_entity

    return RuntimeContext(
        scene=scene,
        camera=camera,
        render=render,
        entities=entities,
        body_entities=body_entities,
    )


def build_runtime_context(program: RigidIR, runtime: RuntimeContext, state: RuntimeState) -> None:
    for body in program.bodies:
        if not body.is_deformable:
            apply_collision_overrides(runtime.body_entities[body.name], body.collision)
    if program.scene.add_ground:
        apply_collision_overrides(runtime.entities["ground"], program.scene.ground_collision)
    state.actuators_by_entity = {
        body.name: (
            configure_actuators(runtime.body_entities[body.name], body.actuators)
            if body.is_articulated
            else {}
        )
        for body in program.bodies
    }
    _configure_follow_camera(program, runtime)

    if runtime.camera is not None and runtime.render is not None:
        runtime.camera.start_recording()
        state.recording_started = True
        if runtime.render.include_initial_frame:
            runtime.camera.render(force_render=runtime.render.force_render)
            state.rendered_frames += 1
