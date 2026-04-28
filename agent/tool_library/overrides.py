from __future__ import annotations

from ..configs import CONFIGS
from ..ir_schema.program import RigidIR
from ..llm_generator.constraints.general_constraints import ensure_program_has_render
from ..llm_generator.constraints.render_defaults import synchronize_render_timing


def apply_system_defaults(program: RigidIR) -> RigidIR:
    patched = ensure_program_has_render(program.model_copy(deep=True))

    if CONFIGS.deformable.simulation_backend == "fem_ipc" and any(body.is_deformable for body in patched.bodies):
        patched.scene.backend = "cpu"

    patched.scene.sim.dt = CONFIGS.runtime.sim_dt

    if patched.scene.render is not None:
        patched.scene.render.render_every_n_steps = CONFIGS.runtime.render_every_n_steps
        patched.scene.render.res = CONFIGS.runtime.render_res

    return synchronize_render_timing(patched)
