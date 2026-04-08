from __future__ import annotations

from ..defaults import DEFAULTS
from ..ir_schema import RigidIR
from ..llm_generator.constraints import ensure_program_has_render, synchronize_render_timing


def apply_system_defaults(program: RigidIR) -> RigidIR:
    patched = ensure_program_has_render(program.model_copy(deep=True))

    patched.scene.sim.dt = DEFAULTS.runtime.sim_dt

    if patched.scene.render is not None:
        patched.scene.render.render_every_n_steps = DEFAULTS.runtime.render_every_n_steps
        patched.scene.render.res = DEFAULTS.runtime.render_res

    return synchronize_render_timing(patched)
