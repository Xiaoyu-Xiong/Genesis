from __future__ import annotations

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="action",
    target_file="src/action.py",
    required_export="run_actions",
    responsibility="controls, simulation stepping, event logs, metrics, and task result artifacts",
    prompt_body="""
    Write `run_actions(scene, actors, *, out_dir: Path, steps: int, render_state=None)`.
    It must:
    - create `out_dir`
	    - apply actor initial velocities when appropriate before the first simulation step, while keeping fixed/static props
	      stable
	    - when multiple actors expose `initial_velocity`, use each value only for its own actor; do not choose a projectile
	      launch velocity by scanning unrelated static props, since static actors often carry explicit zero velocities
	    - identify projectile actors from stable identity fields such as `name`, `logical_name`, `actor_name`, or direct dict
	      keys; treat descriptive `role` strings only as context. A static barrier role may mention "projectile contact",
	      but that must not make the barrier the projectile.
	    - generated mesh actors are not automatically fixed; they may be dynamic when the body worker explicitly makes
	      them dynamic. Never infer static/dynamic status from meshness alone.
	    - never select actors with explicit `fixed`, `static`, `is_static`, or `n_dofs == 0` as the launched projectile.
	      Prefer an exact `dense_launched_sphere` identity when present, and fail loudly rather than applying velocity or
	      force to an explicitly static or zero-dof prop.
	    - when calling `set_dofs_velocity`, match the selected dynamic entity's available dofs. Free rigid bodies usually
	      accept 6 values; actors with zero dofs must be skipped instead of receiving launch velocity.
	    - for physical realism, direct state writes such as setting positions, qpos, dof positions, or dof velocities are
	      initialization-only; after stepping begins, express motion through physics controls such as external forces/torques
	      on rigid links or actuator force commands, not by teleporting bodies or overwriting velocities mid-simulation
    - step the scene exactly `steps` times
    - if `render_state` is not None, call its capture hook before stepping for the initial frame and after each
      `scene.step()`; `render_state` is normally a dict, so use
      `render_state["capture_frame"](render_state, step)` when that key exists. Pass the exact simulation step index and
      let rendering.py downsample to the requested video fps/frame budget; do not create cameras or renderers in
      `action.py`
    - sample actor positions into `event_log.json` with shape
      `{"steps": int, "samples": [{"step": int, "actors": {"name": [x,y,z]}}]}`
    - write `metrics.json`, `summary.json`, and `run_result.json`
    - mark `metrics["success"]` true when execution completes
    Rendering setup, Genesis camera creation, frame saving, and video composition are owned by `rendering.py`.
    """,
)
