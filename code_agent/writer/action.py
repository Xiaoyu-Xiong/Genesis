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
	    - for FEM actors, an `initial_velocity` with three numbers should be applied with `entity.set_velocity(...)` before
	      the first step only; do not call `set_position` or `set_velocity` repeatedly during the simulation to fake
	      deformation
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
	    - do not use external forces as hidden object-following proxies for passive task objects; a manipulated object
	      should move through modeled contacts, joints, actuators, collisions, friction, or an explicitly requested
	      physical effect, and all force paths must be logged in metrics
	    - for XML/MJCF articulated assets, drive the mechanism through the actuator, DOF, or control handles exposed by
	      `body.py` in `actors`; do not bypass the XML-designed actuators by overwriting root poses or link velocities
	      during the simulation. If the required actuator/DOF groups cannot be found, fail with a clear error and write
	      diagnostics naming the missing control contract so the planner/critic can route a repair to body or action.
	    - when controlling XML/MJCF robots or mechanisms, log the chosen control path, actuator names when available,
	      commanded values, and key pose/heading/target-distance samples so the critic can compare the source, metrics,
	      video, and original text prompt.
    - step the scene exactly `steps` times
    - if `render_state` is not None, call its capture hook before stepping for the initial frame and after each
      `scene.step()`; `render_state` is normally a dict, so use
      `render_state["capture_frame"](render_state, step)` when that key exists. Pass the exact simulation step index and
      let rendering.py downsample to the requested video fps/frame budget; do not create cameras or renderers in
      `action.py`
    - do not skip the capture hook when a state cache is requested. The main entrypoint may wrap the normal rendering
      hook so the same call writes `artifacts/state_cache/states/frame_*.npz`; missing these npz files is a hard
      execution failure when `--require-state-cache` is active.
    - expose every dynamic and articulated task entity through the returned `actors` structure with a stable unique
      name and its Genesis entity handle. The shared cache writer discovers qpos, DOF positions, rigid pose, and
      deformable vertex state from these handles; hiding an articulated entity inside opaque metadata makes complete
      replay impossible.
    - articulated state cache is a hard contract: every captured frame must contain replayable qpos or DOF positions
      for every actor with DOFs. Root position/quaternion alone is not sufficient for rollers, drums, robot links,
      gates, paddles, or other joint-driven mechanisms.
    - sample actor positions into `event_log.json` with shape
      `{"steps": int, "samples": [{"step": int, "actors": {"name": [x,y,z]}}]}`
    - for FEM/deformable actors, sample `entity.get_state().pos` when available and record center of mass, bbox min/max,
      height, lateral spread, and a simple deformation proxy into both `event_log.json` samples and `metrics.json`
    - FEM state reads can be expensive; do not call `get_state()` for every FEM actor on every simulation step unless the
      Planner explicitly asks for dense telemetry. Prefer sparse metric samples at the first step, final step, render
      capture steps, and/or a fixed interval such as every 0.25-0.5s; still call `scene.step()` for every physics step.
    - for soft stacks such as jelly_cube_stack, metrics should include collapse/tilt evidence such as initial/final
      stack height, max lateral spread, final top cube height, and whether the stack visibly changed shape
    - write `metrics.json`, `summary.json`, and `run_result.json`
    - mark `metrics["success"]` true when execution completes
    Rendering setup, Genesis camera creation, frame saving, and video composition are owned by `rendering.py`.
    """,
)
