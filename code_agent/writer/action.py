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
    - read each actor dictionary's `initial_velocity` and apply it with `set_dofs_velocity`
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
