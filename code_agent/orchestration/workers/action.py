from __future__ import annotations

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="action",
    target_file="src/action.py",
    required_export="run_actions",
    responsibility="controls, simulation stepping, event logs, metrics, and task result artifacts",
    prompt_body="""
    Write `run_actions(scene, actors, *, out_dir: Path, steps: int = 40)`.
    It must:
    - create `out_dir`
    - read each actor dictionary's `initial_velocity` and apply it with `set_dofs_velocity`
    - step the scene exactly `steps` times
    - sample actor positions into `event_log.json` with shape
      `{"steps": int, "samples": [{"step": int, "actors": {"name": [x,y,z]}}]}`
    - write `metrics.json`, `summary.json`, and `run_result.json`
    - mark `metrics["success"]` true when execution completes
    Do not render. Rendering is owned by `rendering.py`.
    """,
)
