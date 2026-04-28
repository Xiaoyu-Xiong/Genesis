from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from code_agent.codex.runner import CodexResult

WorkerRole = Literal["scene", "body", "action", "rendering"]


@dataclass(slots=True, frozen=True)
class WorkerSpec:
    role: WorkerRole
    target_file: str
    required_export: str
    responsibility: str
    prompt_body: str


@dataclass(slots=True)
class WorkerDispatchResult:
    role: WorkerRole
    ok: bool
    target_path: Path
    codex_result: CodexResult
    worker_report: dict[str, object] | None = None
    error_message: str | None = None


COMMON_RULES = """
You are authoring one module for a generated Genesis simulation project.
You are not alone in the workspace: other workers own other modules.
Do not edit files and do not use filesystem tools. Return the complete target module source code in the `source_code` JSON field.
The coordinator will write `source_code` to the exact target path after your response.
Do not run shell commands. Do not run Python, uv, pytest, Genesis, or any simulation.
Use ASCII only. Keep code compact and robust for CPU smoke execution.
The generated code must run inside the repository's Apptainer environment later, but you must not execute it yourself.
"""

RIGID_API_GUIDE = """
Genesis rigid primitive API constraints:
- Import Genesis as `import genesis as gs`.
- Initialize with `gs.init(backend=gs.cpu if backend == "cpu" else gs.gpu, precision="32", performance_mode=True, logging_level="warning")`.
- Create a scene with `gs.Scene(..., show_viewer=False, show_FPS=False)`.
- Add a ground plane with `scene.add_entity(gs.morphs.Plane())`.
- Use primitive morphs such as `gs.morphs.Box(size=(...), pos=(...), fixed=True/False)`,
  `gs.morphs.Sphere(radius=..., pos=(...))`, and `gs.morphs.Cylinder(radius=..., height=..., pos=(...))`.
- Add all entities before `scene.build()`.
- For a free rigid primitive, initial velocity uses `entity.set_dofs_velocity((vx, vy, vz, wx, wy, wz))`.
- Position samples can use `entity.get_pos()`, converting tensors with detach/cpu/numpy/tolist if needed.
"""
