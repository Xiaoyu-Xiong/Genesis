from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from code_agent.utils.codex import CodexExecResult

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
    codex_result: CodexExecResult
    worker_report: dict[str, object] | None = None
    error_message: str | None = None


COMMON_RULES = """
You are authoring one module for a generated Genesis simulation project.
The full repository and current case workspace are available for context. Read source, contracts, reports, logs,
assets, and generated artifacts as needed before writing your module.
You are not alone in the workspace: other workers own other modules, and you may read their modules to coordinate
interfaces and behavior.
Edit only the exact target file assigned to you. Do not edit any other file.
You may run lightweight read-only inspection commands such as `pwd`, `ls`, `find`, `rg`, `sed`, and `cat`.
Do not run Python, uv, pytest, Genesis, or any simulation. Do not mutate the environment or generated artifacts.
Use ASCII only. Keep code compact and robust for local GPU execution.
The generated code will run through the repository uv environment on the dedicated local GPU later, but you must not
execute it yourself.
"""

RIGID_API_GUIDE = """
Genesis rigid primitive API constraints:
- Import Genesis as `import genesis as gs`.
- Initialize with `gs.init(backend=gs.cpu if backend == "cpu" else gs.gpu, precision="32",
  performance_mode=True, logging_level="warning")`.
- Create a scene with `gs.Scene(..., show_viewer=False, show_FPS=False)`.
- Add a ground plane with `scene.add_entity(gs.morphs.Plane())`.
- Use primitive morphs such as `gs.morphs.Box(size=(...), pos=(...), fixed=True/False)`,
  `gs.morphs.Sphere(radius=..., pos=(...))`, and `gs.morphs.Cylinder(radius=..., height=..., pos=(...))`.
- Add all entities before `scene.build()`.
- For a free rigid primitive, initial velocity uses `entity.set_dofs_velocity((vx, vy, vz, wx, wy, wz))` before the
  first simulation step only. Do this only for dynamic entities with positive dof count; generated mesh entities may be
  fixed or dynamic, so rely on explicit `fixed`/`static` metadata and `n_dofs`, not meshness itself.
- During the simulation loop, do not use direct position or velocity state writes to force a desired trajectory. Use
  physics controls: `scene.sim.rigid_solver.apply_links_external_force(...)`,
  `scene.sim.rigid_solver.apply_links_external_torque(...)`, or `entity.control_dofs_force(...)` for actuated dofs.
- Position samples can use `entity.get_pos()`, converting tensors with detach/cpu/numpy/tolist if needed.
- For generated mesh assets, read `assets/asset_manifest.json` from the prompt. Use entries with
  `source_type == "generated_mesh"` and `status == "ready"`. Instantiate runtime geometry with
  `gs.morphs.Mesh(file=entry["runtime_path"], scale=entry["scale"] or 1.0,
  visual_file=entry.get("visual_path"), file_meshes_are_zup=entry.get("file_meshes_are_zup"), pos=(...),
  fixed=True/False)`.
  `runtime_path` is the strict-manifold simulation/collision mesh. `visual_path` is a seam-aware textured render
  mesh attached to the same rigid entity, not a separate object to instantiate as an independent simulation body.
  `texture_path` is evidence metadata for the transferred base-color image and texture preview checks.
  Use the manifest's logical names, scale factors, coordinate metadata, texture paths, and simulation roles instead of
  guessing mesh paths, sizes, or orientation.
"""
