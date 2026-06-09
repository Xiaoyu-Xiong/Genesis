"""Writer-worker prompt clauses after SimDebug card migration."""

WORKER_COMMON_RULES = """
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
Use the Planner-dispatched SimDebug cards for task-specific debugging experience, physical restrictions, asset rules,
IPC/FEM guidance, and source-aware repair hints.
""".strip()


RIGID_API_GUIDE = """
Genesis rigid primitive API constraints:
- Import Genesis as `import genesis as gs`.
- Initialize with `gs.init(backend=gs.cpu if backend == "cpu" else gs.gpu, precision="32",
  performance_mode=True, logging_level="warning")`.
- Create a scene with `gs.Scene(sim_options=gs.options.SimOptions(dt=sim_dt, substeps=sim_substeps), ...,
  show_viewer=False, show_FPS=False)` using the `sim_dt` and `sim_substeps` arguments passed into `create_scene`.
- Add all entities before `scene.build()`.
- Use primitive morphs such as `gs.morphs.Box(size=(...), pos=(...), fixed=True/False)`,
  `gs.morphs.Sphere(radius=..., pos=(...))`, and `gs.morphs.Cylinder(radius=..., height=..., pos=(...))`.
- Position samples can use `entity.get_pos()`, converting tensors with detach/cpu/numpy/tolist if needed.
- Use Planner-dispatched SimDebug cards for IPC coupling, generated-mesh manifest use, asset restrictions, and physical
  control restrictions.
""".strip()


FEM_IPC_API_GUIDE = """
Genesis FEM+IPC primitive API constraints:
- `create_scene` receives `deformable_cfg`. Use `gs.init(..., precision=deformable_cfg["genesis_precision"], ...)`.
- Scene setup should pass runtime timing through `gs.options.SimOptions(dt=sim_dt, substeps=sim_substeps, gravity=...)`.
- FEM primitive soft bodies can use morphs such as `gs.morphs.Box(...)`, `gs.morphs.Sphere(...)`, and
  `gs.morphs.Cylinder(...)` with config-driven tet resolution.
- FEM state can be sampled with `entity.get_state()` when needed; convert tensors before writing JSON.
- Use Planner-dispatched SimDebug cards for FEM/IPC scope, material parameters, IPC option mapping, initial geometry,
  state metrics, contact/coupling, and soft-body validity restrictions.
""".strip()
