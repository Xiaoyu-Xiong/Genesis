"""Writer-worker prompt clauses."""

from code_agent.prompts.common import PHYSICAL_CAUSALITY_CONTRACT, SCALE_POLICY_GUIDE
from code_agent.prompts.ipc import (
    EXTERNAL_ARTICULATION_MJCF_GUIDE,
    FEM_MATERIAL_SELECTION_GUIDE,
    RIGID_IPC_COUPLING_GUIDE,
)

WORKER_COMMON_RULES = f"""
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
{PHYSICAL_CAUSALITY_CONTRACT}
{SCALE_POLICY_GUIDE}
""".strip()


RIGID_API_GUIDE = f"""
Genesis rigid primitive API constraints:
- Import Genesis as `import genesis as gs`.
- Initialize with `gs.init(backend=gs.cpu if backend == "cpu" else gs.gpu, precision="32",
  performance_mode=True, logging_level="warning")`.
- Create a scene with `gs.Scene(sim_options=gs.options.SimOptions(dt=sim_dt, substeps=sim_substeps), ...,
  show_viewer=False, show_FPS=False)` using the `sim_dt` and `sim_substeps` arguments passed into `create_scene`.
- Add a ground plane with `scene.add_entity(gs.morphs.Plane())`.
- If `deformable_cfg["ipc_enabled"]` is true for a rigid/articulated scene, route contact through IPC by constructing
  `gs.Scene(..., coupler_options=gs.options.IPCCouplerOptions(...))` and mapping IPC values from `deformable_cfg`.
  Strip the config prefix when passing Genesis options, e.g. `ipc_contact_d_hat` -> `contact_d_hat` and
  `ipc_constraint_strength_translation` -> `constraint_strength_translation`.
  Do not pass `ipc_contact_d_hat_adaptive` to Genesis; it is a code-agent runtime switch, and the generated entrypoint
  resolves it into `ipc_contact_d_hat` before `create_scene`.
{RIGID_IPC_COUPLING_GUIDE}
{EXTERNAL_ARTICULATION_MJCF_GUIDE}
- For rigid IPC scenes, set `enable_rigid_rigid_contact` from `deformable_cfg["ipc_enable_rigid_rigid_contact"]`.
  Keep it true when rigid bodies should collide with each other through IPC; keep it false when IPC should only handle
  rigid-soft contact or selected articulated contacts and Genesis's rigid solver should avoid delegated pairs.
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
  If a generated mesh manifest entry is missing, failed, invalid, or cannot be imported, fail clearly and report that
  the Planner should regenerate the mesh asset; do not edit mesh files or approximate the object with ad hoc primitives.
""".strip()


FEM_IPC_API_GUIDE = f"""
Genesis FEM+IPC primitive API constraints:
- FEM materials/entities are allowed only when `deformable_cfg["enabled"]` is true. If it is false, do not instantiate
  FEM materials/entities or deformation-only APIs.
- `gs.options.IPCCouplerOptions` is allowed when `deformable_cfg["ipc_enabled"]` is true. FEM deformable scenes force
  IPC on; rigid-only scenes may also use IPC when this flag is true.
- If `deformable_cfg["ipc_enabled"]` is false, do not instantiate `gs.options.IPCCouplerOptions`.
- Deformable scenes must stay in the FEM+IPC family. Do not use MPM, PBD, SPH, cloth-specific shortcuts, or rigid-only
  substitutes for soft-body prompts.
- `create_scene` receives `deformable_cfg`. Use `gs.init(..., precision=deformable_cfg["genesis_precision"], ...)`.
- Scene setup should pass runtime timing through `gs.options.SimOptions(dt=sim_dt, substeps=sim_substeps, gravity=...)`.
- When FEM bodies should collide through IPC, construct `gs.Scene(..., coupler_options=gs.options.IPCCouplerOptions(...))`.
  Map IPC values from `deformable_cfg`, stripping the `ipc_` config prefix before passing Genesis option names:
  `ipc_newton_*`, `ipc_n_linesearch_iterations`,
  `ipc_linesearch_report_energy`, `ipc_linear_system_*`, `ipc_contact_*`, `ipc_collision_detection_method`,
  `ipc_cfl_enable`, `ipc_sanity_check_enable`, `ipc_constraint_strength_translation`,
  `ipc_constraint_strength_rotation`, `ipc_enable_rigid_ground_contact`, `ipc_enable_rigid_rigid_contact`,
  `ipc_two_way_coupling`, `ipc_enable_rigid_dofs_sync`, and `ipc_free_base_driven_by_ipc`.
  Exclude `ipc_contact_d_hat_adaptive`; it is a code-agent runtime switch, not a Genesis option.
- Use exactly one IPC ground/support surface for the same contact region. If scene.py creates a floor, store it as
  `scene.genesis_static_floor`; body.py should reference that entity in actors instead of adding another coincident
  plane. Duplicate overlapping IPC planes are invalid initial geometry.
- For primitive soft bodies, use morphs such as `gs.morphs.Box(...)`, `gs.morphs.Sphere(...)`, and
  `gs.morphs.Cylinder(...)` with `tet_resolution=deformable_cfg["tet_resolution"]`.
- Place FEM primitives with strictly positive initial clearance. For rotated boxes, compute a conservative half extent
  (`0.5 * side * sqrt(3)` is acceptable) and stack centers with a positive gap; do not rely on unrotated `side / 2`
  when the body has nonzero Euler rotation.
{FEM_MATERIAL_SELECTION_GUIDE}
- The same FEM material should also use `model=deformable_cfg["fem_model"]`,
  `hydroelastic_modulus=deformable_cfg["fem_hydroelastic_modulus"]`,
  `contact_resistance=deformable_cfg["fem_contact_resistance"]`, and
  `hessian_invariant=deformable_cfg["fem_hessian_invariant"]`.
  It must also pass an explicit task-appropriate `friction_mu` chosen by body.py; do not read FEM friction from
  `deformable_cfg`.
- Rigid participants in IPC scenes should use `gs.materials.Rigid(...)` with appropriate IPC coupling fields such as
  `coup_type` and `coup_friction`; the generic `deformable_cfg["friction"]` may be used as a rigid coupling fallback
  when relevant.
- FEM initialization may call `entity.set_velocity(...)` before stepping. After stepping starts, do not fake soft-body
  behavior with repeated `set_position` or `set_velocity` writes. Use gravity, contact, IPC coupling, explicit vertex
  constraints, actuator-driven mechanisms, or clearly requested physical force fields.
- FEM state sampling can use `entity.get_state()`; convert `state.pos`/`state.vel` tensors to CPU/numpy/lists before
  writing JSON. Event logs and metrics should include COM, bbox/height, spread, and a simple deformation proxy for
  soft-body tasks.
- `entity.get_state()` for FEM bodies can be expensive. Sample FEM vertex state sparsely for metrics/evidence, not on
  every physics step by default.
""".strip()
