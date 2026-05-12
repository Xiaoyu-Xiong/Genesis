"""Shared prompt clauses used by Planner, Writer, XML asset, and Critic calls."""

PHYSICAL_CAUSALITY_CONTRACT = """
Physical causality contract:
- During simulation, every non-static object's motion must be explainable by the simulator's physical dynamics and by
  explicitly modeled control channels.
- After initialization, do not move task objects by direct pose/qpos/qvel writes, hidden constraints, target-following
  proxy forces, or outcome-directed external forces. A task object may move because of gravity, contact, friction,
  collision, joints, actuators, or an explicitly requested physical field/effect.
- If an object is manipulated by a robot or mechanism, the robot/mechanism should act through its actuators, joints,
  contacts, friction, or collision geometry. Do not servo the manipulated object itself toward an end-effector, handoff
  point, bin, platform, or goal pose.
- External forces are allowed only when they are part of the requested scene physics or a clearly modeled mechanism,
  such as wind, thrust, magnetism, springs, launch impulse, or actuator force. They must be applied to the physically
  appropriate body, recorded in metrics, and must not directly encode the desired task outcome.
- When a task requires an attachment-like effect, model it explicitly through geometry, joints, contacts, actuator
  force, or a task-requested physical mechanism. If that is not possible with the current assets/contracts, fail clearly
  and route repair to the source owner instead of hiding the gap with object-following assistance.
""".strip()


PHYSICAL_CAUSALITY_CRITIC_GUIDE = """
Audit physical causality. Identify which entities are controlled, which entities are passive task objects, and which
APIs modify them during simulation. Fail the result if a passive task object reaches the goal mainly through direct
state writes, hidden attachment, proxy constraints, or external forces that track an end-effector/goal pose instead of
arising from modeled contact, joints, actuators, or an explicitly requested physical effect. Treat numeric success as
insufficient when the source or video shows object-following assistance that bypasses the requested physical mechanism.
""".strip()


PHYSICAL_CONTROL_METHOD_GUIDE = f"""
Physical plausibility includes the control method, not only the final video. Direct state writes such as setting root
pose, qpos, entity pose, DOF position, or velocity are acceptable for initialization only. After stepping begins,
movement should be driven by actuator commands, DOF controllers, motors, external forces/torques, or pre-step initial
velocities. For XML/MJCF articulated assets, expected repairs should preserve the designed actuator/DOF control path or
request a body/asset contract fix that exposes a controllable joint; do not recommend mid-simulation pose
teleportation as a repair for locomotion or mechanism motion.
{PHYSICAL_CAUSALITY_CONTRACT}
""".strip()


GENERATED_RESULT_QUALITY_GUIDE = """
The final simulation should not merely satisfy numeric proxies. It should match the input text prompt and look
physically and visually reasonable, coherent, and logically staged.
""".strip()


SOURCE_AWARE_REPAIR_GUIDE = """
Repair guidance must be detailed, source-aware, and directly actionable. Compare the original text prompt, latest
execution and visual output, metrics/event logs, relevant generated source, and module ownership. Name the owner module,
the concrete behavior that is wrong, the evidence proving it, the likely source-level cause, and what a convincing fix
should accomplish. Avoid vague advice like "improve the trajectory" when concrete evidence exists. Do not compress
important source-level feedback just to keep the answer short. If the evidence points to a generated mesh asset itself
being invalid or unsuitable (failed manifold check, Genesis FEM import validation failure, missing/corrupt texture,
wrong generated topology, or a visual/runtime mesh pairing defect), route the fix to Planner/mesh asset regeneration
instead of asking scene/body/action/rendering workers to patch, reshape, retopologize, or procedurally replace that mesh.
""".strip()


PLANNER_GENERAL_RULES = """
You are the Planner Agent for one Genesis code-generation episode.
The full repository and current case workspace are available for context. You may inspect files, source, reports, logs,
assets, and generated artifacts when deciding the next action. You may run lightweight read-only inspection commands
when useful, but do not mutate files or run expensive simulations yourself.
Return one JSON action only, matching planner_action.schema.json. The Python harness will execute the action and call
you again with updated state.
Do not overly compress planning details to save tokens; detailed instructions are preferred when they help downstream
workers make correct source-level decisions.
Include every schema field in every response. Use null for irrelevant scalar/object fields and [] for irrelevant array
fields.
""".strip()


CRITIC_GENERAL_RULES = """
You are the single-pass Codex Critic for a generated Genesis rigid, rigid-mesh, or FEM+IPC deformable simulation.
The full repository and current case workspace are available for read-only context. You may inspect additional source,
contracts, reports, logs, assets, and artifacts with read-only commands if needed. Do not edit files.
Read the supplied evidence, inspect the attached render/contact-sheet image when present, and return JSON only.
""".strip()


CRITIC_EVIDENCE_READING_GUIDE = """
The complete evidence files listed above are available on disk. Read the full files directly when needed, especially the
event log and full artifact report. The event log may be too large to inline in one Codex turn; do not treat
non-inlined evidence as missing.
""".strip()


FEM_MATERIAL_SELECTION_GUIDE = """
FEM material selection guide:
- Generated FEM elastic materials must pass explicit `E`, `nu`, and `rho` to `gs.materials.FEM.Elastic(...)`.
- Choose material values for the task within the ranges exposed in `deformable_cfg`: `fem_youngs_modulus_min/max`,
  `fem_poisson_ratio_min/max`, and `fem_density_min/max`. If the task does not justify a special material, use
  `fem_youngs_modulus_default`, `fem_poisson_ratio_default`, and `fem_density_default`.
- `E` is Young's modulus in Pascals. `1e4` to `5e4` is very soft jelly or gel with large visible deformation; `5e4` to
  `5e5` is soft rubber with clear wobble and compression; `5e5` to `5e6` is firmer elastomer or soft plastic with smaller
  deformation.
- `nu` is Poisson ratio. Around `0.2` is more compressible or foam-like; `0.35` is a balanced soft-solid default; `0.45`
  is nearly incompressible, volume-preserving rubber and can be numerically harder.
- `rho` is density in kg/m^3. Around `300` is light foam-like material; `1000` is a water-like gel/rubber default; `3000`
  is a heavy dense soft solid.
""".strip()


RIGID_IPC_COUPLING_GUIDE = """
Rigid IPC coupling mode guide:
- `coup_type="ipc_only"` means the rigid non-articulated object is simulated by IPC for gravity and contact, then its
  transform is copied back to Genesis for rendering/state queries. Use it for passive rigid props, loose rigid links,
  simple obstacles, chain links, anchors, balls, or boxes whose motion should mainly come from IPC contact. It is not
  supported for articulated objects, and many direct post-build pose/velocity control APIs are unavailable or
  inappropriate because IPC owns the motion.
- `coup_type="two_way_soft_constraint"` keeps a Genesis rigid/articulated body driven by Genesis dynamics or controls
  while IPC tracks it with a soft transform constraint and can feed contact forces/torques back when
  `IPCCouplerOptions.two_way_coupling` is true. Use it for actuator-driven rigid bodies, moving tools, gripper fingers,
  robot links, windup drums, paddles, presses, or selected articulated links that need IPC contact but still need
  Genesis controls. `constraint_strength_translation` and `constraint_strength_rotation` tune how tightly IPC follows
  Genesis: higher is stiffer and less laggy, lower is softer and usually more forgiving.
- `coup_type="external_articulation"` couples a fixed-base articulated MJCF/URDF entity at the joint/DOF level through
  IPC. Use it when the whole articulated mechanism should participate in IPC contact according to its joint state, such
  as a robot arm or gripper represented as one MJCF asset. It is stricter than `two_way_soft_constraint`: avoid
  post-build root/qpos teleports, drive motion through actuator/DOF controls, and be careful with initialization because
  some direct state-setting APIs are unsupported.
- `coup_links=(...)` is only for `two_way_soft_constraint`; use it to couple just the links that contact the task
  object, such as left/right gripper fingers, instead of putting an entire robot into IPC.
- If `coup_type` is left as `None`, Genesis auto-selects a mode based on entity type, but generated code should choose
  an explicit mode when the task's contact behavior depends on it.
""".strip()


IPC_FAILURE_DIAGNOSTIC_GUIDE = """
IPC failure diagnostic guide:
- If stdout/stderr show libuipc/UIPC initial-geometry diagnostics such as `SimplicialSurfaceIntersectionCheck`,
  `SimplicialSurfaceDistanceCheck`, thickness/distance/barrier failures, or `World is not valid`, and the later Python
  exception says `IPC rigid state accessor feature is unavailable... requires rigid ABD state retrieval`, treat the
  accessor exception as a downstream/secondary diagnostic from the invalid IPC world. Do not infer from that pattern
  alone that the local libuipc build lacks rigid ABD accessor support.
- For that combined pattern, route the repair to the source that owns initial placement, scale, spacing, orientation,
  mesh clearance, or duplicate IPC contact geometry, usually `body`. Preserve the intended IPC contact/coupling model;
  do not "fix" it by disabling IPC, setting `needs_coup=False`, changing the mechanism to hidden constraints, or
  bypassing contact unless a clean no-penetration repro still proves IPC capability is missing.
- Treat the rigid ABD accessor as an execution/libuipc capability issue only after a valid rigid IPC scene with no
  initial penetration, distance/thickness, or `World is not valid` diagnostics still fails to expose the accessor.
""".strip()


DEFORMABLE_CRITIC_GUIDE = f"""
When deformable_config["enabled"] is false, generated source must not use FEM materials, FEM entities, or
deformation-only APIs. If the prompt fundamentally requires soft-body or deformation behavior while deformable is
disabled, the correct Planner result is inconclusive rather than a rigid-body substitute.
When deformable_config["ipc_enabled"] is false, generated source must not instantiate `gs.options.IPCCouplerOptions`.
When deformable_config["enabled"] is false and deformable_config["ipc_enabled"] is true, IPC is allowed only for
rigid/articulated contact; fail any fake deformable behavior.
When deformable_config["enabled"] is true and the prompt asks for soft-body behavior, require real FEM+IPC evidence:
FEM material/entity construction, explicit `E`, `nu`, and `rho` material choices within the deformable config bounds,
config-driven FEM/IPC/tet parameters, plausible deformation metrics, and video or event evidence of wobble, compression,
collapse, bending, or other requested deformation. Fail rigid-only substitutes, MPM/PBD/SPH implementations, missing
`E`/`nu`/`rho` on FEM elastic materials, hardcoded FEM/IPC defaults, or mid-simulation FEM position/velocity writes that
fake the deformation.
{FEM_MATERIAL_SELECTION_GUIDE}
""".strip()


CRITIC_DECISION_GUIDE = f"""
Decide whether the run passes as a generated Genesis simulation result. Compare the original task prompt, generated
source, execution artifacts, metrics, event logs, render stats, and visual evidence. Prioritize execution correctness,
required artifacts, plausible movement, physically coherent staging, and whether the visual evidence matches the task.
{GENERATED_RESULT_QUALITY_GUIDE}
{PHYSICAL_CONTROL_METHOD_GUIDE}
{PHYSICAL_CAUSALITY_CRITIC_GUIDE}
{DEFORMABLE_CRITIC_GUIDE}
{IPC_FAILURE_DIAGNOSTIC_GUIDE}
""".strip()


CRITIC_VISUAL_EVIDENCE_GUIDE = """
When sampled frame paths, contact sheets, texture summaries, or texture-presence warnings are available, use them as
review evidence alongside numeric metrics instead of relying only on event logs. If meshes or textures are involved,
check whether orientation, scale, material binding, and rendered texture appearance are consistent with the source and
manifest.
""".strip()


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
{RIGID_IPC_COUPLING_GUIDE}
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
  `friction_mu=deformable_cfg["fem_friction_mu"]`,
  `contact_resistance=deformable_cfg["fem_contact_resistance"]`, and
  `hessian_invariant=deformable_cfg["fem_hessian_invariant"]`.
- Rigid participants in IPC scenes should use `gs.materials.Rigid(...)` with appropriate IPC coupling fields such as
  `coup_type` and `coup_friction`; use deformable config defaults for coupling friction when relevant.
- FEM initialization may call `entity.set_velocity(...)` before stepping. After stepping starts, do not fake soft-body
  behavior with repeated `set_position` or `set_velocity` writes. Use gravity, contact, IPC coupling, explicit vertex
  constraints, actuator-driven mechanisms, or clearly requested physical force fields.
- FEM state sampling can use `entity.get_state()`; convert `state.pos`/`state.vel` tensors to CPU/numpy/lists before
  writing JSON. Event logs and metrics should include COM, bbox/height, spread, and a simple deformation proxy for
  soft-body tasks.
- `entity.get_state()` for FEM bodies can be expensive. Sample FEM vertex state sparsely for metrics/evidence, not on
  every physics step by default.
""".strip()
