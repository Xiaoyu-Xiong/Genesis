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


SCALE_POLICY_GUIDE = """
Scale policy:
- Avoid non-uniform scale by default. Unless the input task prompt explicitly requests an exceptional anisotropic
  scaling operation, do not use per-axis scale values such as `scale=(sx, sy, sz)` with unequal components for meshes,
  primitives, imported assets, MJCF/XML assets, or generated asset manifest requests.
- When an object needs a long, flat, thick, thin, or otherwise non-cubic shape, model that shape directly with primitive
  dimensions (`size`, `radius`, `height`), generated-asset geometry, XML primitive geom sizes, or a regenerated asset
  with the intended proportions. Do not create the intended proportions by stretching an already generated/imported
  mesh with non-uniform scale.
- Uniform scalar scale is acceptable for unit conversion, global sizing, and layout fitting. If a rare task genuinely
  requires non-uniform scale, document the reason in the plan or worker report and keep it isolated to that explicit
  requirement.
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


RENDER_CLARITY_GUIDE = """
Rendering clarity requirement:
- Generated videos must clearly show the task-relevant scene, objects, contacts, and motion. Camera position, lookat,
  field of view, clipping planes, lighting, background, capture cadence, and resolution should be chosen so the whole
  requested behavior is readable without severe cropping, tiny objects, occlusion, blank frames, or confusing views.
- If critic or visual evidence says the result is hard to inspect because the camera is too far, too close, poorly
  aimed, poorly lit, cropped, static when tracking is needed, or otherwise unclear, actively repair the rendering module
  and camera parameters. Do not treat unclear rendering as acceptable when the simulation behavior cannot be judged.
- Record final camera parameters and rendering choices in render_stats.json so repairs can be source-aware.
""".strip()


GENERATED_RESULT_QUALITY_GUIDE = f"""
The final simulation should not merely satisfy numeric proxies. It should match the input text prompt and look
physically and visually reasonable, coherent, and logically staged.
{RENDER_CLARITY_GUIDE}
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
If the evidence points to a generated XML/MJCF/URDF asset itself being unsuitable for the requested mechanism
(for example a gripper without a real opposing thumb, fingers that cannot form an enclosing cage, missing useful
actuator affordances, wrong joint axes, invalid link hierarchy, or a body tree whose primitive geometry cannot perform
the requested contact task), route the fix to Planner/XML asset regeneration. Do not keep assigning action/body repairs
when the action is only failing because the generated articulated asset cannot physically do the job.
""".strip()


PLANNER_GENERAL_RULES = f"""
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
{SCALE_POLICY_GUIDE}
""".strip()


def planner_fem_ipc_capability_section(
    *,
    deformable_enabled: bool,
    ipc_enabled: bool,
    deformable_config_path: object,
    deformable_config_text: str,
) -> str:
    return f"""
FEM/IPC capability:
- FEM deformable enabled: {deformable_enabled}
- IPC contact/coupling enabled: {ipc_enabled} (forced on whenever FEM deformable is enabled)
- Effective config contract: {deformable_config_path}
- Effective config values:
{deformable_config_text}
- If FEM deformable is false and the task fundamentally requires soft-body, jelly, elastic, FEM, or visible
  deformation behavior, choose finish with verdict inconclusive. Do not write a rigid-body substitute.
- If FEM deformable is true and the task requires soft-body behavior, use FEM+IPC only. Do not use MPM, PBD, SPH,
  cloth-only shortcuts, or rigid-only substitutes.
- If FEM deformable is false but IPC is true, rigid/articulated contact scenes may use IPC. Tell workers to keep bodies
  rigid/articulated, configure `gs.options.IPCCouplerOptions`, and use rigid IPC coupling materials for contact-heavy
  rigid behavior.
  Use this coupling guide when choosing object roles and body/action contracts:
  {RIGID_IPC_COUPLING_GUIDE}
  Use this IPC failure diagnostic guide when interpreting execution logs:
  {IPC_FAILURE_DIAGNOSTIC_GUIDE}
- If IPC is false, workers must not instantiate `gs.options.IPCCouplerOptions`.
- All FEM `E`/`nu`/`rho` material-range defaults, IPC, tet, and precision defaults must come from `deformable_cfg` /
  contracts/deformable_config.json in generated code. FEM `friction_mu` is intentionally not a config hyperparameter:
  generated body code must choose explicit task-appropriate FEM friction values instead of reading a
  `deformable_cfg["fem_friction_mu"]` override.
  `ipc_contact_d_hat_adaptive` is a code-agent runtime switch: when true, the generated entrypoint computes
  `clamp(min(0.2 * median_mesh_edge_or_bbox_feature), 1e-5 * max_bbox_diag, 2e-3 * max_bbox_diag)` at run time, where
  `max_bbox_diag` is the largest bbox diagonal among all adaptive candidates. It considers generated meshes,
  user/repo-provided meshes, primitive-derived meshes, collision-enabled MJCF/XML primitive geoms, direct
  `gs.morphs.Box/Cylinder/Sphere` primitives written in `src/body.py` or `src/scene.py`, and bbox-only primitive/XML/URDF
  assets that are present in `assets/asset_manifest.json`, with `contracts/planner_output.json` asset_requests as a bbox
  fallback for assets that have no manifest entry yet.
- FEM elastic material choices must include explicit `E`, `nu`, and `rho` values selected from the `deformable_cfg`
  ranges and defaults. Use this material guide when instructing body/action/critic work:
  {FEM_MATERIAL_SELECTION_GUIDE}
""".strip()


def planner_available_actions_section(
    *,
    sim_dt: float,
    sim_substeps: int,
    render_every_n_steps: int,
    render_fps: int,
    render_res: tuple[int, int],
) -> str:
    return f"""
Available actions:
- write_plan: create planner_output for this case. Include a complete `planner_output` object matching
  planner_output.schema.json. Infer duration from the task yourself. Use sim_dt={sim_dt},
  sim_substeps={sim_substeps}, render_fps={render_fps}, render_every_n_steps={render_every_n_steps}, and
  render_res={render_res} unless the task explicitly requires different values. Use mode local_gpu and backend gpu by
  default.
  Make the plan detailed enough that each writer can implement its part without guessing: describe desired layout,
  entity identities, physical roles, timing, camera/render expectations, asset orientation/texture needs, success
  criteria, likely failure modes, and per-module validation expectations.
  For objects that genuinely require generated geometry, add `asset_requests` with asset_type=`generated_mesh` and set
  dispatch_graph.wait_for_asset_manifest=true when writers should wait for generated mesh paths before writing code.
  In each asset request, `bbox` is the approximate positive XYZ size in meters used to guide mesh generation and
  downstream size checks; do not use it for position, lower bounds, centers, or signed extents. `scale` is an optional
  runtime uniform scale factor only: use null unless you have a specific scalar factor, and if you use an array it must
  have three equal positive components. Do not put target dimensions in `scale`; use `bbox` for approximate dimensions.
  The mesh asset pipeline may compute a scalar uniform runtime scale from the generated mesh bbox and requested `bbox`,
  then write that fixed factor to assets/asset_manifest.json for workers to pass to Genesis. If a ready generated mesh's
  geometry is acceptable and only its runtime size metadata is wrong, use update_mesh_asset_metadata with a complete
  revised planner_output instead of retrying mesh generation.
  {SCALE_POLICY_GUIDE}
  Meshy mesh generation accepts at most 800 characters in the final mesh-agent prompt. Keep each generated_mesh request
  concise enough that the assembled prompt from name, purpose, simulation_role, dimensions, texture_needs, and the
  automatic simulation-ready geometry suffix stays within that limit.
  When mesh feedback requires a different prompt, rewrite the affected asset request into a new concise, integrated
  description instead of appending repair text. Put the revised complete planner_output directly on the
  start_mesh_assets action that retries generation; action rationale, notes, and repair_brief are not sent to the mesh
  agent.
  Mesh asset generation owns mesh validity. If a generated mesh's manifold, texture transfer, or Genesis FEM import
  validation fails, regenerate that asset through the mesh asset action; do not ask body/scene workers to rewrite or
  procedurally repair the mesh geometry.
  For articulated robots, grippers, gates, latches, actuated mechanisms, or any task object that is best represented as
  one self-contained primitive MJCF body tree with joints and actuators, add an XML/MJCF asset request with
  asset_type=`generated_xml` or `mjcf`. Describe the required joints, actuator semantics, base behavior, approximate
  dimensions, and control affordances in purpose/simulation_role. XML/MJCF asset requests are primitive-geom only; do
  not use them for textured decorative mesh objects.
  When the plan uses XML/MJCF actuators, make the body/action contracts explicit: body must expose stable actuator
  names, joint names, DOF groups, or control handles in `actors`, and action must drive those handles with Genesis
  actuator/DOF/force control APIs after initialization. Do not ask action to create motion for an XML articulated asset
  by overwriting root pose, qpos, or velocities during the simulation. If the actuator contract is missing, the
  generated code should fail clearly so critic can assign a source-aware body/action repair.
  When XML/MJCF asset feedback requires a different asset prompt or actuator/body-tree specification, rewrite the
  affected XML asset request in a revised complete planner_output and put it directly on the start_xml_assets action
  that retries generation. Do not rely on rationale, notes, or repair_brief to reach the XML asset worker.
  Across all task types, direct state writes such as setting entity pose, root qpos, DOF position, or DOF velocity are
  initialization-only. After stepping begins, motion should be expressed through physically meaningful controls:
  actuator commands, DOF controllers, motors, external forces/torques, or initial velocities set before the first step.
  {PHYSICAL_CAUSALITY_CONTRACT}
  Any object whose requested appearance depends on texture, patterned surface detail, decorative material variation,
  image-like surface content, or nontrivial visual ornamentation must be represented by a Meshy-generated asset request,
  even when the task does not explicitly say "mesh". Do not ask writers to fake those textured objects with plain
  primitive colors or simple Genesis surfaces.
  Prefer a dispatch_graph that enables the code-writing workers to run together after required assets are ready. Treat
  scene, body, action, and rendering as parallel-capable by default when their contracts contain enough shared
  layout/entity/timing detail; add serial edges only for concrete dependencies that truly require seeing another
  worker's generated source or report.
  Module contract required exports must match the current implementation interfaces exactly:
  scene=`create_scene(backend, *, sim_dt, sim_substeps, deformable_cfg)`;
  body=`create_bodies(scene, task, *, deformable_cfg)`;
  action=`run_actions(scene, actors, *, out_dir, steps, render_state=None)`;
  rendering=`setup_rendering(..., render_every_n_steps, render_res)`, `capture_frame`, and `finalize_rendering`.
- start_mesh_assets: start Planner-requested generated mesh assets in the background and return immediately. Use
  `asset_names` to restrict generation to specific asset request names, or null/[] to generate all generated_mesh
  requests. Prefer this over blocking generation when any writer can make progress without the final manifest. If this
  is a retry that changes the mesh prompt, include a complete revised `planner_output` in this action; preserve the
  rest of the plan and rewrite only the affected asset_requests.
- wait_mesh_assets: wait for a previously started background mesh asset job to finish and validate
  assets/asset_manifest.json.
- update_mesh_asset_metadata: synchronously update ready generated mesh manifest metadata without modifying or
  regenerating mesh files. Use this when inspection/execution shows that mesh geometry is acceptable but runtime
  `scale`/`bbox` metadata is wrong. Include a complete revised `planner_output` whose affected generated_mesh
  asset_requests have the corrected uniform `scale` or `bbox`, and use `asset_names` to restrict the update to those
  assets. Do not use this for changed mesh prompt, purpose, texture, topology, or simulation role; use start_mesh_assets
  for those.
- start_xml_assets: start Planner-requested XML/MJCF assets in the background and return immediately. Use `asset_names`
  to restrict generation to specific XML/MJCF asset request names, or null/[] to generate all XML/MJCF requests. XML
  asset workers are parallel-capable by default, and this asset job may overlap with mesh asset jobs and code-writing
  workers that do not need the manifest yet. If this is a retry that changes the XML prompt or actuator/body-tree spec,
  include a complete revised `planner_output` in this action; preserve the rest of the plan and rewrite only the
  affected asset_requests.
- wait_xml_assets: wait for a previously started background XML/MJCF asset job to finish and merge its partial manifest
  into assets/asset_manifest.json.
- inspect_assets: render/inspect selected ready mesh/XML assets from assets/asset_manifest.json and write
  reports/asset_inspection_report.json plus preview images. Use `asset_names` to inspect specific logical assets, or
  null/[] to inspect every manifest entry. This is a diagnostic action for deciding whether a geometry/contact failure
  is caused by body placement/choreography or by unsuitable generated asset geometry/topology/scale.
- spawn_workers: start one or more generation workers. Use `roles` from scene, body, action, rendering. Roles in a
  single spawn_workers action are dispatched concurrently by the harness with no default cap beyond the number of
  requested roles. Prefer maximal safe parallelism: after required assets are ready, usually spawn every missing writer
  role together, because each worker can read planner_output, asset_manifest, repository code, and the case workspace.
  Split dependent work across multiple Planner turns only when you can identify a concrete dependency that would make
  parallel writing likely incorrect. If mesh or XML assets are still running, you may still spawn writer roles whose
  module_contracts do not list asset_dependencies or asset_manifest input dependencies. Wait for the relevant asset jobs
  only before spawning roles that need canonical generated mesh or XML paths.
- run_integrator: wire generated modules into src/main.py.
- run_execution: run generated code through the harness on the local GPU.
- run_critic: ask the read-only critic to evaluate execution artifacts.
- request_repair: send `repair_brief` to the owning worker when critic/execution evidence shows a fix.
  {SOURCE_AWARE_REPAIR_GUIDE}
- run_python: optional controlled `uv run python ...` command. Use `python_args` and cwd repo/case.
- run_pytest: optional controlled `uv run pytest ...` command. Use `pytest_args` and cwd repo/case.
- finish: end the episode with verdict pass, fail, or inconclusive.
""".strip()


PLANNER_ACTION_POLICY_GUIDE = f"""
Action policy:
- If `planner_output_path` is null, choose write_plan.
- If planner_output dispatch_graph.wait_for_asset_manifest is true and assets.status is not_requested, start all
  required asset families. Use start_mesh_assets for generated_mesh requests and start_xml_assets for generated_xml/mjcf
  requests. You can start one family in one Planner turn and the other in the next while the first continues in the
  background.
- If one asset family is already running but another required family is absent from assets.jobs, start the absent family
  before waiting, so independent mesh and XML work can overlap.
- If assets.status is running, choose spawn_workers for non-asset-dependent missing roles, or wait_mesh_assets /
  wait_xml_assets when the next useful writer/integration step requires the manifest.
- If any generation worker is missing or failed, choose spawn_workers or request_repair for the relevant owner.
- If assets/asset_manifest.json or reports/asset_generation_report.json show a generated_mesh entry with status failed,
  failed manifold validation, failed Genesis FEM import validation, missing/corrupt texture, or an unsuitable generated
  topology, choose start_mesh_assets for the affected asset_names with a complete revised planner_output whose affected
  generated_mesh request has been rewritten to incorporate the failure feedback under the Meshy prompt limit. Do not
  request body/scene/action/rendering repair for mesh-intrinsic defects; those workers should only fix placement,
  material use, controls, or rendering around ready assets.
- If asset inspection or execution shows a ready generated_mesh has acceptable geometry but the runtime scale/bbox
  metadata is wrong, choose update_mesh_asset_metadata for the affected asset_names with a complete revised
  planner_output whose affected generated_mesh request has the corrected uniform scale/bbox. Do not use
  start_mesh_assets for metadata-only sizing fixes.
- If reports/xml_asset_generation_report.json shows a generated_xml/mjcf request failed because the asset prompt,
  primitive body tree, joints, or actuators were underspecified or wrong, choose start_xml_assets for the affected
  asset_names with a complete revised planner_output whose affected XML/MJCF request integrates the concrete feedback.
- Geometry failures such as initial intersections, invalid IPC worlds, impossible clearances, or repeated contact
  explosions are not always choreography/body-placement bugs. They can also come from generated mesh defects such as a
  filled hole, wrong topology, oversized scale, inverted volume, unintended extra components, or a link/anchor shape
  that cannot physically interlock as planned. When the evidence is ambiguous and assets are ready, choose
  inspect_assets for the suspicious asset_names before repeatedly asking body to repair placement. If inspection points
  to a mesh scale/bbox metadata issue with otherwise acceptable geometry, use update_mesh_asset_metadata. If inspection
  points to a mesh/XML intrinsic geometry issue, retry the affected asset family with a rewritten complete
  planner_output; if inspection shows assets are usable, route the repair to body/action/scene as appropriate.
- To improve speed, prefer grouping all currently missing writer roles into one spawn_workers action. Keep dependencies
  serial only when a specific worker must inspect another worker's completed source/report before it can write a
  correct module.
- Only choose run_integrator after scene/body/action/rendering are all ok.
- Only choose run_execution after integration is current.
- Only choose run_critic after execution is current.
- Only choose finish pass after the latest critic verdict is pass.
- If the latest critic verdict is inconclusive because `codex_result.error_type` is `codex_usage_limit`, choose finish
  with verdict inconclusive; do not request code repair from a missing/blocked critic review.
- If critic fails and repair budget remains, choose request_repair for the most relevant source owner unless the critic
  routes the defect to Planner-owned asset regeneration. When critic recommends `planner` because a generated mesh or
  XML/MJCF asset is intrinsically wrong for the task, revise the affected asset request in a complete planner_output and
  choose start_mesh_assets/start_xml_assets for the affected asset_names. Do not call request_repair with owner
  `planner`; Planner repairs asset prompts through the asset actions.
- If deterministic artifact checks, stderr, or stdout mention `ipc.initial_penetration`, libuipc initial
  penetration/intersection/thickness/distance/sanity-check failure, first decide whether the concrete evidence points
  to asset geometry/topology/scale or to body placement/clearance. Use inspect_assets when a ready generated asset could
  be the source of the geometry error. Choose update_mesh_asset_metadata when evidence points only to generated mesh
  scale/bbox metadata. Choose start_mesh_assets/start_xml_assets with a revised complete planner_output for intrinsic
  asset defects, or request_repair for `body` when the assets are plausible and the issue is initial placement, spacing,
  orientation, duplicate contact geometry, or choreography. Treat this as geometry repair work, not an
  execution-environment issue, unless the logs clearly show a missing dependency or runtime setup failure.
  If this appears together with `IPC rigid state accessor feature is unavailable...`, treat the accessor message as
  secondary to the invalid IPC world unless the same accessor failure is reproduced without any initial-geometry or
  `World is not valid` diagnostics.
- Prefer run_execution over generic run_python for generated simulations.
- {GENERATED_RESULT_QUALITY_GUIDE}
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


CRITIC_ASSET_EVALUATION_GUIDE = """
Asset and mechanism evaluation:
- Treat generated assets as first-class evidence, not as trusted black boxes. Inspect asset manifests, generated XML/MJCF
  source, worker reports, preview images, and in-scene contact sheets when the task depends on a generated mesh,
  articulated hand/gripper, robot, mechanism, or visually specific object.
- Judge whether the asset's shape, topology, joint axes, actuator contract, scale, and visible geometry can plausibly
  perform the requested task before blaming only action timing. For grasping/manipulation tasks, explicitly check
  whether fingers and thumb can form a real opposing cage around the payload, whether the payload can fit inside that
  grasp volume, and whether closing the actuators would trap rather than sweep the payload away.
- If a simulation fails because the generated asset is morphologically unsuitable, visually wrong, lacks required
  joints/actuators, has the wrong scale/orientation/topology, or cannot provide the requested contact affordance, set
  `recommended_owner` to `planner`. In `repair_summary`, tell Planner to rewrite the affected asset request and rerun
  start_xml_assets/start_mesh_assets with concrete asset-level requirements. Do not keep recommending action/body
  repairs just because metrics report no lift, no squeeze, or lateral escape when the visible/source evidence shows the
  end-effector geometry cannot physically contain or manipulate the object.
- If the generated asset is plausible but its scene placement, initial clearance, material/coupling, or code-side
  metadata is wrong, route to body/scene. If the asset and placement are plausible but the timing, controller, gates, or
  release sequence are wrong, route to action.
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
- Choose explicit `friction_mu` values for each FEM material. Do not read FEM friction from `deformable_cfg`. Around
  `0.0` to `0.1` is slippery or nearly frictionless contact, `0.2` to `0.5` is typical soft rubber/plastic contact, and
  `0.6` to `1.0` is high-friction sticky contact; use values outside that range only when the task clearly justifies it.
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
- When `ipc_enable_rigid_rigid_contact` / `enable_rigid_rigid_contact` is true for a heavy rigid-contact scene, treat
  the scene as pure IPC rigid-rigid contact: do not create any rigid body with
  `coup_type="two_way_soft_constraint"`. Use `coup_type="ipc_only"` for free passive rigid bodies whose motion comes
  from gravity/contact/friction/interlock, and use `coup_type="external_articulation"` for actively driven bodies that
  must also contact IPC-owned rigid bodies.
- Do not rely on Genesis rigid contact as a fallback for `ipc_only` objects. Genesis's rigid collider skips pairs
  involving `ipc_only` links; rigid-rigid contact between such bodies must be handled by IPC. Avoid mixing
  `two_way_soft_constraint` and `ipc_only` in heavy interlocking contact because the soft transform constraint and
  Genesis/IPC state synchronization can create inconsistent states, sudden ejection, or crashes.
- Passive IPC rigid bodies must not be pose-written, velocity-written, force-driven, hidden-welded, or directly
  attached after initialization. Drive only the intended articulation DOFs of active mechanisms, not the passive rigid
  payloads they contact.
- `coup_links=(...)` is only for `two_way_soft_constraint`; use it to couple just the links that contact the task
  object, such as left/right gripper fingers, instead of putting an entire robot into IPC.
- If `coup_type` is left as `None`, Genesis auto-selects a mode based on entity type, but generated code should choose
  an explicit mode when the task's contact behavior depends on it.
""".strip()


EXTERNAL_ARTICULATION_MJCF_GUIDE = """
External-articulation MJCF/XML guide for IPC:
- An MJCF/URDF entity used with `coup_type="external_articulation"` must be a fixed-base articulation, with a fixed
  parent/body and a revolute or prismatic child/body for each driven mechanism.
- Every link that participates in an external-articulation joint must have collision geometry. An empty logical parent
  body can make IPC fail to create an ABD slot for that link.
- If the fixed parent is only a logical mount, add a tiny dummy collision geometry to that parent. The dummy parent geom
  does not have to be a mesh; a primitive MJCF geom such as `type="box"` is fine.
- The dummy geom must be a real nonzero-volume collision geom, not a site, inertial, visual-only marker, zero-area
  plane, line, or empty body. It must participate in collision semantics: do not set both `contype` and `conaffinity`
  to zero. A useful pattern is `contype="1"` with `conaffinity="0"` so the geom remains a collision geom without
  accepting ordinary contact pairs.
- Place dummy mount geometry far from the real contact region so it cannot initially intersect the active mechanism,
  passive rigid bodies, ground, chain links, anchors, or other task geometry. It may be invisible/transparent, but it
  must still import as collision geometry.
- Child/driven geoms, such as a spool or hinge tool, must have real collision geometry; do not provide only visual
  geometry for IPC contact participants.
- Keep XML mesh coordinates aligned with the layout/source mesh coordinates. If a fixed `euler` correction is required
  on a geom, validate the resulting initial configuration with an IPC sanity or distance check before stepping.
- Set joint axes and joint positions from the layout's physical drive axis and pivot. For a spool/anchor-chain style
  scene, the hinge axis should match the spool's physical shaft axis, and the joint position should be at the shaft
  center rather than at a flange or arbitrary bbox point.
- Drive external articulations through actuator/DOF controllers such as `control_dofs_position_velocity`, not by
  forcibly writing DOF velocity with `set_dofs_velocity` during the simulation.
- Cap PD gains and actuator force ranges in heavy IPC contact scenes. Overly strong controllers can cause large
  per-step joint-angle jumps that look like penetration and can destabilize contact.
- For diagnostics, do not trust `get_dofs_velocity()` alone on external articulations whose qpos is recovered from IPC.
  Estimate true angular velocity from step-to-step hinge angle differences when needed.
""".strip()


IPC_FAILURE_DIAGNOSTIC_GUIDE = """
IPC failure diagnostic guide:
- If stdout/stderr show libuipc/UIPC initial-geometry diagnostics such as `SimplicialSurfaceIntersectionCheck`,
  `SimplicialSurfaceDistanceCheck`, thickness/distance/barrier failures, or `World is not valid`, and the later Python
  exception says `IPC rigid state accessor feature is unavailable... requires rigid ABD state retrieval`, treat the
  accessor exception as a downstream/secondary diagnostic from the invalid IPC world. Do not infer from that pattern
  alone that the local libuipc build lacks rigid ABD accessor support.
- For that combined pattern, route the repair to the source that owns initial placement, scale, spacing, orientation,
  mesh clearance, duplicate IPC contact geometry, or generated asset topology. This is often `body`, but it may be a
  generated mesh/XML asset when the asset has a filled hole, wrong scale, extra component, inverted/invalid volume, or a
  shape that cannot physically interlock as requested. Preserve the intended IPC contact/coupling model; do not "fix"
  it by disabling IPC, setting `needs_coup=False`, changing the mechanism to hidden constraints, or bypassing contact
  unless a clean no-penetration repro still proves IPC capability is missing.
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
FEM material/entity construction, explicit `E`, `nu`, `rho`, and agent-selected `friction_mu` material choices, with
`E`/`nu`/`rho` within the deformable config bounds, config-driven FEM/IPC/tet parameters, plausible deformation metrics,
and video or event evidence of wobble, compression, collapse, bending, or other requested deformation. Fail rigid-only
substitutes, MPM/PBD/SPH implementations, missing `E`/`nu`/`rho` or missing `friction_mu` on FEM elastic materials,
hardcoded FEM/IPC defaults, attempts to read `deformable_cfg["fem_friction_mu"]`, or mid-simulation FEM
position/velocity writes that fake the deformation.
{FEM_MATERIAL_SELECTION_GUIDE}
""".strip()


CRITIC_DECISION_GUIDE = f"""
Decide whether the run passes as a generated Genesis simulation result. Compare the original task prompt, generated
source, execution artifacts, metrics, event logs, render stats, and visual evidence. Prioritize execution correctness,
required artifacts, plausible movement, physically coherent staging, and whether the visual evidence matches the task.
{GENERATED_RESULT_QUALITY_GUIDE}
{CRITIC_ASSET_EVALUATION_GUIDE}
{PHYSICAL_CONTROL_METHOD_GUIDE}
{PHYSICAL_CAUSALITY_CRITIC_GUIDE}
{DEFORMABLE_CRITIC_GUIDE}
{IPC_FAILURE_DIAGNOSTIC_GUIDE}
{SCALE_POLICY_GUIDE}
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
