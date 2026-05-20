"""Planner prompt clauses."""

from code_agent.prompts.common import (
    BUILTIN_ASSET_POLICY_GUIDE,
    GENERATED_RESULT_QUALITY_GUIDE,
    PHYSICAL_CAUSALITY_CONTRACT,
    SCALE_POLICY_GUIDE,
    SOURCE_AWARE_REPAIR_GUIDE,
)
from code_agent.prompts.ipc import (
    FEM_MATERIAL_SELECTION_GUIDE,
    IPC_FAILURE_DIAGNOSTIC_GUIDE,
    RIGID_IPC_COUPLING_GUIDE,
)

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
{BUILTIN_ASSET_POLICY_GUIDE}
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
    opt_enabled: bool,
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
  one self-contained MJCF body tree with joints and actuators, add an XML/MJCF asset request with
  asset_type=`generated_xml` or `mjcf`. Describe the required joints, actuator semantics, base behavior, approximate
  dimensions, and control affordances in purpose/simulation_role. XML/MJCF assets may use primitive geoms or generated
  case-workspace mesh files, but must never reference Genesis built-in meshes, external mesh downloads, hfields, or
  texture files. Do not use XML/MJCF only for textured decorative mesh objects.
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
- run_opt: invoke the dedicated Opt Codex subagent for a generated case that appears parameter-limited or
  control-sensitive. Current CONFIGS.opt.enabled={opt_enabled}. If false, do not choose this action. If true, use it
  after integration and preferably after execution/critic evidence shows the case is runnable: code runs, video/metrics
  exist, required entities are present, and the physical causal story is basically plausible, but the target is missed.
  Good Opt candidates have continuous measurable residuals such as distance, speed, angle, timing, pose, friction,
  damping, gain, material/contact, or solver/contact sensitivity, and the generated action/body/scene code exposes real
  control handles that Opt can safely parameterize. Bad Opt candidates are structural failures: missing entities,
  invalid assets, broken imports, impossible geometry, wrong joint axes, caged/stuck objects, missing metrics, invisible
  rendering, or task semantics that need rewriting. If the same owner has already received one or two local repairs and
  the remaining failure is still "behavior is close but off", prefer run_opt before exhausting all repair rounds. The
  Opt subagent receives the case workspace, timing, backend, success criteria, allowed edit scope, and current reports;
  it decides what parameters to expose, which bounds/sigma/strategy to use, whether to run CMA-ES, and whether to return
  success, needs_more_optimization, needs_rewrite, or failed. After a useful Opt result, choose run_execution so the
  normal root artifacts and critic evidence reflect the selected best parameters.
- request_repair: send `repair_brief` to the owning worker when critic/execution evidence shows a fix.
  {SOURCE_AWARE_REPAIR_GUIDE}
- run_python: optional controlled `uv run --no-sync python ...` command. Use `python_args` and cwd repo/case.
- run_pytest: optional controlled `uv run --no-sync pytest ...` command. Use `pytest_args` and cwd repo/case.
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
  body geometry, joints, or actuators were underspecified or wrong, choose start_xml_assets for the affected
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
- If CONFIGS.opt.enabled is true, use this generic Opt decision policy:
  if execution failed, assets are invalid, imports are broken, metrics are absent, or the scene cannot render the
  relevant behavior, choose repair/regeneration instead of Opt.
  elif critic failed and required entities/control handles exist and the miss has a continuous measurable residual,
  choose run_opt before repeated source repair.
  elif critic failed and the evidence is structural, choose request_repair or asset regeneration.
  elif Opt returns success, partial_success, or needs_more_optimization with a useful best, choose run_execution next.
  elif Opt returns needs_rewrite, route repair/regeneration using Opt's diagnosis.
  Opt candidates include missed grasp/release timing, close target misses, plausible-but-unstable contact,
  material/friction/density/damping sensitivity, target pose offsets, controller gain sensitivity, or solver/contact
  tolerance sensitivity. Do not choose run_opt for missing entities, invalid assets, broken imports, impossible
  geometry, caged/stuck mechanisms, absent metrics, invisible target behavior, or cases where task semantics need to be
  rewritten.
- After run_opt returns success, partial_success, or needs_more_optimization, choose run_execution next so
  artifacts/metrics/render and reports/execution_report.json are regenerated from the selected current opt params.
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
