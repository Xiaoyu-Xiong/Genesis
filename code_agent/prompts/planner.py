"""Planner prompt clauses after SimDebug card migration."""

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
Use the Planner-dispatched SimDebug cards as the active source of simulation debugging experience, physical validity
restrictions, asset handling rules, Critic/Opt routing guidance, and repair-routing heuristics.
Use the `simdebug_cards` action field when you want to explicitly dispatch a chosen subset of card ids to downstream
roles. Buckets may include `all`, `workers`/`all_workers`, `scene`, `body`, `action`, `rendering`, `critic`, and `opt`.
Set `simdebug_cards` to null when the harness should send all role/physics-compatible candidates.
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
- Use the Planner-dispatched SimDebug cards for FEM/IPC scope decisions, material policy, IPC config mapping,
  coupling-mode selection, adaptive contact-distance handling, and IPC failure diagnosis.
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
  Make the plan detailed enough that each worker can implement without guessing: describe desired layout, entity
  identities, physical roles, timing, camera/render expectations, asset needs, success criteria, likely failure modes,
  and per-module validation expectations.
  Use SimDebug asset cards when deciding whether to request generated_mesh or XML/MJCF assets, how to rewrite asset
  retries, how to distinguish intrinsic asset defects from metadata-only issues, and how to preserve actuator contracts.
  Prefer a dispatch_graph that enables code-writing workers to run together after required assets are ready. Add serial
  edges only for concrete dependencies that truly require seeing another worker's completed source or report.
  Module contract required exports must match the current implementation interfaces exactly:
  scene=`create_scene(backend, *, sim_dt, sim_substeps, deformable_cfg)`;
  body=`create_bodies(scene, task, *, deformable_cfg)`;
  action=`run_actions(scene, actors, *, out_dir, steps, render_state=None)`;
  rendering=`setup_rendering(..., render_every_n_steps, render_res)`, `capture_frame`, and `finalize_rendering`.
- start_mesh_assets: start Planner-requested generated mesh assets and procedural FEM cloth mesh assets in the
  background and return immediately. Use `asset_names` to restrict generation to specific asset request names, or
  null/[] to generate all generated_mesh and cloth_mesh requests. Use asset_type `cloth_mesh_square`,
  `cloth_mesh_rectangle`, `cloth_mesh_cylinder`, `cloth_mesh_sphere`, or `cloth_mesh` for predefined FEM.Cloth surface
  meshes. Every asset_request must include `cloth_target_edge_length`; set it to null for non-cloth assets or when the
  default is acceptable. For cloth_mesh assets, it is a per-asset target triangle edge length in meters: smaller values
  create denser cloth meshes for folds/contact detail, larger values create cheaper coarser meshes; the generator still
  enforces the configured max face budget. Asset request `bbox` components must all be positive; for flat cloth sheets
  use a small positive thickness dimension such as 0.001, not 0.0. If retrying with changed mesh prompts, include a
  complete revised
  `planner_output`.
- wait_mesh_assets: wait for a previously started background mesh asset job to finish and validate
  assets/asset_manifest.json.
- update_mesh_asset_metadata: synchronously update ready generated mesh manifest metadata without modifying or
  regenerating mesh files. Use this only when geometry is acceptable and runtime scale/bbox metadata is wrong.
- start_xml_assets: start Planner-requested XML/MJCF assets in the background and return immediately. Use `asset_names`
  to restrict generation to specific XML/MJCF asset request names, or null/[] to generate all XML/MJCF requests. If
  retrying with changed XML prompt or actuator/body-tree specification, include a complete revised `planner_output`.
- wait_xml_assets: wait for a previously started background XML/MJCF asset job to finish and merge its partial manifest
  into assets/asset_manifest.json.
- inspect_assets: render/inspect selected ready mesh/XML assets from assets/asset_manifest.json and write
  reports/asset_inspection_report.json plus preview images. Use this to decide whether geometry/contact failures come
  from body placement/choreography or generated asset geometry/topology/scale.
- spawn_workers: start one or more generation workers. Use `roles` from scene, body, action, rendering. Roles in a
  single spawn_workers action are dispatched concurrently by the harness with no default cap beyond requested roles.
  If mesh or XML assets are still running, spawn only writer roles whose contracts do not require the manifest.
  Put role-specific card ids in `simdebug_cards` when a worker should receive a focused subset.
- run_integrator: wire generated modules into src/main.py.
- run_execution: run generated code through the harness on the local GPU.
- run_critic: ask the read-only critic to evaluate execution artifacts.
- run_opt: invoke the dedicated Opt Codex subagent. Current CONFIGS.opt.enabled={opt_enabled}. Use the SimDebug Opt
  cards to decide whether the case is parameter-limited rather than structurally broken.
- request_repair: send `repair_brief` to the owning worker when critic/execution evidence shows a fix. Use SimDebug
  repair-routing cards to choose the correct owner or Planner-owned asset action.
- run_python: optional controlled `uv run --no-sync python ...` command. Use `python_args` and cwd repo/case.
- run_pytest: optional controlled `uv run --no-sync pytest ...` command. Use `pytest_args` and cwd repo/case.
- finish: end the episode with verdict pass, fail, or inconclusive.
""".strip()


PLANNER_ACTION_POLICY_GUIDE = """
Action policy:
- If `planner_output_path` is null, choose write_plan.
- If planner_output dispatch_graph.wait_for_asset_manifest is true and assets.status is not_requested, start all
  required asset families. If one asset family is running but another required family has not started, start the absent
  family before waiting when independent work can overlap.
- If assets.status is running, choose spawn_workers for non-asset-dependent missing roles, or wait_mesh_assets /
  wait_xml_assets when the next useful step requires the manifest.
- If any generation worker is missing or failed, choose spawn_workers or request_repair for the relevant owner.
- Use SimDebug asset, IPC, and source-aware repair cards to decide inspect_assets vs update_mesh_asset_metadata vs
  start_mesh_assets/start_xml_assets vs body/action/scene repair.
- To improve speed, prefer grouping all currently missing writer roles into one spawn_workers action. Keep dependencies
  serial only when a specific worker must inspect another worker's completed source/report.
- Only choose run_integrator after scene/body/action/rendering are all ok.
- Only choose run_execution after integration is current.
- Only choose run_critic after execution is current.
- Use SimDebug Opt cards when deciding run_opt vs repair/regeneration. After a useful Opt result, choose run_execution
  next so root artifacts reflect selected current opt params.
- Only choose finish pass after the latest critic verdict is pass.
- If critic infrastructure failed after configured retries, choose finish with verdict inconclusive; do not request code
  repair from a missing or blocked critic review.
- Prefer run_execution over generic run_python for generated simulations.
""".strip()
