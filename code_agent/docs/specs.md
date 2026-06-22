# Specs

`code_agent/specs/` contains JSON schemas for Planner actions, generated plans, worker reports, asset manifests,
execution reports, critic reports, persisted episode state, and the first optimization contracts. They are runtime
contracts, not a simulation IR.

Current Planner actions:

- `write_plan`
- `start_mesh_assets`
- `wait_mesh_assets`
- `update_mesh_asset_metadata`
- `start_xml_assets`
- `wait_xml_assets`
- `inspect_assets`
- `spawn_workers`
- `run_integrator`
- `run_execution`
- `run_critic`
- `run_opt`
- `request_repair`
- `run_python`
- `run_pytest`
- `finish`

Current generated module interfaces:

- `scene.create_scene(backend, *, sim_dt, sim_substeps, deformable_cfg)`
- `body.create_bodies(scene, task, *, deformable_cfg)`
- `action.run_actions(scene, actors, *, out_dir, steps, render_state=None)`
- `rendering.setup_rendering(...)`
- `rendering.capture_frame(render_state, step)`
- `rendering.finalize_rendering(render_state, *, event_log_path=None, metrics_path=None)`

`planner_output.physics_plan` carries the Planner-selected mode for a case:

- `rigid`: ordinary rigid/articulated simulation, IPC off
- `rigid_ipc`: rigid/articulated simulation with IPC contact/coupling
- `fem_ipc`: FEM soft-body and/or FEM.Cloth simulation, IPC forced on

`contracts/deformable_config.json` carries the effective FEM/IPC contract derived from that plan. `enabled` gates FEM
deformables, and `ipc_enabled` gates `gs.options.IPCCouplerOptions`; FEM enabled forces IPC enabled. These are derived
execution-contract fields, not config-level switches.

The contract provides FEM material ranges/defaults for `E`, `nu`, and `rho`, plus shared FEM/IPC options such as
`fem_model`, hydroelastic/contact-resistance settings, tet resolution, precision, and IPC solver/contact parameters.
It intentionally does not provide a `fem_friction_mu` override: generated body code must choose explicit
task-appropriate FEM `friction_mu` values per material.

`planner_output.execution_plan` carries explicit runtime timing: `sim_dt`, `sim_substeps`,
`render_every_n_steps`, `render_fps`, `render_budget`, and `render_res`. The harness records the resolved values in
`contracts/timing.json` and passes them to `src/main.py` during execution and optimization.

Optimization contract schemas live under `code_agent/specs/opt_schema/`:

- `target_spec.schema.json`: task target, objective terms, and success criteria.
- `opt_space.schema.json`: CMA-ES search variables, defaults, bounds, physical/action ownership, trial budget, and
  optional strategy knobs for phases, restarts, and early stopping.
- `opt_params.schema.json`: structured default/current/best parameter payloads consumed by generated modules.
- `opt_trace_entry.schema.json`: one JSONL record per optimization trial.
- `opt_report.schema.json`: optimization summary, baseline score, best trial, and report paths.
- `opt_subagent_report.schema.json`: structured final output from the Codex Opt subagent to Planner.
- `verification_report.schema.json`: best-result comparison against the target spec.

These schemas define the parameter optimization interface. The current Opt entry point invokes a Codex subagent that can
prepare generated modules/contracts and call the lower-level optimizer runner.
Optimization variables may be owned by `scene`, `body`, or `action`; `rendering` is intentionally excluded from the
optimization surface.
