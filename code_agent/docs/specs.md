# Specs

`code_agent/specs/` contains JSON schemas for Planner actions, generated plans, worker reports, asset manifests,
execution reports, critic reports, and persisted episode state. They are runtime contracts, not a simulation IR.

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

`contracts/deformable_config.json` carries the effective FEM/IPC contract. `enabled` gates FEM deformables, and
`ipc_enabled` gates `gs.options.IPCCouplerOptions`; FEM enabled forces IPC enabled.

The contract provides FEM material ranges/defaults for `E`, `nu`, and `rho`, plus shared FEM/IPC options such as
`fem_model`, hydroelastic/contact-resistance settings, tet resolution, precision, and IPC solver/contact parameters.
It intentionally does not provide a `fem_friction_mu` override: generated body code must choose explicit
task-appropriate FEM `friction_mu` values per material.
