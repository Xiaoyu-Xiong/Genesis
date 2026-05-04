# Specs

`code_agent/specs/` contains machine-readable schemas used by the Planner runtime, Writer subagents, Evaluation, and
shared Utils.

The first implementation should keep specs compact. They constrain process metadata and Codex worker outputs; they are
not a replacement for the old IR and must not become a simulation compiler target.

## Current Schemas

- `planner_output.schema.json`
- `asset_manifest.schema.json`
- `worker_report.schema.json`
- `execution_report.schema.json`
- `critic_report.schema.json`
- `episode_state.schema.json`
- `planner_action.schema.json`
- `xml_worker_report.schema.json`

## Episode Schemas

Planner-led runtime uses compact schemas for the case controller:

- `episode_state.schema.json`: persisted case state, including timing, asset manifest status, worker status, latest
  artifacts, critic verdict, repair queue, budgets, stop condition, command records, and summary pointers.
- `planner_action.schema.json`: one structured action emitted by Planner per turn. Actions include `write_plan`,
  `start_mesh_assets`, `generate_mesh_assets`, `wait_mesh_assets`, `start_xml_assets`, `generate_xml_assets`,
  `wait_xml_assets`, `spawn_workers`, `run_integrator`, `run_execution`, `run_critic`, `request_repair`,
  `run_python`, `run_pytest`, and `finish`. Multiple roles in one `spawn_workers` action form one Planner-selected
  writer batch and run in parallel by default.

These schemas should describe episode state and tool requests only. They should not become a new simulation IR.

## Prompt Template Status

- planner episode prompt is assembled in `planner/agent.py`
- shared prompt clauses live in `utils/general_prompts.py`
- scene worker prompt is assembled from `writer/scene.py`, `writer/common.py`, and `utils/general_prompts.py`
- body worker prompt is assembled from `writer/body.py`, `writer/common.py`, and `utils/general_prompts.py`
- action worker prompt is assembled from `writer/action.py`, `writer/common.py`, and `utils/general_prompts.py`
- rendering worker prompt is assembled from `writer/rendering.py`, `writer/common.py`, and `utils/general_prompts.py`
- integrator entrypoint template is assembled in `utils/integrator.py`
- critic prompt is currently assembled in `evaluation/agent.py`
- XML worker prompt is assembled in `assets/xml/agent.py`

## Worker Ownership Rules

Planner owns the natural-language interpretation step for timing. It must emit `execution_plan.duration_sec`,
`step_budget`, `render_fps`, and `render_budget`; the harness only validates and forwards those numeric values.

`contracts/deformable_config.json` records the effective `CONFIGS.deformable` values for each case. Deformable
generation is gated by `enabled`; generated code receives this contract as `deformable_cfg` and must read FEM, IPC, tet,
precision, and FEM material-range defaults from it instead of hardcoding them. FEM elastic bodies should still make
task-specific material choices: generated `gs.materials.FEM.Elastic(...)` calls must pass explicit `E`, `nu`, and `rho`
values selected from the config ranges, using the config defaults when no special material is needed.

In the Planner-led episode runtime, Planner also owns worker wake-up decisions and repair routing. The harness still
owns execution, validation, sandboxing, artifact collection, and persistence.

- The asset bridge owns Planner-facing mesh and XML/MJCF asset actions. The mesh episode/manifest modules own canonical
  generated mesh runtime paths, Genesis-ready scale factors, coordinate metadata, and texture paths for code writers.
  The XML episode/validation modules own primitive MJCF body-tree generation, joint/actuator/control-interface
  metadata, preview evidence, actuator response checks, and partial manifest entries merged into
  `assets/asset_manifest.json`.
- Scene owns fixed objects, stage setup, global FEM+IPC defaults, configured simulation dt/substeps, artifact layout,
  and optional camera/light anchors for Rendering to refine.
- Body owns movable or task-participating rigid actors and, when deformable is enabled, FEM primitive actors.
- Action owns behavior, controls, metrics, event logging, and final score.
- Rendering owns camera placement, lighting refinements, configured capture cadence/resolution, render output paths, and
  visual validation hints. It may consume Scene/Body/Action exports but should not change task controls or body
  definitions.
- Integrator owns final runnable project wiring.

Current generated module interfaces are:

- `scene.create_scene(backend, *, sim_dt, sim_substeps, deformable_cfg)`
- `body.create_bodies(scene, task, *, deformable_cfg)`
- `action.run_actions(scene, actors, *, out_dir, steps, render_state=None)`
- `rendering.setup_rendering(...)`, `capture_frame`, and `finalize_rendering`

## Rendering Worker Output Requirements

The Rendering Worker schema should require:

- target render files or helper module paths
- camera definitions with position, lookat, resolution, fov, and intended target visibility
- lighting definitions or explicit statement that Scene lighting is sufficient
- render cadence, frame budget, and video fps
- output artifact contract: `render.mp4`, optional `frames/`, optional `render_stats.json`
- failure strategy for cases where Genesis camera rendering is unavailable or too slow
- visual validation notes describing what should be visible and how the Critic should interpret the render
- known limitations such as occlusion, low motion, or diagnostic-only rendering

## Guardrails

- Current writer prompts must state that Codex may edit only the assigned target file.
- Every schema-producing Codex call should use `--output-schema`.
- Every worker report should include changed files, exports, assumptions, commands run, unresolved risks, and handoff
  notes.
- The dispatcher is responsible for validating that the assigned target file was written and contains the required
  export.

## XML Worker Output Requirements

`xml_worker_report.schema.json` requires the standalone XML worker to report:

- path to exactly one generated MJCF/XML file
- confirmation that the XML contains one articulated body and no scene-level props
- MuJoCo import validation status
- joint summary
- actuator/control interface summary for the Action Worker
- known caveats and repair notes

The XML runner records the full validation and preview results in `reports/xml_asset_generation_report.json`. In the
main Planner loop, XML episode results are written to `assets/xml_asset_manifest.json` and merged into the canonical
`assets/asset_manifest.json` using the common `asset_manifest.schema.json` optional MJCF fields.
