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

## Episode Schemas

Planner-led runtime uses compact schemas for the case controller:

- `episode_state.schema.json`: persisted case state, including timing, worker status, latest artifacts, critic verdict,
  repair queue, budgets, stop condition, command records, and summary pointers.
- `planner_action.schema.json`: one structured action emitted by Planner per turn. Actions include `write_plan`,
  `spawn_workers`, `run_integrator`, `run_execution`, `run_critic`, `request_repair`, `run_python`, `run_pytest`, and
  `finish`. Multiple roles in one `spawn_workers` action form one Planner-selected writer batch and may run in
  parallel.

These schemas should describe episode state and tool requests only. They should not become a new simulation IR.

## Prompt Template Status

- planner episode prompt is assembled in `planner/agent.py`
- scene worker prompt is assembled from `writer/scene.py` and `common.py`
- body worker prompt is assembled from `writer/body.py` and `common.py`
- action worker prompt is assembled from `writer/action.py` and `common.py`
- rendering worker prompt is assembled from `writer/rendering.py` and `common.py`
- integrator entrypoint template is assembled in `utils/integrator.py`
- critic prompt is currently assembled in `evaluation/codex_critic.py`
- XML worker

## Worker Ownership Rules

Planner owns the natural-language interpretation step for timing. It must emit `execution_plan.duration_sec`,
`step_budget`, `render_fps`, and `render_budget`; the harness only validates and forwards those numeric values.

In the Planner-led episode runtime, Planner also owns worker wake-up decisions and repair routing. The harness still
owns execution, validation, sandboxing, artifact collection, and persistence.

- Scene owns fixed objects, stage setup, global FEM+IPC defaults, artifact layout, and optional camera/light anchors for
  Rendering to refine.
- Body owns movable or task-participating actors.
- Action owns behavior, controls, metrics, event logging, and final score.
- Rendering owns camera placement, lighting refinements, capture cadence, render output paths, and visual validation
  hints. It may consume Scene/Body/Action exports but should not change task controls or body definitions.
- Integrator owns final runnable project wiring.

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

The XML worker schema should require:

- path to exactly one generated MJCF/XML file
- confirmation that the XML contains one articulated body and no scene-level props
- MuJoCo import validation status
- joint summary
- actuator/control interface summary for the Action Worker
- known caveats and repair notes
