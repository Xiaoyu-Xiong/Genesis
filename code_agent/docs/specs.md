# Specs

`code_agent/specs/` is reserved for machine-readable schemas and prompt templates.

The first implementation should keep specs compact. They constrain process metadata and Codex worker outputs; they are
not a replacement for the old IR and must not become a simulation compiler target.

## Planned Schemas

- `planner_output.schema.json`
- `asset_manifest.schema.json`
- `worker_report.schema.json`
- `review_report.schema.json`
- `execution_report.schema.json`
- `patch_plan.schema.json`
- `critic_report.schema.json`

## Planned Prompt Templates

- planner
- scene worker
- body worker
- action worker
- integrator
- reviewer
- debugger
- critic
- XML worker

## Worker Ownership Rules

- Scene owns fixed objects, stage setup, cameras, lights, global FEM+IPC defaults, and artifact layout.
- Body owns movable or task-participating actors.
- Action owns behavior, metrics, event logging, render triggers, and final score.
- Integrator owns final runnable project wiring.

## Guardrails

- Every writer prompt must include exact allowed write paths.
- Every schema-producing Codex call should use `--output-schema`.
- Every worker report should include changed files, exports, assumptions, commands run, and unresolved risks.

## XML Worker Output Requirements

The XML worker schema should require:

- path to exactly one generated MJCF/XML file
- confirmation that the XML contains one articulated body and no scene-level props
- MuJoCo import validation status
- joint summary
- actuator/control interface summary for the Action Worker
- known caveats and repair notes
