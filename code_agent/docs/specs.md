# Specs

`code_agent/specs/` contains machine-readable schemas used by the current coordinator and planned future workers.

The first implementation should keep specs compact. They constrain process metadata and Codex worker outputs; they are
not a replacement for the old IR and must not become a simulation compiler target.

## Current Schemas

- `planner_output.schema.json`
- `asset_manifest.schema.json`
- `worker_report.schema.json`
- `review_report.schema.json`
- `execution_report.schema.json`
- `patch_plan.schema.json`
- `critic_report.schema.json`

## Prompt Template Status

- planner prompt is currently assembled in `orchestration/suite.py`
- scene worker prompt is assembled from `orchestration/workers/scene.py` and `common.py`
- body worker prompt is assembled from `orchestration/workers/body.py` and `common.py`
- action worker prompt is assembled from `orchestration/workers/action.py` and `common.py`
- rendering worker prompt is assembled from `orchestration/workers/rendering.py` and `common.py`
- integrator
- reviewer
- debugger
- critic prompt is currently assembled in `evaluation/codex_critic.py`
- XML worker

## Worker Ownership Rules

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
- fallback strategy for headless CPU execution when Genesis camera rendering is unavailable or too slow
- visual validation notes describing what should be visible and how the Critic should interpret the render
- known limitations such as occlusion, low motion, or diagnostic-only rendering

## Guardrails

- Current writer prompts must state that Codex should not edit files directly and must return complete module source in
  `source_code`.
- Every schema-producing Codex call should use `--output-schema`.
- Every worker report should include changed files, exports, `source_code`, assumptions, commands run, unresolved risks,
  and handoff notes.
- The coordinator, not Codex, is responsible for writing accepted `source_code` to the target file.

## XML Worker Output Requirements

The XML worker schema should require:

- path to exactly one generated MJCF/XML file
- confirmation that the XML contains one articulated body and no scene-level props
- MuJoCo import validation status
- joint summary
- actuator/control interface summary for the Action Worker
- known caveats and repair notes
