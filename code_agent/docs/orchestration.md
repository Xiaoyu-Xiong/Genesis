# Orchestration

`code_agent/orchestration/` contains the run coordinator, Codex worker dispatcher, deterministic fallback generator, and
integration code. Current implementation status is summarized in [Implementation Status](status.md).

## Responsibilities

- Create and validate run workspaces.
- Write input bundles such as user prompt, repository rules, and capability summaries.
- Dispatch Planner, Scene, Body, Action, Rendering, critic, and repair calls through the Codex layer.
- Validate schema outputs from [Specs](specs.md).
- Enforce worker ownership by writing only coordinator-accepted `source_code` to each worker's target module.
- Route asset requests to [Assets](assets.md).
- Apply retry budgets and owner-routed repair decisions.
- Produce final run summaries.

## Current Flow

`suite.py` loads `case_id|prompt` cases, creates one workspace per case, optionally calls the Codex planner in
`auto` or `required` mode, generates a project, runs it through [Execution](execution.md), and evaluates artifacts
through [Evaluation](evaluation.md).

Generation is selected by `--generation-mode`:

- `codex`: `workers/dispatcher.py` calls four Codex writers. Each writer returns JSON matching
  `worker_report.schema.json`, including complete Python module text in `source_code`. The coordinator writes that text
  to `src/scene.py`, `src/body.py`, `src/action.py`, or `src/rendering.py`. `integrator.py` then writes `src/main.py`.
- `fallback`: `generator.py` writes a deterministic rigid primitive smoke project. This path remains for runner and
  artifact plumbing checks.

## Worker Flow

The active Codex-worker path is:

1. Planner writes a structured dispatch plan and module contracts.
2. Scene Worker authors `src/scene.py` source.
3. Body Worker authors `src/body.py` source.
4. Action Worker authors `src/action.py` source.
5. Rendering Worker authors `src/rendering.py` source.
6. Coordinator materializes the four modules from returned `source_code`.
7. Integrator writes `src/main.py`.
8. Execution runs the project.
9. Deterministic checks and Codex Critic evaluate metrics, event logs, and visual outputs.
10. If the critic recommends `scene`, `body`, `action`, or `rendering`, `repair_worker()` reruns only that writer with
    failure context.

Asset Bridge, XML generation, static review, and a separate debugger-produced `patch_plan.json` remain planned steps.

The Rendering Worker is intentionally separate from Action Worker. Rendering choices often require camera placement,
lighting, capture cadence, renderer fallback, and video/frame validation decisions that should not be hidden inside
task-control code.

## Boundaries

- Do not implement LLM reasoning here.
- Do not put long-term simulation-authoring logic here; Codex workers or the explicit fallback generator own generated
  simulation modules.
- Do not execute Genesis directly; use [Execution](execution.md).
- Do not let Codex workers bypass schema validation or coordinator-controlled file materialization.
