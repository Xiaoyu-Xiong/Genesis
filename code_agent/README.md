# Code Agent

`code_agent/` is the code-native natural-language-to-Genesis pipeline. It generates runnable Genesis Python code plus
artifacts, logs, metrics, and critic reports.

The current implementation stage uses a Planner-led episode runtime plus separate Codex Scene, Body, Action, and
Rendering writers in `workspace-write` sandbox mode. Planner chooses structured actions for the full lifecycle of one
case: writing the plan, waking generation workers, requesting integration and local GPU execution, asking Critic for
evaluation, sending focused repair briefs, and finishing the episode.

Mesh and articulated prompts currently run through primitive stand-ins until the dedicated mesh/XML workers are wired
into generation.

Implementation status is tracked in [Implementation Status](docs/status.md). That document distinguishes completed
infrastructure, partially validated agent-written generation, and unimplemented asset work.

## Current Plan

The implementation plan is [Codex-First Code Agent Pipeline Plan](agentive_code_pipeline_plan.md).

The first-version worker split is represented in code and remains the default generation worker set:

- `Scene Worker`: stage, fixed objects, global FEM+IPC defaults, artifact layout.
- `Body Worker`: movable or task-participating actors.
- `Action Worker`: scripted behavior, controls, metrics, event logging, task score.
- `Rendering Worker`: camera placement, lights, render capture code, frame/video outputs, visual validation signals.
- `Integrator`: deterministic final runnable project wiring.

Writer specs live in four files under `writer/`: `scene.py`, `body.py`, `action.py`, and `rendering.py`.
`writer/dispatcher.py` owns Codex invocation, report parsing, target-file validation, role-set dispatch, and targeted
repair dispatch.

`planner/session.py` starts one Planner episode per case. Planner emits structured actions such as spawning workers,
running integration, executing generated code, invoking Critic, requesting owner repair, running controlled
Python/Pytest commands, or finishing the episode. Shell execution, GPU use, schema validation, write-scope enforcement,
artifact collection, and retry limits stay inside the Python harness.

## Directory Map

| Directory | Purpose |
| --- | --- |
| `planner/` | Planner agent prompt construction, Planner-turn invocation, and per-case episode harness. |
| `writer/` | Scene, Body, Action, and Rendering code-writing subagents plus writer dispatch. |
| `utils/` | Codex invocation, local execution, suite loading, timing, and integration helpers. See [Utils](docs/utils.md). |
| `evaluation/` | Deterministic checks and single-pass critic runner. See [Evaluation](docs/evaluation.md). |
| `specs/` | JSON schemas for planner, worker, critic, execution, and repair reports. See [Specs](docs/specs.md). |
| `assets/` | Asset routing and asset implementations. See [Assets](docs/assets.md). |
| `assets/mesh/` | Meshy / repair / texture asset implementation. See [Mesh Pipeline](docs/mesh.md). |
| `scripts/` | Suite scripts and prompt cases. See [Scripts and Suites](docs/scripts.md). |
| `workspaces/` | Generated run workspaces. See [Workspaces](docs/workspaces.md). |
| `docs/` | Centralized documentation. |

## Execution Rule

Python, `uv`, `pytest`, and Genesis execution should run through the repository uv environment, using the dedicated
local GPU by default for simulation, rendering, profiling, optimization, tests, and examples. CPU execution is only for
explicit CPU requests, unavailable GPU, or clearly CPU-only tasks. Codex workers edit only their assigned generated
module and return a structured report. Workers should not execute Genesis simulations directly.

## Current Validation Command

Run a rigid suite on the local GPU:

```bash
bash code_agent/scripts/rigid_primitives/run.sh \
  --gpu --max-cases 1 --render
```

The planner is always part of the run. Suite timing comes from the planner's structured `execution_plan`; override it
with `--duration-sec`, `--steps`, or `--render-fps` only when a case needs explicit timing.
