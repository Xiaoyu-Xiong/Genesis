# Code Agent

`code_agent/` is the code-native natural-language-to-Genesis pipeline. It does not use the legacy IR as the main
generation target. The output of a run is runnable Genesis Python code plus artifacts, logs, metrics, and critic reports.

The current implementation stage has a thin Codex-worker path plus a deterministic fallback. The Codex path dispatches
separate Scene, Body, Action, and Rendering writers, asks each writer to return structured JSON containing complete
module source, then has the coordinator write those modules and wire `src/main.py`. This avoids relying on Codex file
edits in restricted Apptainer or nested-sandbox environments.

Mesh and articulated prompts currently run through primitive stand-ins until the dedicated mesh/XML workers are wired
into generation.

Implementation status is tracked in [Implementation Status](docs/status.md). That document distinguishes completed
infrastructure from temporary MVP fallbacks, partially validated agent-written generation, and unimplemented asset work.

## Current Plan

The implementation plan is [Codex-First Code Agent Pipeline Plan](agentive_code_pipeline_plan.md).

The first-version worker split is now represented in code:

- `Scene Worker`: stage, fixed objects, global FEM+IPC defaults, artifact layout.
- `Body Worker`: movable or task-participating actors.
- `Action Worker`: scripted behavior, controls, metrics, event logging, task score.
- `Rendering Worker`: camera placement, lights, render capture code, frame/video outputs, visual validation signals.
- `Integrator`: deterministic final runnable project wiring.

Writer specs live in four files under `orchestration/workers/`: `scene.py`, `body.py`, `action.py`, and
`rendering.py`. `dispatcher.py` owns Codex invocation, report parsing, source-code materialization, and targeted repair
dispatch.

## Directory Map

| Directory | Purpose |
| --- | --- |
| `orchestration/` | Suite coordinator, Codex worker dispatch, fallback generation, integration, and retry routing. See [Orchestration](docs/orchestration.md). |
| `codex/` | `codex exec` invocation adapter. See [Codex](docs/codex.md). |
| `execution/` | Apptainer/sbatch execution and artifact collection. See [Execution](docs/execution.md). |
| `evaluation/` | Deterministic checks and single-pass critic orchestration. See [Evaluation](docs/evaluation.md). |
| `specs/` | JSON schemas for planner, worker, critic, execution, and repair reports. See [Specs](docs/specs.md). |
| `assets/` | Asset routing and asset implementations. See [Assets](docs/assets.md). |
| `assets/mesh/` | Migrated Meshy / repair / texture pipeline copied from legacy `agent/mesh`. See [Migrated Mesh Pipeline](docs/mesh.md). |
| `scripts/` | First-pass suite scripts and prompt cases. See [Scripts and Suites](docs/scripts.md). |
| `workspaces/` | Generated run workspaces. See [Workspaces](docs/workspaces.md). |
| `docs/` | Centralized documentation for subdirectories. |

## Execution Rule

All Python, `uv`, `pytest`, and Genesis execution must still run inside Apptainer or through the approved sbatch path.
Codex workers return generated code in structured JSON. The coordinator writes files. Workers should not execute
Genesis simulations directly.

## Current Smoke Command

Run a rigid suite on CPU from the host by letting the script enter Apptainer:

```bash
apptainer exec /ocean/projects/cis250078p/xxiong1/containers/genesis.sif \
  bash code_agent/scripts/rigid_primitives/run.sh \
  --cpu --codex-mode off --generation-mode codex --max-cases 1 --no-render
```

Use `--generation-mode fallback` to exercise the deterministic smoke generator instead of Codex writers. Use
`--codex-mode auto` or `required` only for the planner adapter; writer dispatch is controlled separately by
`--generation-mode`.
