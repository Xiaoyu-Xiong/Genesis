# Code Agent

`code_agent/` is the code-native natural-language-to-Genesis pipeline. It does not use the legacy IR as the main
generation target. The output of a run is runnable Genesis Python code plus artifacts, logs, metrics, and critic reports.

The current implementation stage is scaffold-only except for the migrated mesh pipeline under `assets/mesh`.
New core orchestration, specs, and suite code should not be added until their contracts are reviewed.

## Current Plan

The implementation plan is [Codex-First Code Agent Pipeline Plan](agentive_code_pipeline_plan.md).

The fixed first-version worker split is:

- `Scene Worker`: stage, fixed objects, cameras, global FEM+IPC defaults, artifact layout.
- `Body Worker`: movable or task-participating actors.
- `Action Worker`: scripted behavior, metrics, event logging, render triggers, final score.
- `Integrator`: final runnable project wiring.

## Directory Map

| Directory | Purpose |
| --- | --- |
| `orchestration/` | Deterministic run coordinator and retry routing. See [Orchestration](docs/orchestration.md). |
| `codex/` | `codex exec` invocation adapter. See [Codex](docs/codex.md). |
| `execution/` | Apptainer/sbatch execution and artifact collection. See [Execution](docs/execution.md). |
| `evaluation/` | Deterministic checks and single-pass critic orchestration. See [Evaluation](docs/evaluation.md). |
| `specs/` | Future schemas and prompt templates. See [Specs](docs/specs.md). |
| `assets/` | Asset routing and asset implementations. See [Assets](docs/assets.md). |
| `assets/mesh/` | Migrated Meshy / repair / texture pipeline copied from legacy `agent/mesh`. See [Migrated Mesh Pipeline](docs/mesh.md). |
| `scripts/` | First-pass suite scripts and prompt cases. See [Scripts and Suites](docs/scripts.md). |
| `workspaces/` | Generated run workspaces. See [Workspaces](docs/workspaces.md). |
| `docs/` | Centralized documentation for subdirectories. |

## Execution Rule

All Python, `uv`, `pytest`, and Genesis execution must still run inside Apptainer or through the approved sbatch path.
Codex workers may write generated code but should not execute Genesis simulations directly.
