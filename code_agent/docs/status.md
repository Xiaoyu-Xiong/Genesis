# Implementation Status

This document records what is implemented in `code_agent/` and what is still planned. It should be updated whenever
the pipeline structure changes.

## Implemented

- `code_agent.cli run-suite` exists and can run prompt suites.
- Suite utilities can read `case_id|prompt` files, create per-case workspaces, call the Codex planner, generate a
  runnable project through writer agents, execute it, evaluate artifacts, and write a suite summary.
- Runtime support code has been consolidated under `utils/`: Codex invocation, local execution, generated entrypoint
  integration, timing resolution, and suite loading all live there.
- `planner/agent.py` implements the Planner agent prompt and Codex invocation for each Planner turn.
- `planner/session.py` and `planner/actions.py` implement the Planner-led episode harness. `utils/suite.py` starts one
  `PlannerSession` per case instead of hard-coding the full generation/execution/critic sequence itself.
- Planner turns emit structured actions through `planner_action.schema.json`: `write_plan`, `spawn_workers`,
  `run_integrator`, `run_execution`, `run_critic`, `request_repair`, `run_python`, `run_pytest`, and `finish`.
- `episode_state.schema.json` records the persisted case state shape. Runtime state is written to
  `reports/episode_state.json`; planner actions and worker dispatches are appended to JSONL history files.
- `utils/codex.py` uses explicit `CodexExecRequest` objects, saves JSONL logs and final messages, records stderr, and
  returns structured invocation results.
- Scene, Body, Action, and Rendering writer specs are split across four files:
  `writer/scene.py`, `body.py`, `action.py`, and `rendering.py`.
- `writer/dispatcher.py` dispatches those writers in `workspace-write` sandbox mode, parses
  `worker_report.schema.json`, validates each target module, records `worker_dispatch.json`, and can rerun a single
  owning worker during repair.
- Writer dispatch supports Planner-selected parallel batches. Roles included in one `spawn_workers` action run
  concurrently up to `CONFIGS.harness.max_parallel_workers`; dependent work remains serial by having Planner split it
  across separate turns.
- The current worker protocol is direct-edit based: Codex workers edit only their assigned generated module and return
  structured report metadata.
- `utils/integrator.py` writes the stable `src/main.py` that imports and wires Scene, Body, Action, and
  Rendering modules.
- `utils/timing.py` consumes the planner's structured `execution_plan` plus explicit CLI overrides. It does
  not parse task text itself.
- Local GPU execution is routed through `utils/execution.py` and `utils/local_execution.py`. Generated Genesis code is
  launched through the repository uv environment on the dedicated GPU by default.
- Execution reports collect command metadata, stdout/stderr paths, exit status, timeout state, and discovered artifacts.
- Deterministic evaluation checks execution success, required JSON artifacts, render artifacts when required, and writes
  `artifact_evaluation.json`.
- `evaluation/codex_critic.py` implements a single-pass Codex Critic over execution reports, metrics, event logs, and
  render stats.
- `PlannerSession` can route a failed run back to the Planner-selected owner when that owner is one of the four
  implemented generation writers.
- A rigid primitive case has reached successful Codex writer dispatch with all four generated modules materialized,
  local GPU execution, Genesis camera rendering, deterministic checks, and single-pass Codex Critic evaluation.
- The mesh asset implementation lives under `code_agent/assets/mesh`, but it is not yet connected to the main suite
  generation path.
- Documentation is centralized under `docs/`; per-directory README files were removed except for the package-level
  `code_agent/README.md` and the docs index.
- Package-level re-export files were removed from internal folders. Current imports use explicit module paths such as
  `code_agent.planner.session`, `code_agent.writer.dispatcher`, and `code_agent.utils.execution`.

## Current Runtime Path

- Planner first writes a structured `planner_output` through the `write_plan` action. The harness validates that output,
  resolves duration, step budget, render fps, and target frame count, then writes `contracts/planner_output.json` and
  `contracts/timing.json`.
- Planner chooses subsequent actions from the harness action library. The harness performs the real Codex writer calls,
  including parallel writer batches when Planner selects multiple independent roles, plus integration, local GPU
  execution, critic calls, controlled Python/Pytest commands, and finish handling.
- The generated render path uses Genesis camera rendering hooks owned by the Rendering Worker. It adds cameras before
  `scene.build()`, captures Genesis RGB frames during stepping, and writes video/stat artifacts after execution.
- `evaluation/runner.py` is the top-level evaluator wrapper. It combines artifact checks with the single-pass Codex
  Critic.

## Not Implemented Yet

- Planner-led control is implemented as iterative Planner turns over persisted state. It is not a single persistent OS
  process; each Planner decision is a `codex exec` invocation that consumes `episode_state.json`.
- Episode resume from an interrupted run is not implemented yet; state is persisted for audit and future resume work.
- Planner output is schema-constrained, but strict expansion into per-worker contract files is not yet implemented.
- Worker write-scope validation currently checks the reported target module and required export; git/workspace diff
  audits and richer static ownership checks are not yet implemented.
- Asset Bridge is not yet connected to the suite runtime.
- Meshy generation, mesh repair, and texture transfer are not yet requested from planner asset requests in the main
  run loop.
- Codex XML/MJCF generation is not implemented in the main run loop.
- Generated articulated assets are not yet MuJoCo-import validated.
- The current repair loop uses critic `recommended_owner` directly.
- Retry budgets are still primarily CLI/session parameters, not the full policy from `configs.py`.
- One rigid primitive case has been validated end-to-end through Genesis execution and semantic critic acceptance on the
  dedicated GPU. Broader suite coverage, repeated-run stability, and non-rigid/asset-heavy cases are not yet validated.

## Current Structural Next Step

The next structural step is to harden the Planner-led episode loop rather than adding another manual sequence:

- Validate repeated-run stability on more rigid primitive cases.
- Add resume support from `reports/episode_state.json`.
- Add richer git/workspace diff audits for worker write scope.
- Add Asset Bridge, mesh requests, and Codex XML/MJCF generation as additional Planner-callable actions.
