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
- Planner prompts expose the repository and case workspace as read-only context, encourage detailed downstream
  instructions, and require the final result to be faithful to the task while remaining physically and visually logical.
- `context/genesis.py` builds a suite-level Genesis context pack from selected official documentation and local source
  anchors. Its active non-rigid scope is FEM+IPC only; rigid, articulated, mesh, texture, and rendering context is kept
  only where it supports FEM+IPC scenes. Agent prompts receive compact pointers to the pack and read details on demand.
- `planner/session.py` and `planner/actions.py` implement the Planner-led episode harness. `utils/suite.py` starts one
  `PlannerSession` per case instead of hard-coding the full generation/execution/critic sequence itself.
- Planner turns emit structured actions through `planner_action.schema.json`: `write_plan`, `start_mesh_assets`,
  `generate_mesh_assets`, `wait_mesh_assets`, `spawn_workers`, `run_integrator`, `run_execution`, `run_critic`,
  `request_repair`, `run_python`, `run_pytest`, and `finish`.
- `episode_state.schema.json` records the persisted case state shape. Runtime state is written to
  `reports/episode_state.json`; planner actions and worker dispatches are appended to JSONL history files.
- `utils/codex.py` uses explicit `CodexExecRequest` objects, applies the configured model reasoning effort, saves JSONL
  logs and final messages, records stderr, and returns structured invocation results.
- Scene, Body, Action, and Rendering writer specs are split across four files:
  `writer/scene.py`, `body.py`, `action.py`, and `rendering.py`.
- `writer/dispatcher.py` dispatches those writers in `workspace-write` sandbox mode, parses
  `worker_report.schema.json`, validates each target module, records `worker_dispatch.json`, and can rerun a single
  owning worker during repair.
- Writer prompts expose the repository and case workspace for read-only inspection while keeping writes restricted to
  the assigned generated module.
- Writer dispatch supports Planner-selected parallel batches. Roles included in one `spawn_workers` action run
  concurrently. The default `CONFIGS.harness.max_parallel_workers=None` applies no artificial cap, so all requested
  writer subagents can run at once. Dependent work remains serial only when Planner splits it across separate turns.
- The top-level asset bridge is connected to the Planner action loop through `start_mesh_assets`, `wait_mesh_assets`,
  and the compatibility `generate_mesh_assets`. It delegates selected `generated_mesh` requests to the mesh episode
  runner, which calls the Meshy/repair/texture pipeline, writes `assets/asset_manifest.json`, and records
  `reports/asset_generation_report.json`.
- Mesh asset generation can also run as a background Planner action through `start_mesh_assets`; Planner may dispatch
  non-asset-dependent writer roles while assets are still running, then call `wait_mesh_assets` before any manifest-
  dependent writer or integration step.
- Meshy API submission, polling, and downloads run in parallel for all selected generated-mesh requests by default,
  while local manifold/repair/UV/texture-transfer processing remains serial to keep WSL memory peaks bounded.
- Generated mesh manifest entries include provider coordinate metadata, Genesis-ready scale factors, and transferred
  texture paths so writers can instantiate Meshy assets with the correct orientation, size, and material.
- The current worker protocol is direct-edit based: Codex workers edit only their assigned generated module and return
  structured report metadata.
- `utils/integrator.py` writes the stable `src/main.py` that imports and wires Scene, Body, Action, and
  Rendering modules. It passes configured simulation dt/substeps and render cadence/resolution defaults into the
  generated module interfaces.
- `utils/timing.py` consumes the planner's structured `execution_plan` plus explicit CLI overrides. It does
  not parse task text itself.
- Local GPU execution is routed through `utils/execution.py` and `utils/local_execution.py`. Generated Genesis code is
  launched through the repository uv environment on the dedicated GPU by default.
- Execution reports collect command metadata, stdout/stderr paths, exit status, timeout state, and discovered artifacts.
- Deterministic evaluation checks execution success, required JSON artifacts, render artifacts when required, and writes
  `artifact_evaluation.json`.
- `evaluation/visual.py` samples rendered frames, writes a contact sheet, summarizes generated texture colors, and
  reports texture-presence warnings for the critic.
- `evaluation/codex_critic.py` implements a single-pass Codex Critic over generated source, planner contracts, asset
  manifests, Genesis context packs, execution reports, metrics, event logs, render stats, stdout/stderr, visual
  evidence, and an attached contact-sheet image when available. Its repair guidance is expected to compare prompt,
  source, context, and output artifacts and provide detailed owner-routed source-level changes.
- `PlannerSession` can route a failed run back to the Planner-selected owner when that owner is one of the four
  implemented generation writers.
- A rigid primitive case has reached successful Codex writer dispatch with all four generated modules materialized,
  local GPU execution, Genesis camera rendering, deterministic checks, and single-pass Codex Critic evaluation.
- Documentation is centralized under `docs/`; per-directory README files were removed except for the package-level
  `code_agent/README.md` and the docs index.
- Package-level re-export files were removed from internal folders. Current imports use explicit module paths such as
  `code_agent.planner.session`, `code_agent.writer.dispatcher`, and `code_agent.utils.execution`.

## Current Runtime Path

- Planner first writes a structured `planner_output` through the `write_plan` action. The harness validates that output,
  resolves duration, step budget, render fps, and target frame count, then writes `contracts/planner_output.json` and
  `contracts/timing.json`.
- Suite startup writes a shared `context/genesis/` pack under the suite output directory and copies
  `genesis_context.md/json` into each case's `contracts/` directory before Planner starts.
- Planner chooses subsequent actions from the harness action library. The harness performs the real Codex writer calls,
  mesh asset generation when Planner requests it, parallel writer batches when Planner selects multiple independent
  roles, plus integration, local GPU execution, critic calls, controlled Python/Pytest commands, and finish handling.
- If Planner sets `dispatch_graph.wait_for_asset_manifest`, the harness requires a ready asset manifest only before
  manifest-dependent writer roles or integration. Non-asset-dependent writers may run while the asset job is still in
  progress. The manifest is included in writer prompts once it is ready so generated code can instantiate meshes from
  canonical paths.
- The generated render path uses Genesis camera rendering hooks owned by the Rendering Worker. It adds cameras before
  `scene.build()`, captures Genesis RGB frames during stepping, and writes video/stat artifacts after execution.
- `evaluation/runner.py` is the top-level evaluator wrapper. It combines artifact checks with the single-pass Codex
  Critic and visual evidence generation.

## Operational Notes

- Mesh asset generation depends on `MESHY_API_KEY`. On this machine, the API-key export may live after the
  non-interactive guard in `~/.bashrc`, so Codex or script-launched non-interactive shells can miss it even when an
  interactive terminal sees it. Mesh suite commands should export or load the key in the same command environment
  without logging the secret value.

## Not Implemented Yet

- Planner-led control is implemented as iterative Planner turns over persisted state. It is not a single persistent OS
  process; each Planner decision is a `codex exec` invocation that consumes `episode_state.json`.
- Episode resume from an interrupted run is not implemented yet; state is persisted for audit and future resume work.
- Planner output is schema-constrained, but strict expansion into per-worker contract files is not yet implemented.
- Worker write-scope validation currently checks the reported target module and required export; git/workspace diff
  audits and richer static ownership checks are not yet implemented.
- Mesh-heavy rigid cases now run through the Planner-callable asset bridge and mesh episode runner, but repeated-run
  stability and broader asset coverage still need validation.
- Codex XML/MJCF generation is not implemented in the main run loop.
- Generated articulated assets are not yet MuJoCo-import validated.
- The current repair loop uses critic `recommended_owner` directly.
- Retry budgets are still primarily CLI/session parameters, not the full policy from `configs.py`.
- One rigid primitive case has been validated end-to-end through Genesis execution and semantic critic acceptance on the
  dedicated GPU. Broader suite coverage, repeated-run stability, and FEM+IPC/asset-heavy cases are not yet validated.

## Current Structural Next Step

The next structural step is to harden the Planner-led episode loop rather than adding another manual sequence:

- Validate repeated-run stability on more rigid primitive cases.
- Add resume support from `reports/episode_state.json`.
- Add richer git/workspace diff audits for worker write scope.
- Validate mesh-heavy rigid and FEM+IPC suite cases end-to-end.
- Add Codex XML/MJCF generation as an additional Planner-callable action.
