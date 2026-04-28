# Implementation Status

This document records what is implemented in `code_agent/`, what is only a temporary MVP fallback, and what is still
planned. It should be updated whenever the pipeline structure changes.

## Implemented

- `code_agent.cli run-suite` exists and can run prompt suites.
- Suite orchestration can read `case_id|prompt` files, create per-case workspaces, call the Codex planner adapter,
  generate a runnable project, execute it, evaluate artifacts, and write a suite summary.
- The Codex adapter can run `codex exec`, save JSONL logs, save the final message, record stderr, and return structured
  invocation results.
- Scene, Body, Action, and Rendering writer specs are split across four files:
  `orchestration/workers/scene.py`, `body.py`, `action.py`, and `rendering.py`.
- `orchestration/workers/dispatcher.py` dispatches those writers, parses `worker_report.schema.json`, writes each
  returned `source_code` string to the target module, records `worker_dispatch.json`, and can rerun a single owning
  worker during repair.
- The current worker protocol is JSON-source based: Codex workers run read-only and return complete module source in
  `source_code`; the coordinator writes files. This was chosen because direct Codex file editing hit nested sandbox
  `bwrap ENOSPC` failures in the current environment.
- `orchestration/integrator.py` writes the stable `src/main.py` that imports and wires Scene, Body, Action, and
  Rendering modules.
- CPU execution is routed through Apptainer-aware execution code. Generated Genesis code is not launched through host
  Python.
- Execution reports collect command metadata, stdout/stderr paths, exit status, timeout state, and discovered artifacts.
- Deterministic evaluation checks execution success, required JSON artifacts, render artifacts when required, and writes
  `critic_report.json`.
- `evaluation/codex_critic.py` implements a single-pass Codex Critic over execution reports, metrics, event logs, and
  render stats.
- `orchestration/suite.py` can route a failed run back to the critic-recommended owner when that owner is one of the
  four implemented writers.
- Rigid prompt suites have run through the MVP fallback path on CPU. A single rigid primitive case has also reached
  successful Codex writer dispatch with all four generated modules materialized.
- The legacy mesh pipeline has been migrated under `code_agent/assets/mesh`, but it is not yet connected to the main
  suite generation path.

## Temporary MVP Fallbacks

- `orchestration/generator.py` is a deterministic smoke generator. It writes fixed-template `scene.py`, `body.py`,
  `action.py`, and `main.py` files so the outer pipeline can be tested.
- The deterministic generator uses a tiny rigid primitive scene even for articulated or mesh prompts. It is not the
  final code-native generation strategy.
- The current generated render path is a lightweight top-down trajectory diagnostic assembled from sampled actor
  positions. It is useful for smoke validation but is not the intended final rendering implementation.
- `orchestration/generator.py` remains as `--generation-mode fallback`. It is useful for runner and artifact plumbing
  checks, but it is no longer the only generation path.
- `simple.py` is still the top-level evaluator wrapper. It now combines deterministic checks with the single-pass
  Codex Critic rather than serving as a purely deterministic critic.
- The generated Rendering Worker currently favors diagnostic top-down rendering from `event_log.json` for CPU/headless
  robustness. Full Genesis camera rendering is still future work.

## Not Implemented Yet

- The planner output is not yet fully schema-validated and expanded into strict per-worker contracts.
- Worker write-scope validation currently means coordinator-controlled writes from `source_code`; git/workspace diff
  audits and richer static ownership checks are not yet implemented.
- Static Codex review is not yet connected between generation and execution.
- Asset Bridge is not yet connected to suite orchestration.
- Meshy generation, mesh repair, and texture transfer are not yet requested from planner asset requests in the main
  run loop.
- Codex XML/MJCF generation is not implemented in the main run loop.
- Generated articulated assets are not yet MuJoCo-import validated.
- Debugger repair with a separate owner-routed `patch_plan.json` is not yet connected. The current repair loop uses
  critic `recommended_owner` directly.
- Retry budgets are minimal CLI parameters, not the full policy from `configs.py`.
- Rigid primitive Codex generation has not yet been validated end-to-end through Genesis execution in the current CPU
  environment. A control run of `examples/tutorials/hello_genesis.py` timed out after 120 seconds inside the configured
  Apptainer image, so CPU runtime validation is currently blocked by environment/runtime cost rather than only by
  writer dispatch.

## Current Structural Next Step

The next structural step is not to split more workers. The current four-writer split should be hardened:

- Add static review and import-level checks before execution.
- Make execution validation practical for CPU by either shrinking Genesis smoke workloads further or adding an explicit
  artifact-only dry-run mode while retaining strict Genesis execution as the real acceptance path.
- Connect Asset Bridge, mesh requests, and Codex XML/MJCF generation.
- Replace direct critic-owner repair with a structured `patch_plan.json` debugger step when needed.
