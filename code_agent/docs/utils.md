# Utils

`code_agent/utils/` contains non-agent support code used by Planner, Writer, Evaluation, and the CLI.

## Modules

- `codex.py`: standardizes non-interactive `codex exec` calls. It builds the command, applies sandbox/model/reasoning
  effort/service-tier/schema settings, writes JSONL event logs, captures the final message and stderr, and returns
  `CodexExecResult`. `CONFIGS.codex.service_tier="fast"` enables Codex's official fast mode without lowering reasoning
  effort; set it to `"standard"` for the normal service tier, or `None` to omit the service-tier override entirely.
  Usage-limit responses are classified as `codex_usage_limit`, allowing Planner episodes to end as blocked/inconclusive
  while preserving stale execution artifacts for inspection.
- `general_prompts.py`: stores reusable prompt clauses shared by Planner, Writer, XML asset, and Critic calls,
  including the physical-causality contract and the Writer common API guide.
- `local_execution.py`: runs a generated Python entrypoint in a workspace, captures stdout/stderr, writes
  `execution_report.json`, and discovers artifacts.
- `execution.py`: adapts generated simulation projects to `local_execution.py`; it passes backend, render, timing, and
  artifact arguments to `src/main.py`. Local Genesis simulation subprocesses are serialized with a thread lock plus a
  per-user `/tmp` file lock so parallel cases do not contend for the GPU/renderer during execution.
- `suite.py`: reads `case_id|prompt` files, creates case workspaces, starts `PlannerSession`, and writes suite summary.
  It also builds the suite-level Genesis FEM+IPC context pack before cases start. Selected cases run concurrently by
  default; set `CONFIGS.harness.max_parallel_cases` or CLI `--max-parallel-cases` to cap them.
- `timing.py`: resolves Planner `execution_plan` values and explicit CLI overrides into steps, duration, fps, and target
  video frames.
- `integrator.py`: writes the stable `src/main.py` entrypoint that imports generated Scene, Body, Action, and Rendering
  modules. The entrypoint carries `CONFIGS.runtime` defaults into generated code for simulation dt, substeps, render
  cadence, fps, and camera resolution.

## Boundaries

Utilities do not reason about task quality and do not write generated simulation modules. Planner owns planning and
episode control, Writer owns generated code modules, Evaluation owns pass/fail judgment, and Utils owns repeatable
runtime mechanics.

Planner and Writer may request execution through the harness, but generated workers should not call `uv`, `pytest`, or
Genesis directly. All generated-code execution should go through `utils.execution`.

Writer batches requested by Planner may run concurrently. With the default
`CONFIGS.harness.max_parallel_workers=None`, the harness applies no artificial writer cap: every role included in one
`spawn_workers` action can run at the same time. Planner controls dependencies by choosing which writer roles appear in
the same action and should prefer maximal safe parallelism.
