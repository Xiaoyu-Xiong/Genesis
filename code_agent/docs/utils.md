# Utils

`code_agent/utils/` contains non-agent support code used by Planner, Writer, Evaluation, and the CLI.

## Modules

- `codex.py`: standardizes non-interactive `codex exec` calls. It builds the command, applies sandbox/model/schema
  settings, writes JSONL event logs, captures the final message and stderr, and returns `CodexExecResult`.
- `local_execution.py`: runs a generated Python entrypoint in a workspace, captures stdout/stderr, writes
  `execution_report.json`, and discovers artifacts.
- `execution.py`: adapts generated simulation projects to `local_execution.py`; it passes backend, render, timing, and
  artifact arguments to `src/main.py`.
- `suite.py`: reads `case_id|prompt` files, creates case workspaces, starts `PlannerSession`, and writes suite summary.
- `timing.py`: resolves Planner `execution_plan` values and explicit CLI overrides into steps, duration, fps, and target
  video frames.
- `integrator.py`: writes the stable `src/main.py` entrypoint that imports generated Scene, Body, Action, and Rendering
  modules.

## Boundaries

Utilities do not reason about task quality and do not write generated simulation modules. Planner owns planning and
episode control, Writer owns generated code modules, Evaluation owns pass/fail judgment, and Utils owns repeatable
runtime mechanics.

Planner and Writer may request execution through the harness, but generated workers should not call `uv`, `pytest`, or
Genesis directly. All generated-code execution should go through `utils.execution`.
