# Codex

`code_agent/codex/` is the future home of the `codex exec` adapter.

## Responsibilities

- Build standardized `codex exec` command lines.
- Select sandbox mode from [configs.py](../configs.py).
- Attach role-specific prompts and schemas from [Specs](specs.md).
- Save JSONL event streams.
- Save final messages through `--output-last-message`.
- Record exit code, duration, command metadata, and Codex version.
- Return structured invocation results to [Orchestration](orchestration.md).

## Invocation Policy

- Planner and reviewer run read-only.
- Scene, Body, Action, Integrator, debugger, critic, and XML workers run with the narrowest useful write scope.
- Batch calls use `--ask-for-approval never`.
- Schema-producing calls use `--output-schema`.
- Calls use `--json` so events can be audited.

## Guardrails

- Do not let workers run host-side Python, `uv`, `pytest`, or Genesis simulations.
- Do not let writer prompts omit allowed write paths.
- Do not hide failed invocations.
