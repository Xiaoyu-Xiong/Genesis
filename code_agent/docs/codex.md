# Codex

`code_agent/codex/` contains the `codex exec` adapter used by orchestration.

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
- Planner, reviewer, critic, and current Scene/Body/Action/Rendering writers run read-only.
- Current writers do not edit files. They return complete module source in the `source_code` field of
  `worker_report.schema.json`; orchestration writes accepted source to disk.
- Future debugger/XML workers should still use the narrowest useful sandbox for their task.
- Batch calls use non-interactive `codex exec`; the current CLI version does not expose `--ask-for-approval`, so
  approval behavior is controlled by the configured sandbox and Codex config.
- Schema-producing calls use `--output-schema`.
- Calls use `--json` so events can be audited.

## Current MVP Behavior

`run_codex_exec()` writes JSONL events, a final-message file, and stderr logs. `codex-mode auto` records planner
success or failure in `contracts/planner_output.json`; `codex-mode required` should be used when planner failure must
stop a run.

Writer generation is controlled by `--generation-mode codex|fallback`, not by `--codex-mode`. In `codex` mode the
dispatcher calls four writer prompts and parses their schema-constrained final JSON. In `fallback` mode the deterministic
generator is used without writer calls.

## Guardrails

- Do not let workers run host-side Python, `uv`, `pytest`, or Genesis simulations.
- Do not let writer prompts imply that Codex should edit files directly in the current implementation.
- Do not hide failed invocations.
