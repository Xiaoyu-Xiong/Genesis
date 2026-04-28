# Evaluation

`code_agent/evaluation/` is the future home of deterministic checks and critic orchestration.

## Responsibilities

- Validate schema outputs.
- Check worker write scopes.
- Check artifact completeness.
- Summarize execution reports.
- Run task metrics from generated code outputs.
- Call the single-pass Codex Critic on task text, metrics, event logs, frames, and video.
- Produce repair-ready failure classifications.

## Non-Goals

- Do not run Genesis directly outside [Execution](execution.md).
- Do not replace task metrics with video-only judgment.
- Do not implement a two-stage critic in the first version.
