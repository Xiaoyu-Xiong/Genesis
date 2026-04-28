# Evaluation

`code_agent/evaluation/` contains deterministic checks and the single-pass Codex Critic wrapper.

## Responsibilities

- Validate schema outputs.
- Check worker write scopes.
- Check artifact completeness.
- Summarize execution reports.
- Run task metrics from generated code outputs.
- Call the single-pass Codex Critic on task text, metrics, event logs, render stats, and execution evidence.
- Produce repair-ready failure classifications.

## Current MVP Behavior

`deterministic.py` reads `reports/execution_report.json`, checks successful exit, validates discovered
`summary.json` and `metrics.json`, and treats render artifacts as optional unless explicitly required.

`codex_critic.py` calls a read-only Codex Critic with task text, deterministic results, execution report excerpts,
metrics, event log excerpts, and render stats. The critic returns JSON matching `critic_report.schema.json`. Direct
frame/video semantic sampling is still future work.

`simple.py` is the active top-level evaluator. It combines deterministic checks with the single-pass Codex Critic and
writes the merged `reports/critic_report.json`. The two-stage critic is intentionally not implemented.

The current render check validates that a render artifact exists and is non-empty when rendering is required. The Codex
Critic can also consume `render_stats.json` and event-log evidence. This is still lighter than a full semantic video
critic.

## Rendering Responsibility

Final render setup should be authored by a dedicated Rendering Worker, not hand-coded in the coordinator. Evaluation
should consume the Rendering Worker's declared artifact contract and validation hints, then combine those render signals
with metrics and event logs. If rendering fails, repair should route to Rendering Worker unless the failure evidence
points to missing bodies, bad stage layout, or incorrect action timing.

## Non-Goals

- Do not run Genesis directly outside [Execution](execution.md).
- Do not replace task metrics with video-only judgment.
- Do not implement a two-stage critic in the first version.
