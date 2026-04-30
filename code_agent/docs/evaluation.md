# Evaluation

`code_agent/evaluation/` contains deterministic checks and the single-pass Codex Critic wrapper.

## Responsibilities

- Validate schema outputs.
- Check worker write scopes.
- Check artifact completeness.
- Summarize execution reports.
- Run task metrics from generated code outputs.
- Create lightweight visual evidence from rendered frames and generated asset textures.
- Call the single-pass Codex Critic on task text, generated source, Genesis context, planner contracts, asset manifests,
  metrics, event logs, render stats, visual evidence, stdout/stderr, and execution evidence.
- Produce repair-ready failure classifications.

## Current Behavior

`deterministic.py` reads `reports/execution_report.json`, checks successful exit, validates discovered
`summary.json` and `metrics.json`, and treats render artifacts as optional unless explicitly required. Its public
entry point is `evaluate_artifacts()`.

`visual.py` samples rendered frames, writes `reports/visual_contact_sheet.jpg`, summarizes sampled frame colors, reads
generated mesh texture metadata from `assets/asset_manifest.json`, and reports whether saturated texture colors are
underrepresented in sampled frames. It does not make semantic pass/fail decisions by itself; it gives the critic visual
evidence to combine with task metrics and execution logs.

`codex_critic.py` calls a read-only Codex Critic with task text, deterministic results, complete generated source,
Genesis context, planner output, timing contracts, asset manifests, execution reports, metrics, event logs, render
stats, stdout/stderr, and `visual_evaluation.json`. When a contact sheet exists, it is attached to the Codex call as an
image so the critic can inspect rendered appearance directly. The critic is instructed to compare the prompt, source,
numeric evidence, and visual result, then return detailed source-aware repair guidance in JSON matching
`critic_report.schema.json`.

`runner.py` is the active top-level evaluator. It combines artifact checks with the single-pass Codex Critic and writes
the merged `reports/critic_report.json`.

The current render check validates that a render artifact exists and is non-empty when rendering is required. The Codex
Critic can also consume `render_stats.json`, event-log evidence, sampled frame paths, an attached contact-sheet image,
texture-presence warnings, and the generated source that produced those artifacts. This prevents
texture/orientation/source-contract issues from being judged on numeric logs alone.

## Rendering Responsibility

Final render setup should be authored by a dedicated Rendering Worker, not hand-coded in generic utilities. Evaluation
should consume the Rendering Worker's declared artifact contract and validation hints, then combine those render signals
with metrics and event logs. If rendering fails, repair should route to Rendering Worker unless the failure evidence
points to missing bodies, bad stage layout, or incorrect action timing.

## Non-Goals

- Do not run Genesis directly outside [Utils](utils.md) execution helpers.
- Do not replace task metrics with video-only judgment.
- Do not implement a two-stage critic in the first version.
