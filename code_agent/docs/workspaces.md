# Workspaces

`code_agent/workspaces/` will contain generated run workspaces.

Run workspaces can become large. Commit only curated examples or small fixtures intentionally.

## Standard Layout

```text
code_agent/workspaces/<run_id>/
  inputs/
    user_prompt.md
    repo_rules.md
    capabilities.md
  contracts/
    planner_output.json
    scene_brief.json
    scene_plan.json
    module_contracts.json
    asset_requests.json
  assets/
    asset_manifest.json
  src/
    scene.py
    body.py
    action.py
    main.py
  logs/
    codex_planner.jsonl
    codex_scene.jsonl
    codex_body.jsonl
    codex_action.jsonl
    codex_integrator.jsonl
    codex_review.jsonl
    execution.stdout
    execution.stderr
  artifacts/
    run_result.json
    event_log.json
    metrics.json
    render.mp4
    frames/
  reports/
    static_review.json
    execution_report.json
    critic.json
    patch_plan.json
  summary.json
```
