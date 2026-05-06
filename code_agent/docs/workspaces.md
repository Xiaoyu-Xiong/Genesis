# Workspaces

Generated runs live under `code_agent/workspaces/`. They can become large; commit only curated fixtures or examples.

Typical case layout:

```text
inputs/
contracts/
assets/
src/
logs/
reports/
artifacts/
summary.json
```

Key files:

- `contracts/planner_output.json`
- `contracts/timing.json`
- `contracts/deformable_config.json`
- `assets/asset_manifest.json`
- `src/scene.py`, `body.py`, `action.py`, `rendering.py`, `main.py`
- `reports/episode_state.json`
- `reports/planner_actions.jsonl`
- `reports/dispatch_history.jsonl`
- `reports/execution_report.json`
- `reports/critic_report.json`
- `artifacts/metrics.json`
- `artifacts/event_log.json`
- `artifacts/render.mp4`

Suite roots also contain a shared `context/genesis/` cache copied into each case contract directory.
