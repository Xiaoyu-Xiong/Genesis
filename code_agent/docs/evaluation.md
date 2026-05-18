# Evaluation

`evaluation/` merges deterministic checks, visual evidence, and the read-only Codex Critic.

Flow:

1. `deterministic.py` checks execution exit status, JSON artifacts, metrics, and required render presence.
2. `visual.py` samples frames, writes a contact sheet, and summarizes texture/color evidence.
3. `agent.py` sends task text, source, contracts, assets, generated asset source/preview evidence, metrics, logs,
   render stats, visual evidence, stdout/stderr, and Genesis context to the Critic.
4. `runner.py` writes the merged `reports/critic_report.json`.

The critic is expected to judge task faithfulness, physical causality, visual clarity, generated asset morphology, and
source-level repair ownership. If a generated mesh/XML/MJCF asset is intrinsically unsuitable for the task, the critic
should route to `planner` with `asset_diagnostics` so Planner can rewrite the affected asset request and rerun the
appropriate asset action. IPC initial-penetration/sanity-check failures are routed to Body only when the ready assets
look plausible and the evidence points to placement or clearance rather than asset topology/shape/scale.
