# Evaluation

`evaluation/` merges deterministic checks, visual evidence, and the read-only Codex Critic.

Flow:

1. `deterministic.py` checks execution exit status, JSON artifacts, metrics, and required render presence.
2. `visual.py` samples frames, writes a contact sheet, and summarizes texture/color evidence.
3. `agent.py` sends task text, source, contracts, assets, metrics, logs, render stats, visual evidence, stdout/stderr,
   and Genesis context to the Critic.
4. `runner.py` writes the merged `reports/critic_report.json`.

The critic is expected to judge task faithfulness, physical causality, visual clarity, and source-level repair
ownership. IPC initial-penetration/sanity-check failures are routed to Body because they usually mean invalid initial
placement or clearance.
