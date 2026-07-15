# End-to-End Codex Baseline

This baseline runs one Codex agent per case. The agent acts as the merged planner/writer/repair loop: it writes the
complete generated simulation under the case workspace, optionally declares Meshy `generated_mesh` asset requests in
`contracts/planner_output.json`, may call the Meshy asset pipeline itself, may run locked Genesis simulations while it
debugs, and then returns a final worker-style report.

Example:

```bash
uv run --no-sync python -m baselines.end_to_end_codex.cli run-suite \
  --tasks-file code_agent/scripts/rigid_primitives/cases.txt \
  --out-dir code_agent/workspaces/baselines/e2e_rigid_primitives \
  --max-parallel-cases 4 \
  --gpu
```

Codex calls may run in parallel across cases. Any simulation launched by the agent should go through:

```bash
uv run --no-sync python -m baselines.end_to_end_codex.case_tools run-simulation --case-dir <case-dir> --backend gpu
```

That command delegates to `code_agent.utils.execution.run_generated_simulation`, so simulation execution remains
serialized by the shared lock even when multiple baseline cases are generating/debugging in parallel. Meshy requests can
be launched with:

```bash
uv run --no-sync python -m baselines.end_to_end_codex.case_tools generate-mesh-assets --case-dir <case-dir>
```

The suite runner still performs a final official execution after the Codex call finishes, so summaries stay comparable.
