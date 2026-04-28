# Orchestration

`code_agent/orchestration/` is the future home of the deterministic run coordinator.

## Responsibilities

- Create and validate run workspaces.
- Write input bundles such as user prompt, repository rules, and capability summaries.
- Dispatch Planner, Scene, Body, Action, Integrator, reviewer, debugger, critic, and XML workers through the Codex layer.
- Validate schema outputs from [Specs](specs.md).
- Enforce worker write scopes.
- Route asset requests to [Assets](assets.md).
- Apply retry budgets and owner-routed repair decisions.
- Produce final run summaries.

## Boundaries

- Do not implement LLM reasoning here.
- Do not generate simulation code here.
- Do not execute Genesis directly; use [Execution](execution.md).
- Do not let Codex workers bypass schema validation or write-scope checks.
