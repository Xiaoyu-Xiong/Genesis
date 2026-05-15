# Utils

`utils/` contains runtime mechanics shared by Planner, writers, evaluation, and CLI.

- `codex.py`: non-interactive `codex exec` wrapper with schema output, logs, final messages, stderr, timeout, model,
  sandbox, reasoning effort, and service-tier handling.
- `suite.py`: loads cases, creates workspaces, builds Genesis context, starts `PlannerSession`, and writes suite
  summaries.
- `integrator.py`: writes stable `src/main.py` and passes runtime defaults plus `deformable_cfg` to generated modules.
- `adaptive_ipc.py`: computes the runtime adaptive IPC `contact_d_hat` report used by generated `src/main.py`, including
  mesh, primitive, MJCF/XML, and bbox fallback candidates.
- `execution.py`: runs generated projects through uv and serializes local Genesis subprocesses with a process lock.
- `local_execution.py`: captures stdout/stderr, execution metadata, and artifact paths.
- `timing.py`: resolves Planner timing plus CLI overrides.
- `general_prompts.py`: shared Planner/Writer/Critic prompt contracts.

Generated workers should not run Genesis or mutate the environment directly; execution goes through these utilities.
