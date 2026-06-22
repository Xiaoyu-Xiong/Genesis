# Utils

`utils/` contains runtime mechanics shared by Planner, writers, evaluation, and CLI.

- `codex.py`: non-interactive `codex exec` wrapper with schema output, logs, final messages, stderr, timeout, model,
  sandbox, reasoning effort, service-tier handling, quota wait/resume, and optional multi-account rotation.
- `suite.py`: loads cases, creates workspaces, builds Genesis context, starts `PlannerSession`, and writes suite
  summaries.
- `integrator.py`: writes stable `src/main.py` and passes runtime defaults plus `deformable_cfg` to generated modules.
- `adaptive_ipc.py`: computes the runtime adaptive IPC `contact_d_hat` report used by generated `src/main.py`, including
  mesh, primitive, MJCF/XML, and bbox fallback candidates.
- `execution.py`: runs generated projects through uv and serializes local Genesis subprocesses with a process lock.
- `local_execution.py`: captures stdout/stderr, execution metadata, and artifact paths.
- `timing.py`: resolves Planner timing plus CLI overrides.

Generated workers should not run Genesis or mutate the environment directly; execution goes through these utilities.
Prompt text and prompt builders live in `code_agent/prompts/`, not `utils/`.

## Codex Quota Recovery

`utils/codex.py` catches `codex_usage_limit` failures centrally. When one account hits quota, the wrapper tries the
next configured account. If every configured account is quota-limited, it pauses and periodically probes all accounts
until one can run again, then retries the original Codex request with the recovered account.

Prepare account profiles outside the repo by logging each account into a separate `CODEX_HOME`, for example:

```bash
CODEX_HOME=/secure/codex/main codex login
CODEX_HOME=/secure/codex/backup codex login
```

Then point the suite at those profiles:

```bash
export CODE_AGENT_CODEX_ACCOUNTS="main=/secure/codex/main;backup=/secure/codex/backup"
```

For longer runs, a JSON file is easier to manage:

```json
{
  "accounts": [
    { "name": "main", "codex_home": "/secure/codex/main" },
    { "name": "backup", "codex_home": "/secure/codex/backup" }
  ]
}
```

Use it with `CODE_AGENT_CODEX_ACCOUNT_FILE=/secure/codex/accounts.json`. Do not store auth directories or account files
with secrets in the repository. Quota probe timing can be tuned with `CODE_AGENT_CODEX_QUOTA_PROBE_INTERVAL_SEC`,
`CODE_AGENT_CODEX_QUOTA_PROBE_INITIAL_DELAY_SEC`, and `CODE_AGENT_CODEX_QUOTA_PROBE_TIMEOUT_SEC`; set
`CODE_AGENT_CODEX_QUOTA_WAIT=0` to restore fail-fast behavior.
