# Scripts and Suites

`code_agent/scripts/` contains `cases.txt` files and thin `run.sh` wrappers. A case line is:

```text
case_id|prompt
```

Direct CLI form:

```bash
uv run python -m code_agent.cli run-suite \
  --tasks-file code_agent/scripts/rigid_primitives/cases.txt \
  --out-dir code_agent/workspaces/suites/rigid_primitives/dev \
  --gpu --max-cases 1 --render
```

Useful flags:

- `--gpu` / `--cpu`
- `--max-cases N`
- `--max-parallel-cases N`
- `--render` / `--no-render`
- `--duration-sec N`, `--steps N`, `--render-fps N`
- `--repair-rounds N`
- `--timeout-sec N`
- `--enable-deformable` / `--disable-deformable`
- `--enable-ipc` / `--disable-ipc`

Modes:

- ordinary rigid: `--disable-deformable --disable-ipc`
- rigid+IPC: `--disable-deformable --enable-ipc`
- FEM+IPC: `--enable-deformable`

Execution uses the repository uv environment. Long background WSL runs should use explicit `PATH` and
`LD_LIBRARY_PATH` when launched through `systemd-run --user`; user services may not inherit interactive shell setup.

Mesh generation requires `MESHY_API_KEY` in the same non-interactive environment that launches the suite.
