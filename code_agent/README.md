# Code Agent

`code_agent/` turns a natural-language case into runnable Genesis code, artifacts, metrics, render output, and critic
reports.

The active runtime is Planner-led:

1. `utils/suite.py` reads `case_id|prompt` cases and creates one workspace per case. A prompt may optionally contain
   `@layout path/to/layout.json`; the path is resolved relative to the `cases.txt` file and injected into Planner and
   worker prompts as source-derived layout context. JSON layouts can also declare `reusable_assets` that reference local
   or GitHub mesh/texture files; those assets are copied or downloaded verbatim, sanity-checked once, and exposed as
   `repo_asset` entries in `assets/asset_manifest.json`.
2. `planner/session.py` runs Planner turns. Planner emits one JSON action at a time.
3. `planner/action_handlers/` executes actions: start/wait asset jobs, spawn writers, integrate, run, evaluate, repair,
   or finish.
4. `writer/` owns four generated modules: `scene.py`, `body.py`, `action.py`, and `rendering.py`.
5. `evaluation/` combines deterministic artifact checks, visual evidence, and the read-only Codex Critic.

## Physics Modes

The suite exposes two switches:

- `--enable-deformable` / `--disable-deformable`
- `--enable-ipc` / `--disable-ipc`

Supported modes:

- ordinary rigid: deformable off, IPC off
- rigid/articulated + IPC contact: deformable off, IPC on
- FEM deformable + IPC: deformable on, IPC forced on

Generated code receives the effective contract at `contracts/deformable_config.json` as `deformable_cfg`.

## Assets

Planner can start mesh or XML/MJCF asset jobs in the background:

- `start_mesh_assets` / `wait_mesh_assets`
- `start_xml_assets` / `wait_xml_assets`
- `inspect_assets` for ready asset previews and geometry summaries

Ready assets are merged into `assets/asset_manifest.json`. Writers must use manifest paths and metadata rather than
guessing filesystem locations.

Layouts can provide ready reusable mesh assets without invoking text-to-mesh generation. These entries use
`source_type="repo_asset"` and preserve the source file bytes; sanity-check results are reported in
`reports/layout_asset_report.json` and embedded in each manifest entry.

## Run

Use the repository uv environment:

```bash
uv run python -m code_agent.cli run-suite \
  --tasks-file code_agent/scripts/rigid_primitives/cases.txt \
  --out-dir code_agent/workspaces/suites/rigid_primitives/dev \
  --gpu --max-cases 1 --render
```

For rigid+IPC:

```bash
uv run python -m code_agent.cli run-suite \
  --tasks-file code_agent/scripts/siggraph_paper_demos/cases.txt \
  --out-dir code_agent/workspaces/suites/siggraph_paper_demos/dev \
  --disable-deformable --enable-ipc --gpu --max-cases 1 --render
```

For FEM+IPC:

```bash
uv run python -m code_agent.cli run-suite \
  --tasks-file code_agent/scripts/deformable_primitives/cases.txt \
  --out-dir code_agent/workspaces/suites/deformable_primitives/dev \
  --enable-deformable --gpu --max-cases 1 --render
```

More details live in [docs](docs/README.md).
