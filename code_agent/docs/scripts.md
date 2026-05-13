# Scripts and Suites

`code_agent/scripts/` contains `cases.txt` files and thin `run.sh` wrappers. A case line is:

```text
case_id|prompt
```

Cases may optionally inject a structured layout file by putting an `@layout` directive inside the prompt:

```text
case_id|@layout relative/path/to/layout.json prompt text...
```

The path is resolved relative to the `cases.txt` file. If no `@layout` directive is present, suite loading behaves as
before. When present, the layout is copied into the case workspace under `inputs/layout.*`, summarized in
`inputs/layout_context.md`, and injected into Planner and worker prompts so generated modules can reuse source-derived
coordinates, mesh dimensions, and initial poses without relying on natural-language reconstruction.

JSON layouts may also declare reusable mesh assets in a top-level `reusable_assets` list. Each asset can point to a
local file or to a GitHub-hosted file; declared textures/material sidecars are copied or downloaded with the mesh.
Layout assets are never repaired or post-processed. The suite runs the same read-only manifold/tetgen sanity check used
by generated meshes and records the result in the manifest. A `repo_asset` is marked ready when the file was
materialized and is loadable; strict manifold/tetgen failures are surfaced as validation warnings so original rigid
surface meshes can still be reused deliberately.

```json
{
  "reusable_assets": [
    {
      "logical_name": "source_chain_link",
      "repo": "https://github.com/ipc-sim/rigid-ipc",
      "ref": "main",
      "mesh": "meshes/wrecking-ball/link.obj",
      "texture": "meshes/wrecking-ball/link_base_color.png",
      "scale": [1.0, 1.0, 1.0],
      "bbox": [1.0, 1.5, 0.190211],
      "file_meshes_are_zup": false,
      "simulation_role": "source chain link collision mesh"
    }
  ]
}
```

For local assets, set `"mesh": "relative/path/to/mesh.obj"`; paths are resolved relative to the layout file. For GitHub
assets, either provide a raw/blob URL directly or provide `repo` + `ref` + relative `mesh` path. Optional `visual`,
`material`, `texture`, and `textures` references follow the same rules.

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
