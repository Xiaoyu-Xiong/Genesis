# Mesh Pipeline

`code_agent/assets/mesh/` contains the Meshy API, repair, texture transfer, and validation flow used by the code-agent
asset layer.

## Responsibilities

- Meshy preview generation.
- Optional Meshy texture refine.
- Manifold checks.
- Repair via fTetWild / pytetwild-backed flow.
- Texture transfer onto repaired meshes.
- Textured validation renders.

## Submodules

- `assets/mesh/workflow/`: Meshy API calls, stage helpers, and summaries.
- `assets/mesh/repair/`: manifold checks and repair.
- `assets/mesh/texture/`: UV generation, texture baking, OBJ/MTL rewriting, and validation renders.

## Main-Pipeline Integration

Planner exposes mesh generation through `start_mesh_assets`, `wait_mesh_assets`, and the compatibility
`generate_mesh_assets` action. These actions consume `planner_output.asset_requests` entries whose `asset_type` is
`generated_mesh`, run the Meshy/repair/texture pipeline, and write an episode-level `assets/asset_manifest.json`.

Planner can pass `asset_names` to generate a subset of requests, or leave it null/empty to generate all mesh requests.
`start_mesh_assets` launches the asset job in the background, allowing Planner to dispatch writer roles that do not need
the manifest while assets are still running. `wait_mesh_assets` blocks only when the next useful step requires canonical
mesh paths. `generate_mesh_assets` remains as a compatibility action that starts and waits in one turn.

The asset bridge splits Meshy work into two phases. Meshy API submission, polling, and asset downloads are parallel by
default for every selected asset (`CONFIGS.meshy_request.max_parallel_api_requests=None` means no cap within that
episode). Local processing then runs conservatively with `CONFIGS.meshy_request.max_parallel_local_processing=1`,
because manifold checks, fTetWild repair, UV parameterization, and texture baking can each allocate large transient
buffers and concurrent textured mesh post-processing can destabilize WSL. The bridge writes
`reports/asset_generation_report.json` before starting and updates it after API completions and each locally processed
asset, so interrupted runs leave a useful progress record.

## Meshy Environment Note

Mesh generation requires `MESHY_API_KEY` in the environment used by the non-interactive
`uv run python -m code_agent.cli` process. On this machine, `~/.bashrc` returns early for non-interactive shells before
the later API-key export line, so plain `source ~/.bashrc` from scripts or Codex commands may still leave
`MESHY_API_KEY` unset. For mesh suite runs, either export the key before invoking the suite or explicitly load the
`export MESHY_API_KEY=...` line from shell config in the same command environment. Do not print the key value in logs.

## Integration Contract

Scene and Body workers must not guess mesh output paths. The asset bridge must expose canonical runtime-ready paths
through `asset_manifest.json`.

For generated mesh bodies, use repaired mesh paths under `processed/` as runtime geometry. Treat raw textured OBJ files
as visual or texture sources, not as runtime collision geometry.

Meshy OBJ outputs are provider Y-up in the current pipeline. The bridge records this with
`file_meshes_are_zup=false` and precomputes Genesis scale factors from the requested bounding box after the Y-up to
Genesis Z-up conversion. Generated code should pass both fields directly into `gs.morphs.Mesh` instead of deriving
another orientation or scale.

Texture transfer writes `processed/base_color.png` and a seam-aware `processed/repaired_textured.obj` render mesh that
binds the rebaked texture through `processed/repaired_textured.mtl`. The strict-manifold `processed/repaired.obj`
remains the simulation/collision mesh. For rigid generated assets, generated scene/body code should instantiate one
entity with `gs.morphs.Mesh(file=runtime_path, visual_file=visual_path, ...)`, so the render mesh follows the same rigid
link transform without becoming a separate object. `texture_path` remains manifest evidence and a fallback location for
preview/debug checks.

Genesis flips Trimesh texture V coordinates during mesh import before handing UVs to the rasterizer. The
texture-transfer stage therefore canonicalizes textured render OBJs by flipping their `vt` V coordinate once and marking
the file with `# code_agent: genesis_texture_v_flipped`. This preserves the seam-split `f v/v` layout while matching
Genesis render orientation.

The mesh validation renderer follows the same default: it loads the OBJ/MTL pair directly. The optional `--texture`
argument is only a fallback/debug override for meshes that do not carry a usable material binding.

Generated mesh outputs for suite cases live under each case workspace at `assets/mesh/`.
