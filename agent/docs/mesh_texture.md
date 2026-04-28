# Mesh and Texture Pipeline

This document covers standalone mesh generation, repair, texture generation, texture transfer, and the current main-pipeline render integration.

## Standalone Mesh Pipeline

[agent/mesh](../mesh) contains the standalone mesh pipeline. The root keeps the CLI, shared models, and main
orchestration; implementation details live in three subdirectories:

- `workflow/`: Meshy API access, pipeline-stage helpers, and asset summaries
- `repair/`: manifold checks and fTetWild / pytetwild repair
- `texture/`: repaired-mesh UV generation, texture transfer, and textured validation renders

## Code-Agent Migration Note

The code-native pipeline has a migrated copy of this mesh implementation under
[code_agent/assets/mesh](../../code_agent/assets/mesh). That copy is documented in
[code_agent/docs/mesh.md](../../code_agent/docs/mesh.md) and is the mesh asset implementation used by the new
[code_agent plan](../../code_agent/agentive_code_pipeline_plan.md).

The migration intentionally preserves the existing Meshy, repair, texture-transfer, and validation responsibilities.
The new `code_agent` layer should only normalize asset requests, call the migrated mesh pipeline, and summarize outputs
into an Asset Manifest. MJCF/XML generation is not migrated from the legacy XML agent; the code-native plan routes it to
a dedicated Codex XML worker instead.

Legacy `agent/` runs still use this `agent/mesh` path and [agent/configs.py](../configs.py). Code-agent runs should use
[code_agent/configs.py](../../code_agent/configs.py) for static Meshy and repair defaults.

Main responsibilities:

- Meshy preview generation
- optional Meshy texture refine
- manifold check
- repair via fTetWild / pytetwild-backed pipeline
- repaired textured mesh export
- textured multi-view validation render

Public CLI:

- `agent.mesh.cli generate`
- `agent.mesh.cli manifold-check`
- `agent.mesh.cli render-textured-views`

Example:

```bash
uv run python -m agent.mesh.cli generate \
  --prompt "Create a single rubber duck with a readable toy-like texture." \
  --generate-texture \
  --out-dir agent/generated_meshes/example_duck \
  --out agent/generated_meshes/example_duck/result.json
```

## Mesh Asset Stages

For a textured mesh asset, the common on-disk structure is:

- `downloads/`: raw Meshy preview download
- `textured/`: raw textured OBJ/MTL/base color from Meshy refine
- `processed/`: repaired mesh and transferred texture used by the pipeline

Common output files:

- `processed/repaired.obj`
- `processed/repaired.mtl`
- `processed/base_color.png`
- `raw_manifold_check.json`
- `manifold_check.json`
- `repair.json`
- `repair_attempts.json`
- `metadata.json`

## Texture Flow

Current textured-mesh flow is:

1. Meshy preview generates raw geometry
2. optional Meshy refine generates raw textured OBJ assets
3. manifold / repair pipeline produces repaired surface geometry
4. texture is transferred onto the repaired mesh
5. repaired textured mesh is used for standalone validation renders
6. the main deformable pipeline reuses repaired textured data for render-side UV handling

For generated mesh bodies, the canonical runtime geometry path is the repaired mesh under `processed/`:

- use `processed/repaired.obj` as `bodies[].shape.file` in the main runtime IR
- treat `textured/model.obj` and its MTL / texture images as auxiliary texture assets, not as the main runtime mesh path

Current repaired-mesh rebake is no longer vertex-color-only. The active transfer path uses `xatlas` for target UV atlas generation, then rasterizes target-atlas texels, lifts each covered texel center back to a 3D point on the target surface, projects that point to the source textured mesh with `igl.point_mesh_squared_distance`, and samples the raw source texture there. This preserves raw texture detail much better than the older "sample color once per target vertex, then linearly interpolate inside each target triangle" path.

The deformable render path is no longer a separate experiment; it is wired into the active FEM mesh path.

Implementation is split by stage:

- `texture/transfer.py`: transfer orchestration and result packaging
- `texture/parameterization.py`: target UV unwrap via `xatlas`
- `texture/bake.py`: source-surface projection and per-texel baking
- `texture/obj_io.py`: OBJ / MTL rewriting and small mesh-file transforms
- `texture/render_views.py`: textured multi-view validation renders

## Current Recommended Repaired-Mesh Texture Path

The current recommended repaired-mesh texture path is:

1. `xatlas` unwrap on the repaired target mesh
2. true per-texel bake on the target atlas
3. source-surface projection with `igl.point_mesh_squared_distance`
4. texture-resolution cap from [agent/configs.py](../configs.py) to keep bake cost bounded

Older parameterization experiments using PyMeshLab-based LSCM, harmonic, Voronoi-atlas, or trivial-per-triangle UV generation are no longer part of the active path.

## Runtime Validation

For runtime-facing regressions, validate on the actual `agent.cli run` path or on a repo-local equivalent that uses the exact IR camera setup while bypassing IPC only when GPU-independent first-frame debugging is required.

## Mesh Prompting Notes

The current mesh-side prompting and validation rules emphasize:

- manifold-ready outputs
- reuse of identical geometry when visually acceptable
- mesh `scale` as the correct knob for global size correction
- density bounds and stable initial spacing
