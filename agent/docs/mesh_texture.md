# Mesh and Texture Pipeline

This document covers standalone mesh generation, repair, texture generation, texture transfer, and the current main-pipeline render integration.

## Standalone Mesh Pipeline

[agent/mesh](../mesh) contains the standalone mesh pipeline.

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

Current repaired-mesh rebake is no longer vertex-color-only. The active transfer path prefers `xatlas` for target UV atlas generation, falls back to older parameterization paths only if needed, then rasterizes target-atlas texels, lifts each covered texel center back to a 3D point on the target surface, projects that point to the source textured mesh with `igl.point_mesh_squared_distance`, and samples the raw source texture there. This preserves raw texture detail much better than the older "sample color once per target vertex, then linearly interpolate inside each target triangle" path.

The current recommended validation loop for deformable-texture changes is to render the no-IPC first frame with the exact IR camera specification, using:

- [agent/scripts/_temp_render_firstframe_noipc.py](../scripts/_temp_render_firstframe_noipc.py)

This keeps the runtime render path honest without requiring the full IPC stack just to inspect the initial textured frame.

The deformable render path is no longer a separate experiment; it is wired into the active FEM mesh path.

## Main-Pipeline Texture Debugging

There is a dedicated debug utility:

- [agent/scripts/debug_main_pipeline_texture.py](../scripts/debug_main_pipeline_texture.py)

This is used to inspect:

- remesh-stage texture handling
- TetGen boundary-stage texture handling
- render-side asset construction used by the deformable FEM path

## Mesh Prompting Notes

The current mesh-side prompting and validation rules emphasize:

- manifold-ready outputs
- reuse of identical geometry when visually acceptable
- mesh `scale` as the correct knob for global size correction
- density bounds and stable initial spacing
