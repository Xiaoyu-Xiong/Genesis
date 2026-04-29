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

## Integration Contract

Scene and Body workers must not guess mesh output paths. The asset bridge must expose canonical runtime-ready paths
through `asset_manifest.json`.

For generated mesh bodies, use repaired mesh paths under `processed/` as runtime geometry. Treat raw textured OBJ files
as visual or texture sources, not as runtime collision geometry.

Generated mesh outputs default to `code_agent/generated_meshes`.
