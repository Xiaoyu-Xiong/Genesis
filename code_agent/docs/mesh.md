# Migrated Mesh Pipeline

`code_agent/assets/mesh/` contains a copied version of the legacy mesh pipeline from `agent/mesh`, excluding
`__pycache__`.

It is the mesh asset implementation used by `code_agent` during the first migration phase. The intent is to reuse the
existing Meshy API, repair, texture transfer, and validation flow rather than rewrite it.

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

The migrated copy defaults generated mesh outputs to `code_agent/generated_meshes`.

## Legacy Documentation

The original mesh flow is documented in [agent/docs/mesh_texture.md](../../agent/docs/mesh_texture.md). That document
also points back to this migrated copy.
