# Mesh Pipeline

`assets/mesh/` turns Planner `generated_mesh` requests into Genesis-ready manifest entries.

Flow:

1. `request_adapter.py` selects requests and builds concise Meshy prompts.
2. `episode.py` preserves unselected ready mesh entries, regenerates selected geometry requests, applies explicit
   metadata-only updates on request, and writes progress reports.
3. `pipeline.py` downloads, repairs, validates manifold geometry, transfers texture metadata, and records profiles.
4. `validation.py` checks whether the repaired mesh can be imported as Genesis FEM geometry.
5. `manifest.py` writes the canonical entry consumed by Scene/Body workers.

Important manifest fields:

- `runtime_path`: strict simulation/collision mesh
- `visual_path`: optional textured visual mesh for the same entity
- `texture_path`: texture evidence
- `scale`: Genesis runtime scale. Prefer a scalar uniform factor; length-3 vectors are allowed only for legacy entries
  or explicit equal-component uniform scale.
- `bbox`: runtime bbox after the manifest scale. Planner `bbox` is an approximate target size, not a scale vector.
- `file_meshes_are_zup`: coordinate convention
- `asset_request`: source Planner request used to decide whether future retries can reuse the mesh geometry
- `validation`: manifold, texture, and Genesis import status

Use Planner action `update_mesh_asset_metadata` when a ready generated mesh's geometry is acceptable and only sizing
metadata needs to change. The command reuses the existing `runtime_path`, refreshes scalar uniform `scale`/`bbox`, and
reruns Genesis FEM import validation without making another Meshy generation request. `start_mesh_assets` remains the
geometry-generation path for prompt, mesh shape, texture, topology, or role changes.

Local mesh processing is intentionally serial by default through
`CONFIGS.meshy_request.max_parallel_local_processing=1` to keep WSL memory peaks bounded.
