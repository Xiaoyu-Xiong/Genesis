# Mesh Pipeline

`assets/mesh/` turns Planner `generated_mesh` requests into Genesis-ready manifest entries.

Flow:

1. `request_adapter.py` selects requests and builds concise Meshy prompts.
2. `episode.py` preserves unselected ready mesh entries, regenerates selected requests, and writes progress reports.
3. `pipeline.py` downloads, repairs, validates manifold geometry, transfers texture metadata, and records profiles.
4. `validation.py` checks whether the repaired mesh can be imported as Genesis FEM geometry.
5. `manifest.py` writes the canonical entry consumed by Scene/Body workers.

Important manifest fields:

- `runtime_path`: strict simulation/collision mesh
- `visual_path`: optional textured visual mesh for the same entity
- `texture_path`: texture evidence
- `scale`: Genesis scale factors
- `file_meshes_are_zup`: coordinate convention
- `validation`: manifold, texture, and Genesis import status

Local mesh processing is intentionally serial by default through
`CONFIGS.meshy_request.max_parallel_local_processing=1` to keep WSL memory peaks bounded.
