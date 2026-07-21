# Mesh Pipeline

`assets/mesh/` turns Planner `generated_mesh` requests and procedural `cloth_mesh*` requests into Genesis-ready
manifest entries.

## Flow

1. `request_adapter.py` selects requests and builds concise Meshy prompts.
2. `episode.py` preserves reusable entries, generates selected assets, and applies metadata-only updates.
3. `pipeline.py` downloads, repairs, checks manifold geometry, and prepares texture provenance.
4. `remesh_integration.py` applies the automatic face budget or a Planner-requested remesh.
5. `validation.py` verifies the selected geometry as rigid, volumetric FEM, and/or FEM.Cloth input.
6. `manifest.py` writes the canonical entry consumed by Scene and Body workers.

Meshy-generated closed cloth shells use `source_type="cloth_mesh"` and the FEM.Cloth import path. Procedural open
cloth sheets use their dedicated `cloth_target_edge_length` generator and do not pass through the closed-manifold
remesher.

Local mesh processing is serial by default through `CONFIGS.meshy_request.max_parallel_local_processing=1` to bound
WSL memory peaks.

## Manifest Contract

- `runtime_path`: simulation and collision surface.
- `visual_path`: optional textured visual surface for the same entity.
- `texture_path`: base-color texture evidence.
- `scale`: runtime scale, normally a scalar uniform factor.
- `bbox`: runtime bounds after scale; Planner `bbox` is a target size, not a scale vector.
- `file_meshes_are_zup`: source coordinate convention.
- `asset_request`: source request used to decide whether later retries may reuse geometry.
- `validation`: topology, texture, remesh, and Genesis import results.
- `remesh`: original source provenance, requested target, achieved counts, report, and fallback status.

Use `update_mesh_asset_metadata` when geometry is acceptable and only scale or bounds need correction. Geometry,
texture, topology, or semantic-shape changes require regeneration.

## Automatic Remesh

After repair and texture preparation, every manifold-valid Meshy mesh is compared with the configured target plus its
upper tolerance. With the defaults, the skip limit is `5000 * (1 + 0.50) = 7500` faces.

- At or below `floor(target * (1 + tolerance))`: return `skipped_not_needed`; do not invoke PyMeshLab or create remesh
  output. Meshes below the target are never upsampled.
- Valid remesh: switch `runtime_path`, `visual_path`, and `texture_path` to the validated output.
- Failed remesh: return `failed_fallback_original` and run normal validation on the untouched original asset.
- Invalid source topology: skip remesh; downsampling is not a topology repair mechanism.

The target face count is approximate. The configured tolerance defaults to 50 percent. A candidate is accepted only
after target, manifold/winding/TetGen, texture, and required Genesis import checks pass.

## Planner Tool

`remesh_mesh_assets` lets Planner reduce a valid generated mesh that is still too expensive. It requires explicit
`asset_names` and exactly one of `target_face_count` or `target_edge_length`; runtime validation enforces this contract.
Planner chooses a new target after failure: the tool never silently relaxes or substitutes parameters.

Each request starts from recorded original provenance rather than chaining from a previous reduced output. Success
atomically updates the selected manifest entry. Failure returns its stage and report while leaving the entry unchanged;
multi-asset calls may commit successful entries independently.

Do not use remesh for a wrong silhouette, missing parts, bad articulation, invalid source topology, or prompt mismatch.
Those cases require asset regeneration.

## Texture And Import Validation

The physical `repaired.obj` and textured `repaired_textured.obj` may have different vertex counts because UV seams
duplicate visual vertices. Validation preserves a bounded many-to-one visual-to-physical mapping while keeping one
physical vertex at each location.

An accepted output must provide:

- manifold, winding-consistent, TetGen-ready physical geometry;
- a non-uniform baked base-color image and valid UV faces when texture was requested;
- rigid import using the physical mesh plus optional `visual_file`;
- volumetric FEM and FEM.Cloth imports with seam-aware visual mapping.

Outputs live under the asset root:

```text
remesh/auto_faces_5000/              # automatic attempt
remesh/planner_faces_TARGET/         # Planner face target
remesh/planner_edge_TARGET/          # Planner edge target
  remesh_report.json
  processed/
    repaired.obj
    repaired_textured.obj
    repaired_textured.mtl
    base_color.png
```

The low-level tool can also be tested directly:

```bash
uv run --no-sync python -m code_agent.assets.mesh.remesh \
  --input-mesh ASSET_ROOT/processed/repaired.obj \
  --output-dir code_agent/workspaces/manual_remesh_validation/my_asset \
  --target-face-count 3000 \
  --scale 1.0 \
  --no-file-meshes-are-zup \
  --tet-resolution 2
```

For textured input, provide the original Meshy textured OBJ, base-color image, and the repair
`centroid_before_translation` alignment together. Omitting any member of that provenance is a hard failure rather than
permission to drop the texture.
