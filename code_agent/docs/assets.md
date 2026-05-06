# Assets

Planner can generate two asset families:

- mesh assets from `asset_type="generated_mesh"`
- XML/MJCF articulated assets from `asset_type="generated_xml"` or `asset_type="mjcf"`

The only Planner asset actions are:

- `start_mesh_assets`
- `wait_mesh_assets`
- `start_xml_assets`
- `wait_xml_assets`

Asset jobs run in the background so non-dependent writers can start while assets are being prepared. The wait actions
validate partial manifests and merge ready entries into `assets/asset_manifest.json`.

## Mesh

`assets/mesh/episode.py` selects mesh requests, calls Meshy, repairs and validates geometry, transfers texture metadata,
checks Genesis FEM import readiness, and writes:

- `assets/asset_manifest.json`
- `reports/asset_generation_report.json`

Mesh writers must instantiate the manifest entry directly:

- `runtime_path`: simulation/collision mesh
- `visual_path`: textured render mesh for the same Genesis entity
- `texture_path`: evidence metadata
- `scale` and `file_meshes_are_zup`: pass through to `gs.morphs.Mesh`

Do not create separate simulation and visual entities for one generated object.

## XML / MJCF

`assets/xml/episode.py` runs one worker per XML/MJCF request. Each asset must be one articulated body tree with its own
joints and actuators, not a full scene. Validation checks XML structure, MuJoCo import, static previews, and a small
actuator-response probe.

The manifest entry exposes actuator names, joint names, control ranges, base behavior, and placement assumptions so
Body and Action workers do not reverse-engineer semantics from raw XML.
