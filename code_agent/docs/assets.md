# Assets

Planner can generate two asset families:

- mesh assets from `asset_type="generated_mesh"`
- XML/MJCF articulated assets from `asset_type="generated_xml"` or `asset_type="mjcf"`

Planner asset actions are:

- `start_mesh_assets`
- `wait_mesh_assets`
- `start_xml_assets`
- `wait_xml_assets`
- `inspect_assets`

Asset jobs run in the background so non-dependent writers can start while assets are being prepared. The wait actions
validate partial manifests and merge ready entries into `assets/asset_manifest.json`.
`inspect_assets` writes `reports/asset_inspection_report.json` plus preview images for ready mesh/XML assets so Planner
can distinguish body placement errors from asset topology, scale, or articulation defects.

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

For IPC-heavy rigid-rigid scenes, keep coupling choices mutually consistent. When `ipc_enable_rigid_rigid_contact` is
true, passive free rigid bodies should use `coup_type="ipc_only"`, actively driven contact mechanisms should use
`coup_type="external_articulation"`, and code generation should avoid `coup_type="two_way_soft_constraint"` in that
scene. Genesis rigid contact is not a fallback for `ipc_only` bodies because those links are skipped by the Genesis
rigid collider; rigid-rigid contact must be handled by IPC.

For MJCF/URDF assets that will be loaded as IPC `external_articulation`, every parent and child link participating in a
driven joint must have collision geometry. A logical fixed mount may use a tiny nonzero-volume dummy collision geom, and
that dummy may be a primitive MJCF geom such as `type="box"` rather than a mesh. It must remain a collision geom, so do
not set both `contype` and `conaffinity` to zero; `contype="1"` with `conaffinity="0"` is acceptable when the dummy
should not accept ordinary contact pairs. Place dummy mount geometry away from the real task contact region.
