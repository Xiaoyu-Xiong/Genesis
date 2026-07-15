# Assets

Planner can generate these asset families:

- mesh assets from `asset_type="generated_mesh"`
- procedural FEM.Cloth surface assets from `asset_type="cloth_mesh*"`
- XML/MJCF articulated assets from `asset_type="generated_xml"` or `asset_type="mjcf"`

Planner asset actions are:

- `start_mesh_assets`
- `wait_mesh_assets`
- `update_mesh_asset_metadata`
- `start_xml_assets`
- `wait_xml_assets`
- `inspect_assets`

Asset jobs run in the background so non-dependent writers can start while assets are being prepared. The wait actions
validate partial manifests and merge ready entries into `assets/asset_manifest.json`.
`inspect_assets` writes `reports/asset_inspection_report.json` plus preview images for ready mesh/XML assets so Planner
can distinguish body placement errors from asset topology, scale, or articulation defects.

## Built-In Genesis Assets

Code-agent cases must not use prepackaged files under `genesis/assets`. Agents should build rigid objects from
primitives, request generated XML/MJCF assets, request generated Meshy assets, or use explicit user-provided layout
assets copied into the case workspace. XML/MJCF may reference mesh files only when those files are generated
case-workspace assets, not Genesis prepackaged assets.

This is enforced in two places:

- Codex planner, writer, critic, XML, and Opt invocations run with `genesis/assets` hidden by a `bwrap` sandbox when
  `CONFIGS.codex.hide_builtin_assets_from_agents` is true.
- Planner outputs and generated `src/*.py` files are statically rejected if they reference `genesis/assets`,
  `get_assets_dir()`, Genesis package-path asset derivations such as `gs.__file__`, or built-in-style relative paths
  like `xml/...`, `urdf/...`, and `meshes/...`.

## Mesh

`assets/mesh/episode.py` selects mesh requests, calls Meshy, repairs and validates geometry, transfers texture metadata,
checks Genesis FEM import readiness, and writes:

- `assets/asset_manifest.json`
- `reports/asset_generation_report.json`

Mesh writers must instantiate the manifest entry directly:

- `runtime_path`: simulation/collision mesh
- `visual_path`: textured render mesh for the same Genesis entity
- `texture_path`: evidence metadata
- `scale` and `file_meshes_are_zup`: pass through to `gs.morphs.Mesh`; generated mesh `scale` should normally be a
  scalar uniform runtime factor
- `bbox`: runtime bbox after scale, for placement and sizing checks
- `asset_request`: source request used for safe metadata-only updates

Do not create separate simulation and visual entities for one generated object.
If only generated mesh sizing metadata is wrong, Planner should use `update_mesh_asset_metadata`. That action reuses the
ready mesh entry, updates scalar uniform scale/bbox metadata, and reruns Genesis FEM import validation without another
Meshy request. Prompt, geometry, texture, or role changes should still regenerate through `start_mesh_assets`.

For FEM.Cloth assets, simple sheets, ribbons, cylinders, and spheres should use the procedural `cloth_mesh*` asset
types. Complex closed manifold cloth shells may use `asset_type="generated_mesh"` when `purpose` or `simulation_role`
explicitly identifies the asset as a `FEM.Cloth closed manifold shell` or `cloth shell`; the Meshy pipeline then writes
the ready entry back as `source_type="cloth_mesh"` and validates it through the FEM.Cloth import path.

## XML / MJCF

`assets/xml/episode.py` runs one worker per XML/MJCF request. Each asset must be one body tree, not a full scene.
Actively controlled mechanisms need named non-free joints and actuators. XML/MJCF assets may use primitive geoms or
generated case-workspace mesh files. Mesh references are validated so they cannot point at `genesis/assets`, repository
sample meshes, external URIs, or paths outside the generated XML/case asset roots. Explicit passive projectiles may
instead be a single movable body with one named freejoint, collision geoms, and no actuators; validation accepts this
only when the source asset request clearly describes a passive/free rigid projectile.

Validation checks XML structure, MuJoCo import, static previews, and a small actuator-response probe for actuated
assets. The actuator-response probe is skipped, with a positive validation note, for accepted passive freejoint assets.

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
