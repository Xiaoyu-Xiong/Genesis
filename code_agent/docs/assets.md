# Assets

`code_agent/assets/` owns asset-facing implementation.

The current Planner-connected asset implementations are the mesh pipeline under `assets/mesh` and the MJCF/XML
pipeline under `assets/xml`. Planner can launch either asset family in the background, dispatch writer roles that do
not need the manifest yet, and then wait for the relevant asset job before manifest-dependent code generation or
integration.

## Mesh Assets

Mesh requests are routed to [Mesh Pipeline](mesh.md). Planner writes `asset_requests` with
`asset_type="generated_mesh"` and then starts or waits on the mesh asset flow according to the dependency state.

The top-level asset bridge is intentionally thin. It keeps the Planner-facing import path stable and delegates concrete
mesh work to `code_agent/assets/mesh/episode.py`, which normalizes selected requests, calls Meshy generation, runs
repair/manifold checks and optional texture transfer, and writes:

- `assets/asset_manifest.json`
- `reports/asset_generation_report.json`

Meshy API submission, polling, and downloads run in parallel for all selected assets by default. Local mesh processing
runs one asset at a time through `CONFIGS.meshy_request.max_parallel_local_processing=1`; this avoids stacking local
fTetWild repair, UV parameterization, and texture baking memory peaks when a Planner requests several textured meshes in
one episode. The report file is created at the start of generation and updated after API completions and each locally
processed asset, so a WSL or process interruption should still show the last completed asset and current phase.

Planner can launch this flow with `start_mesh_assets` and continue dispatching writer roles that do not require the
asset manifest. Roles whose module contracts list asset dependencies or explicit `asset_manifest` inputs are held until
`wait_mesh_assets` completes and the manifest validates. The older `generate_mesh_assets` action is still available for
intentionally blocking asset generation.

Scene and Body workers receive the manifest in their prompts. They should use ready `generated_mesh` entries directly
instead of guessing paths or searching the filesystem.

Generated Meshy OBJ assets are treated as provider Y-up unless the manifest says otherwise. Planner asset `scale` and
`bbox` fields are positive XYZ dimensions in meters, never positions or signed bounds. The mesh manifest adapter
converts raw mesh bounds into Genesis coordinates, emits Genesis scale factors rather than raw target dimensions, and
records `file_meshes_are_zup=false` for those assets. Writers must pass both `scale` and `file_meshes_are_zup` through
to `gs.morphs.Mesh`.

When `visual_path` and `texture_path` are present, `runtime_path` remains the strict-manifold simulation/collision mesh
and `visual_path` points to the seam-aware textured render mesh for the same logical asset. Rigid generated code should
instantiate one Genesis entity with `gs.morphs.Mesh(file=runtime_path, visual_file=visual_path, ...)`; it must not create
separate simulation and visual entities for the same generated object. `texture_path` is retained as evidence and a
preview/debug fallback for the transferred base-color image. The visual OBJ carries a one-time Genesis texture-V
canonicalizing marker so Genesis's importer and rasterizer agree on image orientation while preserving seam-split UV
vertices.

## MJCF / XML Assets

MJCF/XML generation lives under `code_agent/assets/xml`.

- `assets/xml/agent.py`: standalone Codex worker loop for writing one MJCF file, validating it, rendering static preview
  evidence, and writing reports.
- `assets/xml/validation.py`: thin validation entry point that coordinates XML parsing, structural checks, MuJoCo import,
  and report assembly.
- `assets/xml/validation_core/`: validation internals split into XML collectors, rule checks, and manifest conversion.
- `assets/xml/actuation.py`: lightweight MuJoCo actuator response check that applies bounded controls and verifies a
  measurable joint response.
- `assets/xml/preview.py`: MuJoCo offscreen preview renderer for front, side, isometric, and top views. It automatically
  caps preview resolution to the MJCF offscreen framebuffer instead of requiring generated XML to contain preview-only
  render settings.

The XML worker uses the shared [Utils](utils.md) Codex invocation layer and writes XML from scratch. It is currently a
standalone single-asset loop, while `assets/xml/episode.py` wraps multiple Planner asset requests into a parallel
episode-level XML job.

Planner exposes XML generation through `start_xml_assets`, `wait_xml_assets`, and the compatibility
`generate_xml_assets` action. These actions consume `planner_output.asset_requests` entries whose `asset_type` is
`generated_xml`, `generated_mjcf`, `mjcf`, or a close XML/MJCF alias. Planner can pass `asset_names` to generate a
subset of XML requests, or leave it null/empty to generate all XML/MJCF requests.

XML workers run in parallel by default according to `CONFIGS.xml_asset.max_parallel_workers=None`. Each generated asset
gets its own output directory under `assets/xml/`, its own worker logs and attempt reports, and a validated manifest
entry. The episode wrapper writes `assets/xml_asset_manifest.json` as a partial manifest; the Planner action executor
merges it with mesh outputs into the canonical `assets/asset_manifest.json`.

### XML Scope

Each XML asset must contain exactly one articulated body.

Allowed contents:

- one MJCF `<mujoco>` root
- one articulated body tree under `<worldbody>`
- joints, inertials, geoms, sites, sensors, tendons, equality constraints, and assets needed only by that body
- actuators for that articulated body

Disallowed contents:

- extra scene-level ground planes
- free-standing props
- multiple unrelated articulated bodies
- cameras, lights, task objects, bins, arenas, ramps, projectiles, or fixed obstacles
- global simulation stage design that belongs to the Scene Worker
- `<option>` or other global simulation settings such as gravity, timestep, solver, integrator, or global contact
  options

The Scene Worker places the articulated asset in the generated Genesis scene. The Body Worker instantiates the asset
only when it is movable, free-base, actuated, or task-participating.

### Static Validation

After writing XML, the XML worker runs MuJoCo import validation.

The validation loads the XML with MuJoCo only to confirm syntax and model construction. It does not run a full Genesis
simulation. Standalone runs execute this validation through the repository uv environment and record the result in a
standalone `asset_manifest.json` next to the generated XML.

The validation report should include:

- XML path
- MuJoCo import status
- parser error, if any
- joint names and ranges
- actuator names and target joints/sites
- equality joint coupling when a passive joint is intentionally coupled to an actuated joint
- warnings or assumptions
- static preview image paths and image nonblank statistics
- actuator response result, including the command vector and measured generalized-coordinate response

### Actuator Contract

The XML worker owns actuator design for the articulated body. After generating XML, it must expose a control interface
for the Action Worker.

The Asset Manifest entry for the XML asset must include:

- actuator names
- actuator type or control mode
- controlled joint or tendon
- recommended command range
- neutral command
- suggested open-loop control snippets or schedule hints
- any coupling between actuators
- whether the base is fixed, free, or expected to be fixed by Scene placement

The Action Worker should not reverse-engineer actuator semantics from raw XML when the manifest can provide them.

## Repo Assets

Existing repository assets should be found, validated, and registered without unnecessary regeneration.

## Asset Manifest

Every asset record should include:

- logical name
- source type
- runtime path
- visual path
- texture path
- bbox
- Genesis scale factors
- mesh coordinate convention such as `file_meshes_are_zup`
- physical role
- validation status
- known caveats
- suggested Genesis construction pattern
