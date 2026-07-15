# Path Tracing Lighting and Material Debug-Card Proposal

This note records proposed SimDebug card changes for final GPU path-traced
rendering. It is intentionally a proposal first: debug cards should be updated
only after the material behavior below is verified on a generated case.

## User Requirements

- Final path-traced renders should not look over-lit or flat. They should show
  clear soft-shadow structure and layered lighting, so the result reads as a
  path-traced studio render rather than a simple OpenGL-style preview.
- Agents should choose richer materials according to object semantics instead
  of defaulting most objects to diffuse surfaces.
- Metal objects should use metallic or satin-metal materials where appropriate,
  with roughness tuned for readable highlights rather than mirror clutter.
- Glass, transparent shells, containers, and other clear objects should not be
  implemented only by lowering RGBA alpha in a diffuse material. In final
  path-traced renders, they should use renderer-native transparent or
  transmissive material settings that produce believable refraction and
  internal light transport.
- Path tracing is mandatory for the final render stage. A physics/debug-raster
  pass may prove the simulation, but it cannot be the final accepted video.
- The final path-tracing stage is an iterative look-dev loop. Agents should
  keep adjusting rendering, lighting, camera, exposure, background, and
  materials until the path-traced image is genuinely polished, not stop after
  the first technically successful RayTracer run.

## Proposed Card Changes

### final_path_tracing_siggraph_guideline

- Add a stronger lighting recipe:
  - Prefer an intentional key/fill/rim hierarchy rather than uniform ambient
    illumination.
  - Use large area or sphere lights to create visible soft contact shadows.
  - Reduce ambient/background intensity when the image looks flat or
    overexposed.
  - Check start/mid/end frames for grounded soft shadows, not only brightness.
- Add a hard workflow rule:
  - Final acceptance requires `render_stats.path_tracing.enabled == true`.
  - Critic should reject final renders that are merely debug raster evidence or
    first-pass RayTracer outputs with flat/over-lit look-dev.
  - Planner should route multiple rendering/body/scene repair rounds after
    entering final path tracing until final visual quality is accepted.
- Add richer material guidance:
  - Pick materials from object semantics: metal for rails/tools/magnets,
    glass/transmission for transparent containers, rough plastic/rubber for
    soft bodies, satin or ceramic finishes for neutral props.
  - Avoid making every object diffuse unless the prompt implies matte objects.
  - Tune roughness to keep highlights informative and avoid mirror-like visual
    confusion.
- Add transparent-material guidance:
  - In final path tracing, transparent objects should use native
    transmissive/refractive material parameters when available.
  - RGBA alpha alone is acceptable only for debug raster visibility or simple
    ghosted overlays, not for final glass-like objects.
  - render_stats.json should record transparent_materials, including material
    model, IOR/transmission/opacity or equivalent renderer fields.

### visual_style_readability_guideline

- Add that visual richness is still subordinate to physical evidence:
  materials should clarify object roles and contacts rather than hide them.
- Encourage semantic materials for unspecified objects when they improve
  readability, while keeping colors and reflections controlled.

### render_visual_evidence_restriction

- Keep debug raster simple and fast. It may still use diffuse approximations or
  alpha materials for inspection.
- Explicitly route high-quality lighting, shadows, and refraction to the final
  path-tracing pass, not every debug iteration.

### planner/two-stage workflow

- Treat debug-raster critic pass as "physics accepted", not "case complete".
- Require a subsequent `final_path_traced` execution and critic pass before
  `finish pass`.
- Prefer render-only replay from the accepted state cache for final look-dev so
  repeated material/lighting iterations do not rerun physics.
- If final path tracing fails or looks mediocre, repair rendering/body/scene and
  rerun final path tracing. Do not collapse this into a single optional pass.

## Validation Plan

Use the generated case:

```text
code_agent/workspaces/suites/dataset_train_batch_20260702_175310_path_tracing_pipeline_retrain_with_api_keys/rigid_magnetic_objects_sphere_external_bar_magnets_pure_rigid
```

Research steps:

1. Inspect the generated source and render_stats.json to see whether the
   transparent sphere shell already uses a path-tracing refractive material or
   only diffuse RGBA alpha.
2. Inspect Genesis surface/material APIs and existing path tracing examples for
   glass, transmission, opacity, IOR, roughness, metalness, and RayTracer
   compatibility.
3. If the generated case uses only alpha transparency, modify this case as a
   controlled experiment:
   - enable GPU RayTracer/WavePathIntegrator for the final render profile;
   - assign the transparent sphere shell a native refractive/glass material;
   - add semantic metallic materials for magnets and small magnetic objects;
   - tune lighting for visible path-traced soft shadows without overexposure;
   - record material and lighting metadata in render_stats.json.
4. Run a short validation render first, then inspect stats/logs and sample
   frames. If successful, render enough frames or a still to prove the material
   path works.
5. Update this proposal with the verified Genesis material recipe, then convert
   the relevant parts into SimDebug card edits, catalog rebuild, and tests.

## Replay Cache Requirement

Render-only look development must consume an existing physics state cache:

```text
artifacts/state_cache/manifest.json
artifacts/state_cache/states/frame_*.npz
```

If these files are absent, the final renderer must fail the render-only replay
path instead of silently stepping physics again or inventing geometry state.
This is especially important for old suite outputs generated before the cache
contract was enforced: they can still be inspected for material choices, but
they are not valid inputs for no-simulation replay unless a real cache is
generated from the accepted physics run.

The material/debug investigation should therefore report two separate results:

- whether the generated source already uses native path-traced refractive
  materials;
- whether the existing artifacts contain a valid cache for render-only replay.

## Investigation Results

The 20260702_175310 generated suite did not already use final path-traced
refraction:

- `rigid_magnetic_objects_sphere_external_bar_magnets_pure_rigid/src/body.py`
  used `gs.surfaces.Smooth(... opacity=...)` for the transparent sphere and
  shell wall elements, plus diffuse/smooth materials for magnetic objects.
- `artifacts/render_stats.json` recorded `path_tracing.enabled=false`.
- The original 175310 artifacts did not contain `artifacts/state_cache/*.npz`,
  because that suite was launched before the execution layer forced
  `--save-state-cache --require-state-cache`.

After the execution-layer cache fix, a short probe on the same case wrote:

```text
artifacts_refractive_cache_probe/state_cache/manifest.json
artifacts_refractive_cache_probe/state_cache/states/frame_000.npz
```

`verify_state_cache_manifest(..., require_npz=True)` passed with
`frame_count=1`, `frame_steps=[0]`, and 284 cached actors. Subsequent material
look-dev runs used `--render-only --replay-cache .../manifest.json`, producing
RayTracer frames without stepping physics.

Important correction from the material-only replay:

- A previous experimental render hid collision tiles and added a matte floor
  during final look-dev. That is not allowed for render-only replay because it
  changes the visible geometry set. The rule is now: render-only replay may
  alter renderer settings, camera, lights, exposure, and existing material
  parameters only.
- A strict material-only replay on the original geometry succeeded technically:
  `renderer=RayTracer`, `path_tracing.enabled=true`, `replay_only=true`,
  `frames_replayed=1`, `transparent_materials[].surface=gs.surfaces.Glass`,
  and `alpha_rgba_used=false`.
- The same strict render also showed the current geometry limitation: the
  visual sphere behaves like a solid Glass primitive and the dense wall tiles
  remain visible because they are part of the cached scene geometry. This is the
  correct failure mode for render-only replay; it should trigger scene/body
  repair before physics acceptance, not a render-only geometry workaround.

Verified Genesis recipe:

- Enable final render with GPU `gs.renderers.RayTracer`, tracing depth 24, and
  camera `spp=256` for preview validation plus denoise enabled. Higher spp
  should be used for final videos when runtime allows.
- Use `gs.surfaces.Glass(color=(1.0, 1.0, 1.0), roughness=0.006, ior=1.33)`
  for the semantic transparent sphere shell. The stats must record
  `transparent_materials[].alpha_rgba_used=false`.
- Use `gs.surfaces.Metal(..., metal_type=..., roughness=...)` for magnetic or
  metallic objects instead of leaving everything diffuse.
- Render-only replay must not add, remove, hide, or replace geometry. It may
  only adjust rendering attributes such as camera, lights, exposure, renderer
  profile, and existing surface/material parameters.
- If dense collision-proxy wall tiles, helper cages, or a primitive solid
  sphere would make the final glass look wrong, that is a scene/body geometry
  modeling issue, not a render-only repair. The agent must author the semantic
  visual geometry before the accepted physics/cache run, then regenerate the
  cache from that source.
- A single Glass primitive sphere renders as a solid glass volume in RayTracer.
  Hollow glass containers need actual thin-shell geometry with inner and outer
  surfaces, or another renderer-supported hollow-shell representation, created
  before cache acceptance.
- RayTracer sphere lights are visible geometry. Large area/sphere lights must
  be placed outside the camera frustum or the final image will show giant light
  balls. A useful pattern is a large warm key behind/above the camera, weak cool
  fill, and a subtle rim light, with a light neutral environment. A matte floor
  is useful only when such support geometry already exists or is part of the
  pre-cache scene; render-only replay should not invent one.
- `gs.surfaces.Emission(emissive_texture=gs.textures.ColorTexture(...))` is a
  robust way to configure a constant RayTracer environment; avoid relying on
  shortcut fields if they collide with Pydantic texture resolution.

## Open Questions

- For full videos, how many final look-dev repair rounds should the planner
  budget before escalating to a human?
- For generated transparent shells, should the body worker prefer clean visual
  mesh shells over primitive solid spheres when a hollow object is semantically
  intended?
