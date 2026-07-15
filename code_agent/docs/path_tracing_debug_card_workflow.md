# Path Tracing Debug-Card Workflow

This note records the rendering workflow requested during the July 2026 path
tracing integration work, plus the practical OptiX/GPU lessons learned while
debugging `rigid_sphere_twisting_edge_slide_contact`.

## User Requirements

- Keep intermediate physics debugging cheap and readable. Use ordinary
  rasterized evidence renders while body/action/scene physics is still failing.
- Enable GPU path tracing only after the physical behavior has passed critic
  review, then tune the final render toward SIGGRAPH-demo quality.
- Final video acceptance must require path tracing. A debug raster video may
  pass physics validation, but it is not a final deliverable when rendering is
  requested.
- Once the agent enters final path-tracing look-dev, it should iterate on
  renderer, camera, exposure, lighting hierarchy, shadows, floor/background,
  and materials until Critic accepts the image quality; a single successful
  RayTracer execution is not enough if the result is flat, over-lit, noisy,
  poorly framed, or materially bland.
- Save enough simulation state during every physics run so final rendering can
  be replayed without re-running physics.
- Treat per-frame `.npz` state cache files as a hard artifact requirement. A
  cache manifest without readable `.npz` files must fail validation before any
  render-only replay attempts to read it.
- Add tests at each implementation stage that changes runtime behavior:
  state-cache enforcement, integrator/execution modes, and representative
  rigid/deformable/cloth generated-code paths.

## Two-Stage Rendering Plan

1. Debug pass:
   - Use the normal Genesis rasterizer or low-cost camera render.
   - Optimize for physical correctness, evidence clarity, and fast iteration.
   - Save a state cache whenever requested by the planner or execution mode.

2. Final pass:
   - Run only after the critic accepts the physics.
   - Prefer render-only replay from a validated state cache.
   - Use GPU `gs.renderers.RayTracer` with `WavePathIntegrator` and OptiX
     denoising.
   - Repeat rendering/body/scene repair and `final_path_traced` execution until
     `render_stats.path_tracing.enabled` is true and the final Critic accepts
     both task readability and visual polish.
   - Record `path_tracing.enabled`, backend, spp, denoise flag, integrator,
     lights, camera, replay manifest, and source run id in `render_stats.json`.

## State Cache Contract

Every cache lives under:

```text
artifacts/state_cache/
  manifest.json
  states/frame_000.npz
  states/frame_001.npz
  ...
```

The manifest must record:

- schema version and cache kind
- frame steps and relative `.npz` paths
- simulation step count, `sim_dt`, `sim_substeps`, backend, and render profile
- actor names, actor replay contracts, required state-array shapes, and source hashes for generated files
- whether the cache is intended for replay

The validator must fail if:

- `manifest.json` is missing or malformed
- any frame entry lacks an `.npz` path
- any referenced `.npz` file is missing
- any referenced `.npz` file cannot be opened by NumPy

Rigid state should include per-frame actor transforms when available. Articulated
rigid bodies must include qpos or DOF positions on every frame; root pose alone
cannot reproduce joint/link motion. The manifest records each actor's replay mode
and required arrays, and strict validation rejects missing, non-finite, or
shape-changing channels before final rendering.
Deformable/FEM/cloth state should include visible surface or state vertices when
available. Static geometry and material metadata should be stored once in the
manifest or as references.

## OptiX/GPU Runtime Lessons

- WSL exposes `/usr/lib/wsl/lib/libnvoptix.so.1` as a small DXCore loader stub.
  It does not export `optixQueryFunctionTable`, so it is not sufficient for
  Genesis/LuisaRender OptiX path tracing.
- A complete NVIDIA OptiX runtime bundle was installed under:

```text
/opt/nvidia-optix-595/lib
/opt/nvidia-optix-595/share/nvidia
```

- Non-interactive launches must explicitly put `/opt/nvidia-optix-595/lib`
  before `/usr/lib/wsl/lib` in `LD_LIBRARY_PATH`.
- The successful GPU path used:

```text
LD_LIBRARY_PATH=/opt/nvidia-optix-595/lib:<LuisaRender build bin>:<repo .venv cuda lib>:/usr/lib/wsl/lib
```

- A systemd or other non-interactive render must preserve both `PATH` for `uv`
  and the CUDA/OptiX `LD_LIBRARY_PATH`; otherwise it may silently load the WSL
  OptiX stub and fail before denoising.

## Path-Traced Look-Dev Lessons

- Use `WavePathIntegrator`, `camera_denoise=True`, and 512 spp or higher for
  final still/video work. Previews may use lower spp.
- Use `tracing_depth` around 16-32; 24 worked well in the validated rigid slide
  case.
- Use large sphere lights for soft shadows, but project-check them so visible
  light meshes do not enter the camera frame.
- Avoid uniform high ambient light. A polished final render should have a clear
  key/fill/rim hierarchy, visible soft contact shadows, and non-flat tonal
  layering.
- Prefer a light studio backdrop such as warm light gray, pale blue-gray, or
  soft off-white, with a non-pure-white floor, controlled warm key light, cool
  fill, and small rim lights.
- Do not inherit the debug raster pure-white fallback as the final look unless
  the prompt explicitly asks for a white stage or product-style background.
- Avoid mirror-like material settings that make small actors disappear into
  reflected bright environments. Increase roughness or use rough plastic/satin
  materials for readability when needed.
- Choose semantically varied materials. Metal props should use metallic or
  satin-metal surfaces, soft/rubber/cloth objects should use rougher readable
  materials, and transparent shells/windows/plates should use true
  path-traced transmission/refraction such as `gs.surfaces.Glass` rather than
  only RGBA alpha.
- Validate start/mid/end frames before accepting a final video; final rendering
  should not crop the trajectory after only checking the first frame.

## Implementation Order

1. Add this long-term memory document.
2. Add/update SimDebug cards for two-stage rendering, final path tracing, state
   cache replay, replay consistency, camera framing, visual evidence, and fresh
   artifact rules.
3. Implement state-cache utilities and hard `.npz` validation.
4. Extend the generated main entrypoint and execution helper with
   `debug_raster`, `final_path_traced`, cache save, replay, and render-only
   modes.
5. Validate generated-code paths for rigid, deformable, and cloth cases.
