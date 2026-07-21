# Path Tracing Workflow

The final rendering pipeline separates fast physics debugging from GPU path-traced look development.

## Two Stages

1. `debug_raster`: iterate on scene, body, action, contact, metrics, and camera readability with inexpensive raster
   evidence. Every accepted physics run writes a state cache.
2. `final_path_traced`: after Critic accepts physics, replay that cache without stepping simulation and iterate on
   renderer, camera, exposure, lighting, background, and existing material properties until Critic accepts the result.

A raster video can prove physics but is never the final deliverable when rendering is requested. Final acceptance
requires `render_stats.path_tracing.enabled=true`, a valid final video, and a successful visual-quality review. One
technically successful RayTracer call is insufficient when the image remains flat, noisy, poorly framed, or materially
unclear.

## State Cache And Replay

Each physics run writes:

```text
artifacts/state_cache/
  manifest.json
  source_snapshot/src/*.py
  states/frame_000.npz
  states/frame_001.npz
```

The manifest records timing, backend, render profile, frame steps, relative NPZ paths, source hashes, actor contracts,
and required state-array shapes. Validation fails before replay when the manifest is malformed, an NPZ path is absent,
a file is missing or unreadable, or an actor channel is missing, non-finite, or changes shape.

Rigid actors cache transforms. Articulated bodies additionally cache qpos or DOF state on every frame; root pose alone
cannot reconstruct link motion. Deformable and cloth actors cache visible/state vertices. Static geometry and material
metadata are stored once.

A source hash mismatch is classified from the actual source diff as `render_only`, `physics_affecting`, or
`indeterminate`. Only `render_only` may reuse accepted NPZ data. Render-only replay may change renderer settings,
camera, lights, exposure, background parameters, and existing material properties, but it may not add, remove, hide,
replace, or reshape geometry.

## GPU Runtime

Final rendering uses GPU `gs.renderers.RayTracer`, `WavePathIntegrator`, and camera denoising. Preview renders may use
64-256 spp; final work normally starts at 512 spp. A tracing depth around 16-32 is appropriate, with 24 validated in the
rigid slide case.

On WSL, `/usr/lib/wsl/lib/libnvoptix.so.1` is only a loader stub. Non-interactive jobs must put the complete OptiX
runtime first in `LD_LIBRARY_PATH`, currently:

```text
/opt/nvidia-optix-595/lib:<LuisaRender bin>:<repo CUDA lib>:/usr/lib/wsl/lib
```

Systemd launches must also preserve the repository `uv` path and CUDA environment. The execution preflight should fail
clearly on OptiX/CUDA load errors instead of silently falling back to CPU or raster rendering.

## Lighting

- Prefer a deliberate key/fill/rim hierarchy over strong uniform ambient illumination.
- Use large sphere lights for visible soft contact shadows, but projection-check every light because RayTracer sphere
  lights are renderable geometry and must remain outside the camera frame.
- Reduce environment and fill intensity when the scene is flat or overexposed.
- Prefer light studio backgrounds such as warm gray, pale blue-gray, or soft off-white with a non-pure-white floor.
- Check start, middle, and end frames for framing, grounded shadows, exposure, and moving-subject readability.

## Materials

Choose surfaces from object semantics rather than making everything diffuse:

- metal and magnets: `gs.surfaces.Metal` or readable satin metal with controlled roughness;
- rubber, cloth, and soft bodies: rougher surfaces that preserve deformation cues;
- transparent shells, panes, and containers: native path-traced transmission/refraction, such as
  `gs.surfaces.Glass(color=(1.0, 1.0, 1.0), roughness=0.006, ior=1.33)`.

RGBA alpha is acceptable for raster inspection or ghost overlays, not final glass. `render_stats.json` records the
material model and confirms `alpha_rgba_used=false` for refractive objects.

Geometry semantics remain a pre-cache modeling responsibility. A Glass sphere is a solid refractive volume; a hollow
container needs actual inner and outer shell surfaces before physics/cache acceptance. A prompt asking for a thin
screen or pane requires thin-layer geometry, not an entire solid glass cylinder. Render-only look development must not
repair these mistakes by hiding collision geometry or inventing replacement shells.

## Final Evidence

`render_stats.json` records at least:

- path-tracing enabled flag, backend, integrator, spp, tracing depth, and denoise;
- camera, environment, lights, and light-visibility checks;
- material models, roughness, metal type, glass IOR, and alpha usage;
- `replay_only=true`, cache manifest, source run id, and replayed frame count;
- final frame sequence and video metadata.

The video is assembled from saved `frame_*.png` images rather than camera recorder output. Critic checks that the video
frame count and duration agree with both `render_stats.json` and the frame sequence.
