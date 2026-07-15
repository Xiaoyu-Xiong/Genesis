from __future__ import annotations

from code_agent.prompts.common import RENDER_CLARITY_GUIDE

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="rendering",
    target_file="src/rendering.py",
    required_export="setup_rendering",
    responsibility="Genesis camera setup, render capture hooks, frame/video output, and visual validation signals",
    prompt_body=f"""
    Write these exports:
    - `setup_rendering(
          scene,
          actors,
          *,
          out_dir: Path,
          steps: int,
          fps: int,
          duration_sec: float | None = None,
          target_video_frames: int | None = None,
          render_every_n_steps: int = 1,
          render_res: tuple[int, int] = (640, 480),
          render_profile: str = "debug_raster",
      ) -> dict`
    - `capture_frame(render_state: dict, step: int) -> None`
    - `finalize_rendering(render_state: dict, *, event_log_path: Path | None = None, metrics_path: Path | None = None) -> dict`

    Rendering must use Genesis' camera renderer. Do not implement a 2D event-log diagnostic renderer.
    {RENDER_CLARITY_GUIDE}
    `setup_rendering` runs before `scene.build()` and must:
    - honor the supplied `steps`, `fps`, `duration_sec`, and `target_video_frames`; do not replace them with local
      magic defaults
    - design camera parameters for the task, including camera position, lookat, fov, and fps
    - prefer a fixed camera fitted from the full cached trajectory. If following is necessary, follow the stable
      whole-body center of mass of a fixed primary actor set; never follow a selected vertex, contact point, bbox
      extremum, or an actor identity that changes per frame
    - smooth any following target and camera distance with a dead zone, low-pass filter, and bounded frame-to-frame
      movement. Do not disable this smoothing during render-only replay or drive zoom from the raw per-frame bbox
    - for FEM/deformable tasks, frame the full soft-body interaction so compression, wobble, collapse, spread, bending,
      and contact with supports or the ground are visible throughout the video
    - use the supplied `render_res` as the camera resolution unless there is a strong task-specific reason to override
      it, and record the final resolution in `render_stats.json`
    - call `scene.add_camera(...)` with `GUI=False`
    - in `debug_raster`, keep rendering fast and readable with the native Rasterizer camera
    - in `final_path_traced`, use the scene's GPU RayTracer path, set camera render spp/denoise from
      `scene.genesis_path_tracing` when available, and record a complete path_tracing block in render_stats.json
    - optionally add Genesis lights only when the current renderer supports that API; for RayTracer prefer lights
      configured in scene.py through RayTracer-supported mesh/sphere/emissive light mechanisms
    - treat RayTracer sphere/mesh/emissive lights as visible geometry. Analytically project every light sphere or mesh
      world bbox through every final camera pose (or a conservative swept frustum) and require that no light geometry
      intersects the image; do not rely on white-background pixels to reveal a white light ball
    - clear stale `frame_*.png` files from the output frames directory before saving new frames
    - compute a capture cadence across simulation steps. If `target_video_frames` is provided, capture exactly that many
      frames spread from step 0 through `steps` whenever possible; otherwise capture every `render_every_n_steps`
      simulation steps, including step 0 and the final step when possible.
    - return a render_state dict containing the Genesis camera, output paths, fps, duration, target frame count, capture
      step set, frame list, and `capture_frame` callable

    `capture_frame` is called during simulation and must:
    - treat the incoming `step` as the simulation step index and save a frame only when it is selected by the cadence;
      action.py will call the hook at step 0 and after every simulation step
    - do not rely on `camera.start_recording()` / `camera.stop_recording()` for the final video; camera recording can be
      unreliable for path-traced renders. The saved PNG frames are the authoritative video source.
    - call `camera.render(rgb=True, depth=False, segmentation=False, normal=False, force_render=True)` for raster
      renders, and include the configured `spp` and `denoise` arguments for final path-traced renders when supported
    - save Genesis-rendered RGB frames to:
      `out_dir / "frames" / "frame_000.png"` etc.

    `finalize_rendering` runs after simulation and must:
    - always compose `out_dir / "render.mp4"` directly from the saved `frame_*.png` Genesis camera RGB images; do not
      accept a Genesis camera recording result as the primary video artifact
    - prefer ffmpeg/imageio/`gs.tools.animate` over `camera.stop_recording`; if a camera recording is attempted for a
      secondary diagnostic, it must not replace the frame-composed `render.mp4`
    - verify or record that the encoded video contains the same frame count as the saved PNG sequence whenever practical
    - preserve the supplied fps so `num_frames / fps` matches the requested duration when `duration_sec` was supplied
    - write `out_dir / "render_stats.json"`

    Required artifacts:
    - `out_dir / "frames" / "frame_000.png"` etc.
    - `out_dir / "render.mp4"`
    - `out_dir / "render_stats.json"`
    `render_stats.json` must include `rendered`, `renderer`, `video_path`, `frames_dir`, `num_frames`,
    `fps`, `duration_sec`, `target_video_frames`, `video_duration_sec`, `frame_steps`, `max_rgb_std`,
    `max_frame_delta`, `camera`, `used_genesis_camera`, `video_writer_strategy`, and, when available,
    `video_frame_count_verified` / `video_duration_verified_sec`.
    If using final path tracing or replay rendering, include path-tracing and replay provenance fields:
    `path_tracing.enabled=true`, backend, integrator, spp, denoise, tracing depth, lights, camera overrides,
    background_style, floor_material, material notes, `replay_only`, and `physics_cache_manifest`. Also record
    `light_visibility_checks` with light bounds, tested camera frames, frustum intersections, and minimum clearance;
    `camera_follow` with target/COM source, smoothing, per-frame pose/distance, and maximum adjacent-frame deltas when
    following is enabled; and `palette_roles` for the subject, supports, floor, and background.
    Final path-traced renders are a look-dev stage, not a one-shot technical check: inspect start/mid/end frames and
    tune camera, exposure, soft shadows, light intensity ratios, background/floor, and material readability until the
    result is genuinely polished.
    Preserve prompt-specified colors, but do not leave all important actors, walls, windows, supports, floor, and
    background in the same white or near-white range. After repeated low-contrast rejection, make a substantive
    palette, background-value, roughness, exposure, or key/fill change instead of another tiny exposure tweak.
    The video must be non-empty, must be encoded from the saved Genesis camera RGB frames, and must not be a single-frame
    camera-recording artifact when multiple PNG frames were captured.
    Do not change simulation state, controls, scene setup, or body definitions.
    """,
)
