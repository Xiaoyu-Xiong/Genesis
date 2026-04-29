from __future__ import annotations

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="rendering",
    target_file="src/rendering.py",
    required_export="setup_rendering",
    responsibility="Genesis camera setup, render capture hooks, frame/video output, and visual validation signals",
    prompt_body="""
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
      ) -> dict`
    - `capture_frame(render_state: dict, step: int) -> None`
    - `finalize_rendering(render_state: dict, *, event_log_path: Path | None = None, metrics_path: Path | None = None) -> dict`

    Rendering must use Genesis' native camera renderer. Do not implement a 2D event-log diagnostic renderer.
    `setup_rendering` runs before `scene.build()` and must:
    - honor the supplied `steps`, `fps`, `duration_sec`, and `target_video_frames`; do not replace them with local
      magic defaults
    - design camera parameters for the task, including camera position, lookat, fov, resolution, and fps
    - call `scene.add_camera(...)` with `GUI=False`
    - optionally add Genesis lights only when the current renderer supports `scene.add_light(...)`; never fail if lights
      are unsupported
    - clear stale `frame_*.png` files from the output frames directory before saving new frames
    - compute a capture cadence across simulation steps. If `target_video_frames` is provided, capture exactly that many
      frames spread from step 0 through `steps` whenever possible; otherwise capture at a reasonable cadence.
    - return a render_state dict containing the Genesis camera, output paths, fps, duration, target frame count, capture
      step set, frame list, and `capture_frame` callable

    `capture_frame` is called during simulation and must:
    - treat the incoming `step` as the simulation step index and save a frame only when it is selected by the cadence;
      action.py will call the hook at step 0 and after every simulation step
    - start camera recording lazily on the first saved frame, or otherwise use Genesis camera-rendered RGB frames
    - call `camera.render(rgb=True, depth=False, segmentation=False, normal=False, force_render=True)`
    - save Genesis-rendered RGB frames to:
      `out_dir / "frames" / "frame_000.png"` etc.

    `finalize_rendering` runs after simulation and must:
    - use the Genesis camera recording API when available, e.g. `camera.stop_recording(save_to_filename=..., fps=...)`
      to write `out_dir / "render.mp4"`
    - if Genesis video recording fails, compose the video from Genesis-rendered RGB frames, not from a synthetic 2D plot
    - preserve the supplied fps so `num_frames / fps` matches the requested duration when `duration_sec` was supplied
    - write `out_dir / "render_stats.json"`

    Required artifacts:
    - `out_dir / "frames" / "frame_000.png"` etc.
    - `out_dir / "render.mp4"`
    - `out_dir / "render_stats.json"`
    `render_stats.json` must include `rendered`, `renderer`, `video_path`, `frames_dir`, `num_frames`,
    `fps`, `duration_sec`, `target_video_frames`, `video_duration_sec`, `frame_steps`, `max_rgb_std`,
    `max_frame_delta`, `camera`, and `used_genesis_camera`.
    The video must be non-empty and the frames must come from Genesis camera RGB output.
    Do not change simulation state, controls, scene setup, or body definitions.
    """,
)
