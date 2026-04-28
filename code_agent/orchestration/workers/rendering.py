from __future__ import annotations

from .common import WorkerSpec


SPEC = WorkerSpec(
    role="rendering",
    target_file="src/rendering.py",
    required_export="render_outputs",
    responsibility="render artifact creation, frame/video output, and visual validation signals",
    prompt_body="""
    Write `render_outputs(*, out_dir: Path, event_log_path: Path | None = None, metrics_path: Path | None = None)`.
    For CPU robustness, implement a diagnostic top-down render from `event_log.json` instead of Genesis camera render.
    Use Pillow and imageio if available. Write:
    - `out_dir / "frames" / "frame_000.png"` etc.
    - `out_dir / "render.mp4"`
    - `out_dir / "render_stats.json"`
    `render_stats.json` must include `rendered`, `renderer`, `video_path`, `frames_dir`, `num_frames`,
    `max_rgb_std`, `max_frame_delta`, and `samples`.
    The video must be non-empty and the frames should visibly show actor positions over time.
    Do not change simulation state, controls, scene setup, or body definitions.
    """,
)
