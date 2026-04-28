from __future__ import annotations

from copy import deepcopy
import math

from ...ir_schema.program import RigidIR
from ...ir_schema.scene import RenderIR

DEFAULT_RENDER_VIDEO_PATH = "agent/runs/llm_generated/render.mp4"

_DEFAULT_RENDER_CONFIG: dict[str, object] = {
    "output_video": DEFAULT_RENDER_VIDEO_PATH,
    "fps": 60,
    "res": [640, 480],
    "camera_pos": [3.5, 0.5, 2.5],
    "camera_lookat": [0.0, 0.0, 0.5],
    "camera_up": [0.0, 0.0, 1.0],
    "camera_fov": 40.0,
    "near": 0.1,
    "far": 20.0,
    "gui": False,
    "render_every_n_steps": 1,
    "include_initial_frame": True,
    "force_render": False,
}


def default_render_config(*, output_video: str | None = None) -> dict[str, object]:
    config = deepcopy(_DEFAULT_RENDER_CONFIG)
    if output_video is not None and output_video.strip():
        config["output_video"] = output_video.strip()
    return config


def apply_default_render_to_payload(
    payload: dict[str, object],
    *,
    output_video: str | None = None,
) -> dict[str, object]:
    normalized = dict(payload)
    scene_any = normalized.get("scene")
    scene = dict(scene_any) if isinstance(scene_any, dict) else {}

    render_any = scene.get("render")
    if render_any is None:
        scene["render"] = default_render_config(output_video=output_video)
    elif isinstance(render_any, dict):
        merged = default_render_config(output_video=output_video)
        merged.update(render_any)
        output_path = merged.get("output_video")
        if not isinstance(output_path, str) or not output_path.strip():
            merged["output_video"] = default_render_config(output_video=output_video)["output_video"]
        scene["render"] = merged

    normalized["scene"] = scene
    return normalized


def ensure_program_has_render(
    program: RigidIR,
    *,
    output_video: str | None = None,
) -> RigidIR:
    patched = program.model_copy(deep=True)
    if patched.scene.render is None:
        patched.scene.render = RenderIR.model_validate(default_render_config(output_video=output_video))
        return patched

    if not patched.scene.render.output_video.strip():
        patched.scene.render.output_video = str(default_render_config(output_video=output_video)["output_video"])
    return patched


def synchronize_render_timing(program: RigidIR) -> RigidIR:
    patched = program.model_copy(deep=True)
    render = patched.scene.render
    if render is None:
        return patched

    dt = float(patched.scene.sim.dt)
    render_every = max(1, int(render.render_every_n_steps))
    render.render_every_n_steps = render_every

    derived_fps = 1.0 / (dt * render_every)
    if derived_fps > 240.0:
        render.render_every_n_steps = max(render_every, int(math.ceil(1.0 / (dt * 240.0))))
        derived_fps = 1.0 / (dt * float(render.render_every_n_steps))

    render.fps = max(1, min(240, int(round(derived_fps))))
    return patched
