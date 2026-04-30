from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from code_agent.configs import CONFIGS


def evaluate_visual_artifacts(*, run_dir: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Create lightweight visual evidence for the critic without making pass/fail decisions."""

    run_dir = run_dir.resolve()
    reports_dir = run_dir / "reports"
    output_path = output_path or (reports_dir / "visual_evaluation.json")
    contact_sheet_path = reports_dir / "visual_contact_sheet.jpg"
    render_stats = _read_json(run_dir / "artifacts" / "render_stats.json")
    frames_dir = run_dir / "artifacts" / "frames"
    frame_paths = _sample_frame_paths(frames_dir, render_stats=render_stats)
    frame_summaries = [_summarize_image(path) for path in frame_paths]
    if frame_paths:
        _write_contact_sheet(frame_paths, contact_sheet_path, max_width=CONFIGS.critic.max_width)

    texture_summaries = _texture_summaries(run_dir / "assets" / "asset_manifest.json")
    texture_presence = [
        _texture_presence(texture, frame_paths)
        for texture in texture_summaries
        if texture.get("texture_path") and texture.get("mean_rgb")
    ]
    warnings = [
        item["warning"]
        for item in texture_presence
        if item.get("warning")
    ]
    report = {
        "evaluator": "visual_evidence",
        "schema_version": 1,
        "run_dir": str(run_dir),
        "sampled_frames": [str(path) for path in frame_paths],
        "sampling": {
            "sample_every_sec": CONFIGS.critic.sample_every_sec,
            "max_frames": CONFIGS.critic.max_frames,
            "source_fps": _infer_fps(render_stats, _reported_frame_count(render_stats, frames_dir)),
        },
        "contact_sheet_path": str(contact_sheet_path) if frame_paths else None,
        "frame_summaries": frame_summaries,
        "texture_summaries": texture_summaries,
        "texture_presence": texture_presence,
        "warnings": warnings,
        "created_at_unix": time.time(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _sample_frame_paths(
    frames_dir: Path,
    *,
    render_stats: dict[str, Any] | None = None,
    sample_every_sec: float | None = None,
    max_frames: int | None = None,
) -> list[Path]:
    if not frames_dir.is_dir():
        return []
    frames = sorted(path for path in frames_dir.glob("frame_*.png") if path.is_file())
    if not frames:
        return []

    sample_every_sec = CONFIGS.critic.sample_every_sec if sample_every_sec is None else float(sample_every_sec)
    max_frames = CONFIGS.critic.max_frames if max_frames is None else int(max_frames)
    max_frames = max(1, max_frames)

    fps = _infer_fps(render_stats, len(frames))
    if fps is None or sample_every_sec <= 0:
        return _uniform_sample_frames(frames, max_frames)

    interval = max(1, int(round(sample_every_sec * fps)))
    sampled = frames[::interval]
    if sampled[-1] != frames[-1]:
        sampled.append(frames[-1])
    if len(sampled) <= max_frames:
        return sampled
    return _uniform_sample_frames(frames, max_frames)


def _uniform_sample_frames(frames: list[Path], count: int) -> list[Path]:
    count = max(1, int(count))
    if len(frames) <= count:
        return frames
    if count == 1:
        return [frames[0]]
    indices = sorted({round(i * (len(frames) - 1) / (count - 1)) for i in range(count)})
    return [frames[index] for index in indices]


def _infer_fps(render_stats: dict[str, Any] | None, frame_count: int) -> float | None:
    if not isinstance(render_stats, dict):
        return None
    fps = _positive_float(render_stats.get("fps"))
    if fps is not None:
        return fps
    duration = _positive_float(render_stats.get("video_duration_sec"))
    if duration is None:
        duration = _positive_float(render_stats.get("duration_sec"))
    if duration is None or frame_count <= 1:
        return None
    return frame_count / duration


def _reported_frame_count(render_stats: dict[str, Any] | None, frames_dir: Path) -> int:
    if isinstance(render_stats, dict):
        try:
            count = int(render_stats.get("num_frames"))
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            return count
    if not frames_dir.is_dir():
        return 0
    return sum(1 for path in frames_dir.glob("frame_*.png") if path.is_file())


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _summarize_image(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return {
        "path": str(path),
        "size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "mean_rgb": _round_vec(rgb.mean(axis=(0, 1))),
        "std_rgb": _round_vec(rgb.std(axis=(0, 1))),
        "colorfulness": round(float(np.mean(np.max(rgb, axis=2) - np.min(rgb, axis=2))), 6),
    }


def _write_contact_sheet(frame_paths: list[Path], output_path: Path, *, max_width: int) -> None:
    thumb_width = max(1, int(max_width // 2))
    thumb_height = max(1, int(round(thumb_width * 9 / 16)))
    label_height = 25
    thumbs: list[Image.Image] = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width, thumb_height))
        canvas = Image.new("RGB", (thumb_width, thumb_height + label_height), (20, 20, 20))
        canvas.paste(image, ((thumb_width - image.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb_height + 4), path.name, fill=(230, 230, 230))
        thumbs.append(canvas)
    cols = min(3, max(1, len(thumbs)))
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumb_width, rows * (thumb_height + label_height)), (12, 12, 12))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * thumb_width, (idx // cols) * (thumb_height + label_height)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _texture_summaries(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        return []
    summaries: list[dict[str, Any]] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        texture_path = asset.get("texture_path")
        item: dict[str, Any] = {
            "logical_name": asset.get("logical_name"),
            "texture_path": texture_path,
            "file_meshes_are_zup": asset.get("file_meshes_are_zup"),
            "scale": asset.get("scale"),
        }
        if isinstance(texture_path, str) and Path(texture_path).is_file():
            item.update(_summarize_texture(Path(texture_path)))
        summaries.append(item)
    return summaries


def _summarize_texture(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as exc:  # noqa: BLE001
        return {"texture_error": f"{type(exc).__name__}: {exc}"}
    return {
        "texture_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "mean_rgb": _round_vec(rgb.mean(axis=(0, 1))),
        "mean_saturation": round(float(_saturation(rgb).mean()), 6),
    }


def _texture_presence(texture: dict[str, Any], frame_paths: list[Path]) -> dict[str, Any]:
    mean_rgb = texture.get("mean_rgb")
    if not isinstance(mean_rgb, list) or len(mean_rgb) != 3:
        return {"logical_name": texture.get("logical_name"), "max_color_presence": 0.0}
    target = np.asarray(mean_rgb, dtype=np.float32)
    target_sat = float(texture.get("mean_saturation") or 0.0)
    fractions: list[float] = []
    for path in frame_paths:
        try:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        except Exception:  # noqa: BLE001
            continue
        diff = np.linalg.norm(rgb - target.reshape((1, 1, 3)), axis=2)
        fractions.append(float(np.mean(diff < 0.18)))
    max_presence = max(fractions or [0.0])
    warning = None
    if target_sat > 0.2 and max_presence < 0.002:
        warning = "saturated_texture_color_underrepresented_in_sampled_frames"
    return {
        "logical_name": texture.get("logical_name"),
        "texture_mean_rgb": mean_rgb,
        "texture_mean_saturation": target_sat,
        "max_color_presence": round(max_presence, 6),
        "warning": warning,
    }


def _saturation(rgb: np.ndarray) -> np.ndarray:
    max_c = np.max(rgb, axis=2)
    min_c = np.min(rgb, axis=2)
    return np.where(max_c <= 1e-6, 0.0, (max_c - min_c) / max_c)


def _round_vec(vec: np.ndarray) -> list[float]:
    return [round(float(item), 6) for item in vec.tolist()]
