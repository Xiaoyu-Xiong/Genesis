from __future__ import annotations

import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from code_agent.configs import CONFIGS
from code_agent.evaluation.visual import evaluate_visual_artifacts
from code_agent.io_utils import dump_json
from code_agent.io_utils import load_json_object
from code_agent.scores.physical.report import compact_file, file_digest, file_size, unique_paths

SOURCE_SUFFIXES = {".py", ".xml", ".urdf", ".mjcf", ".json", ".yaml", ".yml", ".toml"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
IGNORED_SOURCE_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "logs",
    "frames",
    "physical_score_video_frames",
}


@dataclass(slots=True, frozen=True)
class EvidenceBundle:
    index: dict[str, Any]
    image_paths: list[Path]


def prepare_evidence(*, run_dir: Path, code_root: Path, case_id: str | None) -> EvidenceBundle:
    reports_dir = run_dir / "reports"
    physical_frame_dir = reports_dir / "physical_score_video_frames"
    physical_frame_dir.mkdir(parents=True, exist_ok=True)

    visual_report = ensure_visual_report(run_dir)
    videos = discover_video_paths(run_dir)
    extracted_frames: list[Path] = []
    if videos and not visual_contact_sheet_path(visual_report):
        extracted_frames = extract_video_frames(videos[0], physical_frame_dir)
        if extracted_frames:
            contact_sheet_path = reports_dir / "physical_score_video_contact_sheet.jpg"
            write_contact_sheet(extracted_frames, contact_sheet_path)
            visual_report = {
                "contact_sheet_path": str(contact_sheet_path),
                "sampled_frames": [str(path) for path in extracted_frames],
                "source_video": str(videos[0]),
            }

    source_paths = discover_source_paths(code_root=code_root, run_dir=run_dir)
    data_paths = discover_data_paths(run_dir)
    frame_paths = discover_frame_paths(run_dir, visual_report=visual_report, extracted_frames=extracted_frames)
    asset_images = discover_asset_preview_images(run_dir)
    image_paths = unique_paths(
        [
            *([Path(path)] if (path := visual_contact_sheet_path(visual_report)) else []),
            *asset_images,
            *frame_paths[:4],
        ],
        limit=12,
    )

    index: dict[str, Any] = {
        "schema_version": 1,
        "case_id": case_id,
        "run_dir": str(run_dir),
        "code_root": str(code_root),
        "source_paths": [str(path) for path in source_paths],
        "source_digest": [file_digest(path, max_chars=16000) for path in source_paths[:12]],
        "data_paths": {name: str(path) for name, path in data_paths.items()},
        "data_digest": {name: compact_file(path) for name, path in data_paths.items()},
        "video_paths": [str(path) for path in videos],
        "frame_paths": [str(path) for path in frame_paths[:40]],
        "asset_preview_images": [str(path) for path in asset_images],
        "image_paths_sent_to_scorer": [str(path) for path in image_paths],
        "visual_report": visual_report,
        "file_sizes_bytes": {
            **{f"source:{path.name}": file_size(path) for path in source_paths[:30]},
            **{f"data:{name}": file_size(path) for name, path in data_paths.items()},
            **{f"video:{path.name}": file_size(path) for path in videos[:8]},
        },
        "notes": [
            "SBAR-v1 is implementation-agnostic: source files are discovered from the supplied code root.",
            "The visual contact sheet and asset previews are attached as image inputs when available.",
            "The scorer may inspect source and artifact paths on disk but must not modify files or rerun simulation.",
        ],
        "created_at_unix": time.time(),
    }
    index_path = reports_dir / "physical_score_evidence_index.json"
    index["index_path"] = str(index_path)
    dump_json(index, index_path)
    return EvidenceBundle(index=index, image_paths=image_paths)


def discover_source_paths(*, code_root: Path, run_dir: Path) -> list[Path]:
    preferred = [
        run_dir / "src" / "scene.py",
        run_dir / "src" / "body.py",
        run_dir / "src" / "action.py",
        run_dir / "src" / "rendering.py",
        run_dir / "src" / "main.py",
        code_root / "main.py",
        code_root / "app.py",
    ]
    paths: list[Path] = [path for path in preferred if path.is_file()]
    search_root = code_root if code_root.exists() else run_dir
    for path in sorted(search_root.rglob("*")):
        if len(paths) >= 120:
            break
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if _is_under_ignored_dir(path, root=search_root):
            continue
        if path not in paths:
            paths.append(path)
    return unique_paths(paths, limit=120)


def discover_data_paths(run_dir: Path) -> dict[str, Path]:
    candidates = {
        "execution_report": run_dir / "reports" / "execution_report.json",
        "critic_report": run_dir / "reports" / "critic_report.json",
        "artifact_evaluation": run_dir / "reports" / "artifact_evaluation.json",
        "visual_evaluation": run_dir / "reports" / "visual_evaluation.json",
        "metrics": run_dir / "artifacts" / "metrics.json",
        "event_log": run_dir / "artifacts" / "event_log.json",
        "render_stats": run_dir / "artifacts" / "render_stats.json",
        "artifact_summary": run_dir / "artifacts" / "summary.json",
        "run_result": run_dir / "artifacts" / "run_result.json",
        "action_report": run_dir / "artifacts" / "action_report.json",
        "case_summary": run_dir / "summary.json",
        "planner_output": run_dir / "contracts" / "planner_output.json",
        "timing_contract": run_dir / "contracts" / "timing.json",
        "deformable_config": run_dir / "contracts" / "deformable_config.json",
        "asset_manifest": run_dir / "assets" / "asset_manifest.json",
    }
    return {name: path for name, path in candidates.items() if path.is_file()}


def discover_video_paths(run_dir: Path) -> list[Path]:
    preferred = [
        run_dir / "artifacts" / "render.mp4",
        run_dir / "artifacts" / "video.mp4",
        run_dir / "render.mp4",
        run_dir / "video.mp4",
    ]
    paths: list[Path] = [path for path in preferred if path.is_file()]
    for path in sorted(run_dir.rglob("*")):
        if len(paths) >= 20:
            break
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES and path not in paths:
            paths.append(path)
    return unique_paths(paths, limit=20)


def discover_frame_paths(
    run_dir: Path,
    *,
    visual_report: dict[str, Any] | None,
    extracted_frames: list[Path],
) -> list[Path]:
    paths: list[Path] = []
    if isinstance(visual_report, dict) and isinstance(visual_report.get("sampled_frames"), list):
        paths.extend(Path(item) for item in visual_report["sampled_frames"] if isinstance(item, str))
    paths.extend(extracted_frames)
    frames_dir = run_dir / "artifacts" / "frames"
    if frames_dir.is_dir():
        paths.extend(sorted(path for path in frames_dir.glob("frame_*.png") if path.is_file())[:40])
    return unique_paths(paths, limit=80)


def discover_asset_preview_images(run_dir: Path) -> list[Path]:
    roots = [run_dir / "assets", run_dir / "reports" / "asset_inspection"]
    paths: list[Path] = []
    preferred_names = {"contact_sheet", "iso", "front", "side", "top"}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.stem in preferred_names or "preview" in path.name:
                paths.append(path)
    return unique_paths(paths, limit=20)


def ensure_visual_report(run_dir: Path) -> dict[str, Any] | None:
    report_path = run_dir / "reports" / "visual_evaluation.json"
    existing = load_json_object(report_path)
    if isinstance(existing, dict) and visual_contact_sheet_path(existing):
        return existing
    frames_dir = run_dir / "artifacts" / "frames"
    if not frames_dir.is_dir():
        return existing
    try:
        return evaluate_visual_artifacts(run_dir=run_dir, output_path=report_path)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def extract_video_frames(video_path: Path, output_dir: Path) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return []
    for stale in output_dir.glob("frame_*.jpg"):
        stale.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "fps=1",
        "-frames:v",
        str(CONFIGS.critic.max_frames),
        str(output_dir / "frame_%03d.jpg"),
    ]
    try:
        subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return sorted(output_dir.glob("frame_*.jpg"))


def write_contact_sheet(frame_paths: list[Path], output_path: Path) -> None:
    if not frame_paths:
        return
    thumb_width = max(1, int(CONFIGS.critic.max_width // 2))
    thumb_height = max(1, round(thumb_width * 9 / 16))
    label_height = 25
    thumbs: list[Image.Image] = []
    for path in frame_paths[: CONFIGS.critic.max_frames]:
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue
        image.thumbnail((thumb_width, thumb_height))
        canvas = Image.new("RGB", (thumb_width, thumb_height + label_height), (18, 18, 18))
        canvas.paste(image, ((thumb_width - image.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb_height + 4), path.name, fill=(235, 235, 235))
        thumbs.append(canvas)
    if not thumbs:
        return
    cols = min(3, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumb_width, rows * (thumb_height + label_height)), (12, 12, 12))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % cols) * thumb_width, (index // cols) * (thumb_height + label_height)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def visual_contact_sheet_path(report: dict[str, Any] | None) -> str | None:
    if not isinstance(report, dict):
        return None
    value = report.get("contact_sheet_path")
    if isinstance(value, str) and Path(value).is_file():
        return value
    return None


def _is_under_ignored_dir(path: Path, *, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return any(part in IGNORED_SOURCE_DIR_NAMES for part in rel.parts)
