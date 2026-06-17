from __future__ import annotations

import json
import importlib.util
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from code_agent.dataset.models import SegmentCandidate, SourceCandidate
from code_agent.dataset.utils import is_probably_video_url, is_ytdlp_supported_url, sha256_file, slugify


@dataclass(slots=True, frozen=True)
class VideoInfo:
    duration_sec: float
    width: int | None = None
    height: int | None = None


@dataclass(slots=True, frozen=True)
class DownloadedVideo:
    path: Path
    sha256: str
    bytes: int
    info: VideoInfo


def download_video(candidate: SourceCandidate, *, out_dir: Path, source_id: str, timeout_sec: float = 120.0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _download_suffix(candidate.video_url)
    out_path = out_dir / f"{source_id}{suffix}"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    local_path = Path(candidate.video_url).expanduser()
    if local_path.exists():
        shutil.copy2(local_path, out_path)
        return out_path

    if is_ytdlp_supported_url(candidate.video_url):
        return _download_with_ytdlp(candidate.video_url, out_path, timeout_sec=timeout_sec)
    if is_probably_video_url(candidate.video_url):
        return _download_direct(candidate.video_url, out_path, timeout_sec=timeout_sec)
    raise RuntimeError(
        "Source URL is not a direct video URL and is not recognized as a supported yt-dlp site. "
        "Use Codex scout to resolve project pages to demo video URLs first."
    )


def describe_download(path: Path) -> DownloadedVideo:
    return DownloadedVideo(
        path=path,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        info=probe_video(path),
    )


def probe_video(path: Path) -> VideoInfo:
    ffprobe = resolve_ffprobe()
    if ffprobe:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if completed.returncode == 0:
            try:
                payload = json.loads(completed.stdout)
                duration = float(payload.get("format", {}).get("duration") or 0.0)
                stream = (payload.get("streams") or [{}])[0]
                return VideoInfo(
                    duration_sec=max(0.0, duration),
                    width=_optional_int(stream.get("width")),
                    height=_optional_int(stream.get("height")),
                )
            except (ValueError, TypeError, json.JSONDecodeError, IndexError):
                pass
    return _probe_video_with_cv2(path)


def cut_clip(source_path: Path, *, start_sec: float, end_sec: float, out_path: Path) -> None:
    if end_sec <= start_sec:
        raise ValueError(f"Invalid clip range: start={start_sec}, end={end_sec}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            resolve_ffmpeg(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(source_path),
            "-t",
            f"{end_sec - start_sec:.3f}",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ],
        check=True,
        timeout=max(60, int((end_sec - start_sec) * 10)),
    )


def build_contact_sheet(video_path: Path, out_path: Path, *, max_frames: int = 12, thumb_width: int = 180) -> None:
    frames = sample_frames(video_path, max_frames=max_frames, thumb_width=thumb_width)
    if not frames:
        raise RuntimeError(f"No frames could be sampled from {video_path}")
    cols = min(4, len(frames))
    rows = (len(frames) + cols - 1) // cols
    label_height = 20
    thumb_height = max(frame.height for _, frame in frames)
    sheet = Image.new("RGB", (cols * thumb_width, rows * (thumb_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (time_sec, frame) in enumerate(frames):
        col = index % cols
        row = index // cols
        x = col * thumb_width
        y = row * (thumb_height + label_height)
        sheet.paste(frame, (x, y + label_height))
        draw.text((x + 4, y + 3), f"{time_sec:.1f}s", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def average_video_hash(video_path: Path, *, max_frames: int = 8) -> str | None:
    frames = sample_frames(video_path, max_frames=max_frames, thumb_width=64)
    if not frames:
        return None
    return _aggregate_hashes([_image_average_hash(frame) for _, frame in frames])


def visual_signature(video_path: Path, *, max_frames: int = 10) -> dict[str, object]:
    """Return a compact approximate visual signature for duplicate detection.

    The frame hashes make speed changes less important because matching uses unordered nearest-frame distances.
    Foreground component hashes make white-background paper demos more robust to crop, speed, and comparison-layout
    changes. Color histograms keep visually unrelated clips with coincident hashes from collapsing together.
    """

    frames = sample_frames(video_path, max_frames=max_frames, thumb_width=240)
    images = [frame for _, frame in frames]
    frame_hashes = [_image_average_hash(frame) for frame in images]
    foreground_component_hashes: list[str] = []
    foreground_component_phashes: list[str] = []
    for image in images:
        for crop in _foreground_component_crops(image):
            foreground_component_hashes.append(_image_average_hash(crop))
            foreground_component_phashes.append(_image_perceptual_hash(crop))
    return {
        "visual_fingerprint": _aggregate_hashes(frame_hashes),
        "frame_fingerprints": frame_hashes,
        "color_histogram": _mean_color_histogram(images),
        "foreground_component_fingerprints": foreground_component_hashes,
        "foreground_component_phashes": foreground_component_phashes,
        "foreground_color_histogram": _mean_foreground_color_histogram(images),
        "signature_version": 2,
    }


def detect_scene_segments(
    video_path: Path,
    *,
    source_id: str = "source",
    threshold: float = 0.25,
    min_segment_sec: float = 1.0,
    sample_fps: float = 2.0,
    max_segments: int = 12,
) -> list[SegmentCandidate]:
    info = probe_video(video_path)
    duration = info.duration_sec
    if duration <= 0:
        return []
    samples = _sample_cv2_arrays(video_path, sample_fps=sample_fps, max_samples=360)
    boundaries = [0.0]
    last_boundary = 0.0
    previous = None
    for time_sec, frame in samples:
        if previous is None:
            previous = frame
            continue
        diff = float(abs(frame.astype("float32") - previous.astype("float32")).mean() / 255.0)
        if diff >= threshold and time_sec - last_boundary >= min_segment_sec:
            boundaries.append(time_sec)
            last_boundary = time_sec
            if len(boundaries) >= max_segments:
                break
        previous = frame
    if duration - boundaries[-1] >= min_segment_sec:
        boundaries.append(duration)
    elif len(boundaries) > 1:
        boundaries[-1] = duration
    else:
        boundaries = [0.0, duration]

    segments = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False), start=1):
        if end - start < min_segment_sec * 0.5:
            continue
        segments.append(
            SegmentCandidate(
                title_slug=f"{slugify(source_id)}_segment_{index:02d}",
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                visual_summary="deterministic scene-change segment candidate",
                reason="Detected by frame-difference fallback.",
                confidence=None,
            )
        )
    return segments


def sample_frames(video_path: Path, *, max_frames: int, thumb_width: int) -> list[tuple[float, Image.Image]]:
    import cv2

    info = probe_video(video_path)
    duration = info.duration_sec
    if duration <= 0:
        return []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    frame_count = max(1, max_frames)
    times = _sample_times(duration, frame_count)
    frames: list[tuple[float, Image.Image]] = []
    try:
        for time_sec in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            if image.width != thumb_width:
                height = max(1, int(image.height * (thumb_width / image.width)))
                image = image.resize((thumb_width, height), Image.Resampling.LANCZOS)
            frames.append((time_sec, image))
    finally:
        capture.release()
    return frames


def resolve_ffmpeg() -> str:
    resolved = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    if Path(resolved).exists():
        return resolved
    raise RuntimeError("ffmpeg not found. Install ffmpeg or add it to PATH before building the dataset.")


def resolve_ffprobe() -> str | None:
    resolved = shutil.which("ffprobe") or "/usr/bin/ffprobe"
    return resolved if Path(resolved).exists() else None


def resolve_ytdlp_command() -> list[str]:
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    ytdlp = shutil.which("yt-dlp")
    if ytdlp:
        return [ytdlp]
    raise RuntimeError(
        "yt-dlp is required for YouTube/Vimeo/Bilibili-style URLs. Install it with `uv pip install yt-dlp`."
    )


def _download_direct(url: str, out_path: Path, *, timeout_sec: float) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": "GenesisDatasetBuilder/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response, out_path.open("wb") as file:
        shutil.copyfileobj(response, file)
    return out_path


def _download_with_ytdlp(url: str, out_path: Path, *, timeout_sec: float) -> Path:
    subprocess.run(
        [
            *resolve_ytdlp_command(),
            "--no-playlist",
            "--format",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format",
            "mp4",
            "--output",
            str(out_path),
            url,
        ],
        check=True,
        timeout=timeout_sec,
    )
    return out_path


def _download_suffix(url: str) -> str:
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if suffix in {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}:
        return suffix
    return ".mp4"


def _probe_video_with_cv2(path: Path) -> VideoInfo:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return VideoInfo(duration_sec=0.0)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = _optional_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = _optional_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frames / fps if fps > 0 else 0.0
        return VideoInfo(duration_sec=max(0.0, duration), width=width, height=height)
    finally:
        capture.release()


def _sample_cv2_arrays(video_path: Path, *, sample_fps: float, max_samples: int) -> list[tuple[float, object]]:
    import cv2

    info = probe_video(video_path)
    duration = info.duration_sec
    if duration <= 0:
        return []
    sample_count = min(max_samples, max(1, int(duration * sample_fps)))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    frames = []
    try:
        for time_sec in _sample_times(duration, sample_count):
            capture.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            frames.append((time_sec, frame))
    finally:
        capture.release()
    return frames


def _sample_times(duration: float, count: int) -> list[float]:
    if count <= 1:
        return [max(0.0, duration * 0.5)]
    margin = min(0.1, duration * 0.05)
    start = margin
    end = max(start, duration - margin)
    return [start + (end - start) * index / (count - 1) for index in range(count)]


def _image_average_hash(image: Image.Image) -> str:
    gray = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    threshold = sum(pixels) / len(pixels)
    bits = 0
    for value in pixels:
        bits = (bits << 1) | int(value >= threshold)
    return f"{bits:016x}"


def _image_perceptual_hash(image: Image.Image) -> str:
    import cv2
    import numpy as np

    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    dct = cv2.dct(gray)
    low_frequency = dct[:8, :8].flatten()[1:]
    if low_frequency.size == 0:
        return _image_average_hash(image)
    threshold = float(np.median(low_frequency))
    bits = 0
    for value in low_frequency:
        bits = (bits << 1) | int(float(value) >= threshold)
    return f"{bits:016x}"


def _aggregate_hashes(hashes: list[str]) -> str | None:
    if not hashes:
        return None
    bit_counts = [0] * 64
    for hash_text in hashes:
        try:
            value = int(hash_text, 16)
        except ValueError:
            continue
        for bit_index in range(64):
            if value & (1 << bit_index):
                bit_counts[bit_index] += 1
    if not any(bit_counts):
        return hashes[0] if hashes else None
    threshold = max(1, len(hashes) / 2)
    aggregate = 0
    for bit_index, count in enumerate(bit_counts):
        if count >= threshold:
            aggregate |= 1 << bit_index
    return f"{aggregate:016x}"


def _mean_color_histogram(images: list[Image.Image], *, bins: int = 8) -> list[float]:
    if not images:
        return []
    histogram = [0.0] * (bins * 3)
    for image in images:
        resized = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        pixels = list(resized.getdata())
        for red, green, blue in pixels:
            histogram[(red * bins) // 256] += 1.0
            histogram[bins + (green * bins) // 256] += 1.0
            histogram[2 * bins + (blue * bins) // 256] += 1.0
    total = sum(histogram)
    if total <= 0:
        return []
    return [value / total for value in histogram]


def _mean_foreground_color_histogram(images: list[Image.Image], *, bins: int = 8) -> list[float]:
    import numpy as np

    if not images:
        return []
    histogram = np.zeros(bins * 3, dtype=np.float64)
    for image in images:
        array, mask = _foreground_array_and_mask(image)
        pixels = array[mask > 0]
        if pixels.size == 0:
            continue
        for channel in range(3):
            channel_hist, _ = np.histogram(pixels[:, channel], bins=bins, range=(0, 256))
            histogram[channel * bins : (channel + 1) * bins] += channel_hist
    total = float(histogram.sum())
    if total <= 0:
        return []
    return [float(value / total) for value in histogram]


def _foreground_component_crops(image: Image.Image, *, max_components: int = 4) -> list[Image.Image]:
    import cv2

    array, mask = _foreground_array_and_mask(image)
    height, width = mask.shape
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(32, int(width * height * 0.006))
    min_width = max(8, int(width * 0.08))
    min_height = max(8, int(height * 0.08))
    components = []
    for label in range(1, labels_count):
        x, y, component_width, component_height, area = [int(value) for value in stats[label]]
        if area < min_area or component_width < min_width or component_height < min_height:
            continue
        components.append((x, y, component_width, component_height, area))
    components.sort(key=lambda item: item[4], reverse=True)
    crops = []
    for x, y, component_width, component_height, _ in components[:max_components]:
        pad = 4
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(width, x + component_width + pad)
        bottom = min(height, y + component_height + pad)
        crop_array = array[top:bottom, left:right]
        if crop_array.size:
            crops.append(Image.fromarray(crop_array, mode="RGB"))
    return crops


def _foreground_array_and_mask(image: Image.Image) -> tuple[object, object]:
    import cv2
    import numpy as np

    array = np.asarray(image.convert("RGB"))
    max_channel = array.max(axis=2)
    min_channel = array.min(axis=2)
    saturation = max_channel - min_channel
    mask = ((max_channel < 238) | (saturation > 18)).astype("uint8") * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return array, mask


def _optional_int(value: object) -> int | None:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
