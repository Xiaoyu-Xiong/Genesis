from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path


class VideoSamplingError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class SampledFrame:
    index: int
    data_url: str


def _encode_jpeg_data_url(raw: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"


def _import_cv2():
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        raise VideoSamplingError("OpenCV is not available for video sampling.") from exc
    return cv2


def _build_interval_indices(total: int, fps: float, sample_every_sec: float, max_frames: int) -> list[int]:
    if total <= 0 or fps <= 0 or sample_every_sec <= 0 or max_frames <= 0:
        return []
    step_frames = max(1, int(round(sample_every_sec * fps)))
    indices = list(range(0, total, step_frames))
    if not indices:
        indices = [0]
    if len(indices) > max_frames:
        indices = indices[:max_frames]
    return indices


def _resize_frame_if_needed(frame, *, max_width: int):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    cv2 = _import_cv2()
    scale = max_width / float(width)
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (max_width, target_height), interpolation=cv2.INTER_LANCZOS4)


def _sample_frames_cv2(
    video_path: Path,
    *,
    sample_every_sec: float,
    max_frames: int,
    max_width: int,
) -> list[SampledFrame]:
    cv2 = _import_cv2()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoSamplingError(f"OpenCV failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    indices = _build_interval_indices(frame_count, fps, sample_every_sec, max_frames)
    if not indices:
        cap.release()
        raise VideoSamplingError(f"Unable to get valid frame indices from video: {video_path}")

    sampled: list[SampledFrame] = []
    try:
        for out_index, frame_index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = _resize_frame_if_needed(frame, max_width=max_width)
            ok_jpg, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok_jpg:
                continue
            sampled.append(SampledFrame(index=out_index, data_url=_encode_jpeg_data_url(encoded.tobytes())))
    finally:
        cap.release()

    if not sampled:
        raise VideoSamplingError("OpenCV sampling failed to sample any frames.")
    return sampled


def probe_video_duration_sec(video_path: Path) -> float:
    cv2 = _import_cv2()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoSamplingError(f"OpenCV failed to open video: {video_path}")
    try:
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0 or fps <= 0:
            raise VideoSamplingError(f"Invalid OpenCV metadata for `{video_path}`: frames={frame_count}, fps={fps}.")
        return frame_count / fps
    finally:
        cap.release()


def sample_video_frames_as_data_urls(
    video_path: Path,
    *,
    sample_every_sec: float = 0.5,
    max_frames: int = 24,
    max_width: int = 640,
) -> list[SampledFrame]:
    if not video_path.exists():
        raise VideoSamplingError(f"Video file not found: {video_path}")
    if sample_every_sec <= 0:
        raise VideoSamplingError("`sample_every_sec` must be positive.")
    if max_frames <= 0:
        raise VideoSamplingError("`max_frames` must be positive.")
    if max_width <= 0:
        raise VideoSamplingError("`max_width` must be positive.")
    return _sample_frames_cv2(
        video_path,
        sample_every_sec=sample_every_sec,
        max_frames=max_frames,
        max_width=max_width,
    )
