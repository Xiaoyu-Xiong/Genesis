from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..configs import CONFIGS
from ..io_utils import dump_json
from ..usage import aggregate_usage_metrics, usage_to_metrics
from .digest import (
    build_compact_input_digest,
    build_input_digest,
    ensure_sectioned_analysis,
    extract_first_json_object,
)
from .prompting import (
    CRITIC_SYSTEM_PROMPT,
    build_compact_critic_prompt_cache_key,
    build_compact_critic_user_content,
    build_critic_prompt_cache_key,
    build_critic_user_content,
    build_stage1_critic_prompt_cache_key,
    build_stage1_critic_user_content,
)
from .video_sampler import SampledFrame, VideoSamplingError, probe_video_duration_sec, sample_video_frames_as_data_urls


def message_usage(message: dict[str, object]) -> dict[str, int]:
    return usage_to_metrics(message.get("_usage") if isinstance(message.get("_usage"), dict) else None)


def build_critic_log_payload(
    *,
    model: str,
    input_digest: dict[str, object],
    frames_used: int,
    raw_response_text: str,
    analysis_json: dict[str, object],
    stage_logs: list[dict[str, object]],
) -> dict[str, Any]:
    return {
        "mode": "critic_evaluation",
        "model": model,
        "input_digest": input_digest,
        "frames_used": frames_used,
        "raw_response_text": raw_response_text,
        "analysis_json": analysis_json,
        "usage_summary": aggregate_usage_metrics([log.get("usage") for log in stage_logs]),
        "stage_logs": stage_logs,
    }


def emit_critic_progress(*, log_path: Path | None, payload: dict[str, Any]) -> None:
    if log_path is not None:
        dump_json(payload, log_path)


def emit_retrieval_progress(
    progress_callback: Callable[[list[dict[str, Any]], str | None, dict[str, Any] | None], None] | None,
    stage_logs: list[dict[str, Any]],
    raw_response_text: str | None,
    analysis_json: dict[str, Any] | None,
) -> None:
    if progress_callback is not None:
        progress_callback(stage_logs, raw_response_text, analysis_json)


def build_single_stage_inputs(
    *,
    task: str,
    ir: dict[str, Any],
    event_pack: dict[str, Any],
    xml_infos_by_body: dict[str, dict[str, Any]],
    xml_texts_by_body: dict[str, str],
    video_duration_sec: float | None,
    sample_every_sec: float,
    max_frames: int,
    max_width: int,
    video_path: Path,
    prompt_variant: str,
) -> tuple[dict[str, Any], list[SampledFrame], list[dict[str, Any]], str, str]:
    sampled_frames = sample_video_frames(
        video_path=video_path,
        sample_every_sec=sample_every_sec,
        max_frames=max_frames,
        max_width=max_width,
    )
    if prompt_variant == "compact":
        input_digest = build_compact_input_digest(
            task=task,
            ir=ir,
            event_pack=event_pack,
            xml_infos_by_body=xml_infos_by_body,
            video_duration_sec=video_duration_sec,
            sample_every_sec=sample_every_sec,
            max_frames=max_frames,
        )
        content = build_compact_critic_user_content(
            task=task,
            ir=ir,
            event_pack=event_pack,
            xml_texts_by_body=xml_texts_by_body,
            input_digest=input_digest,
            sampled_frames=sampled_frames,
        )
        return input_digest, sampled_frames, content, build_compact_critic_prompt_cache_key(), CRITIC_SYSTEM_PROMPT

    input_digest = build_input_digest(
        task=task,
        ir=ir,
        event_pack=event_pack,
        xml_infos_by_body=xml_infos_by_body,
        video_duration_sec=video_duration_sec,
        sample_every_sec=sample_every_sec,
        max_frames=max_frames,
    )
    content = build_critic_user_content(
        task=task,
        ir=ir,
        event_pack=event_pack,
        xml_texts_by_body=xml_texts_by_body,
        input_digest=input_digest,
        sampled_frames=sampled_frames,
    )
    return input_digest, sampled_frames, content, build_critic_prompt_cache_key(), CRITIC_SYSTEM_PROMPT


def build_stage1_inputs(
    *,
    task: str,
    ir: dict[str, Any],
    event_pack: dict[str, Any],
    xml_infos_by_body: dict[str, dict[str, Any]],
    xml_texts_by_body: dict[str, str],
    video_duration_sec: float | None,
    sample_every_sec: float,
    sampled_frames: list[SampledFrame],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if CONFIGS.optimization.critic_stage1_prompt_variant == "full":
        stage1_digest = build_input_digest(
            task=task,
            ir=ir,
            event_pack=event_pack,
            xml_infos_by_body=xml_infos_by_body,
            video_duration_sec=video_duration_sec,
            sample_every_sec=sample_every_sec,
            max_frames=CONFIGS.optimization.critic_stage1_max_frames,
        )
        return stage1_digest, build_critic_user_content(
            task=task,
            ir=ir,
            event_pack=event_pack,
            xml_texts_by_body=xml_texts_by_body,
            input_digest=stage1_digest,
            sampled_frames=sampled_frames,
        )

    stage1_digest = build_compact_input_digest(
        task=task,
        ir=ir,
        event_pack=event_pack,
        xml_infos_by_body=xml_infos_by_body,
        video_duration_sec=video_duration_sec,
        sample_every_sec=sample_every_sec,
        max_frames=CONFIGS.optimization.critic_stage1_max_frames,
    )
    return stage1_digest, build_stage1_critic_user_content(
        task=task,
        input_digest=stage1_digest,
        sampled_frames=sampled_frames,
    )


def parse_analysis_json(raw_text: str) -> dict[str, Any]:
    return ensure_sectioned_analysis(extract_first_json_object(raw_text))


def probe_video_duration(video_path: Path) -> float | None:
    try:
        return probe_video_duration_sec(video_path)
    except VideoSamplingError:
        return None


def sample_video_frames(
    *,
    video_path: Path,
    sample_every_sec: float,
    max_frames: int,
    max_width: int,
) -> list[SampledFrame]:
    try:
        return sample_video_frames_as_data_urls(
            video_path,
            sample_every_sec=sample_every_sec,
            max_frames=max_frames,
            max_width=max_width,
        )
    except VideoSamplingError as exc:
        raise RuntimeError(str(exc)) from exc
