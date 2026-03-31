from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..io_utils import load_json_object
from ..llm_generator.client import OpenAIRequestError, OpenAIResponsesClient, coerce_content_to_text
from ..tool_library import GeneratorParameterOverrides
from .digest import (
    build_compact_input_digest,
    build_input_digest,
    ensure_sectioned_analysis,
    extract_first_json_object,
    load_optional_texts_by_body,
)
from .prompting import (
    CRITIC_SYSTEM_PROMPT,
    build_compact_critic_prompt_cache_key,
    build_compact_critic_user_content,
    build_critic_hosted_prompt_ref,
    build_critic_prompt_cache_key,
    build_critic_user_content,
)
from .video_sampler import VideoSamplingError, probe_video_duration_sec, sample_video_frames_as_data_urls


class CriticEvaluationError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class CriticEvaluationInput:
    task: str
    ir_path: Path
    event_pack_path: Path
    video_path: Path
    xml_paths_by_body: dict[str, Path] = field(default_factory=dict)
    sample_every_sec: float = 0.5
    max_frames: int = 24
    max_width: int = 640
    generator_parameter_overrides: GeneratorParameterOverrides | None = None


@dataclass(slots=True, frozen=True)
class CriticEvaluationResult:
    model: str
    analysis_json: dict[str, object]
    input_digest: dict[str, object]
    frames_used: int
    raw_response_text: str


def evaluate_prompt_event_video(
    *,
    client: OpenAIResponsesClient,
    model: str,
    eval_input: CriticEvaluationInput,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    hosted_prompt_id: str | None = None,
    hosted_prompt_version: str | None = None,
    prompt_variant: str = "full",
) -> CriticEvaluationResult:
    if prompt_variant not in {"full", "compact"}:
        raise CriticEvaluationError(f"Unsupported critic prompt_variant `{prompt_variant}`.")

    try:
        ir = load_json_object(eval_input.ir_path, label="IR")
        event_pack = load_json_object(eval_input.event_pack_path, label="Event pack")
        xml_infos_by_body = load_optional_texts_by_body(eval_input.xml_paths_by_body)
    except ValueError as exc:
        raise CriticEvaluationError(str(exc)) from exc

    video_duration_sec = _probe_video_duration(eval_input.video_path)
    sampled_frames = _sample_video_frames(eval_input)

    if prompt_variant == "compact":
        input_digest = build_compact_input_digest(
            task=eval_input.task,
            ir=ir,
            event_pack=event_pack,
            xml_infos_by_body=xml_infos_by_body,
            video_duration_sec=video_duration_sec,
            sample_every_sec=eval_input.sample_every_sec,
            max_frames=eval_input.max_frames,
            parameter_overrides=eval_input.generator_parameter_overrides,
        )
        content = build_compact_critic_user_content(
            task=eval_input.task,
            ir=ir,
            event_pack=event_pack,
            xml_texts_by_body={body_name: info["text"] for body_name, info in xml_infos_by_body.items()},
            input_digest=input_digest,
            sampled_frames=sampled_frames,
        )
        prompt_cache_key = build_compact_critic_prompt_cache_key()
    else:
        input_digest = build_input_digest(
            task=eval_input.task,
            ir=ir,
            event_pack=event_pack,
            xml_infos_by_body=xml_infos_by_body,
            video_duration_sec=video_duration_sec,
            sample_every_sec=eval_input.sample_every_sec,
            max_frames=eval_input.max_frames,
            parameter_overrides=eval_input.generator_parameter_overrides,
        )
        content = build_critic_user_content(
            task=eval_input.task,
            ir=ir,
            event_pack=event_pack,
            xml_texts_by_body={body_name: info["text"] for body_name, info in xml_infos_by_body.items()},
            input_digest=input_digest,
            sampled_frames=sampled_frames,
        )
        prompt_cache_key = build_critic_prompt_cache_key()
    hosted_prompt = build_critic_hosted_prompt_ref(
        hosted_prompt_id=hosted_prompt_id,
        hosted_prompt_version=hosted_prompt_version,
    )

    try:
        message = client.responses_completion(
            model=model,
            messages=[{"role": "user", "content": content}] if hosted_prompt is not None else [
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            prompt=hosted_prompt,
            prompt_cache_key=prompt_cache_key,
            response_format={"type": "json_object"},
        )
    except OpenAIRequestError as exc:
        raise CriticEvaluationError(str(exc)) from exc

    raw_text = coerce_content_to_text(message.get("content"))
    try:
        analysis_json = ensure_sectioned_analysis(extract_first_json_object(raw_text))
    except ValueError as exc:
        raise CriticEvaluationError(str(exc)) from exc

    return CriticEvaluationResult(
        model=model,
        analysis_json=analysis_json,
        input_digest=input_digest,
        frames_used=len(sampled_frames),
        raw_response_text=raw_text,
    )


def _probe_video_duration(video_path: Path) -> float | None:
    try:
        return probe_video_duration_sec(video_path)
    except VideoSamplingError:
        return None


def _sample_video_frames(eval_input: CriticEvaluationInput):
    try:
        return sample_video_frames_as_data_urls(
            eval_input.video_path,
            sample_every_sec=eval_input.sample_every_sec,
            max_frames=eval_input.max_frames,
            max_width=eval_input.max_width,
        )
    except VideoSamplingError as exc:
        raise CriticEvaluationError(str(exc)) from exc
