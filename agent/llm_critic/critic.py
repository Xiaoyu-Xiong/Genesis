from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..configs import CONFIGS
from ..io_utils import load_json_object
from ..llm_generator.client.openai_client import OpenAIRequestError, OpenAIResponsesClient
from ..llm_generator.client.responses_format import coerce_content_to_text
from ..usage import aggregate_usage_metrics
from .evaluation_support import (
    build_critic_log_payload,
    build_single_stage_inputs,
    build_stage1_inputs,
    emit_critic_progress,
    emit_retrieval_progress,
    message_usage,
    parse_analysis_json,
    probe_video_duration,
    sample_video_frames,
)
from .digest import (
    load_optional_texts_by_body,
)
from .prompting import (
    CRITIC_RETRIEVAL_SYSTEM_PROMPT,
    CRITIC_STAGE1_SYSTEM_PROMPT,
    build_critic_hosted_prompt_ref,
    build_stage1_critic_prompt_cache_key,
    build_stage2_critic_prompt_cache_key,
    build_stage2_retrieval_user_content,
)
from .tool_library import CriticToolLibrary
from .video_sampler import SampledFrame


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


@dataclass(slots=True, frozen=True)
class CriticEvaluationResult:
    model: str
    analysis_json: dict[str, object]
    input_digest: dict[str, object]
    frames_used: int
    raw_response_text: str
    usage_summary: dict[str, int] = field(default_factory=dict)
    stage_logs: list[dict[str, object]] = field(default_factory=list)


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
    log_path: Path | None = None,
) -> CriticEvaluationResult:
    if prompt_variant not in {"full", "compact"}:
        raise CriticEvaluationError(f"Unsupported critic prompt_variant `{prompt_variant}`.")

    try:
        ir = load_json_object(eval_input.ir_path, label="IR")
        event_pack = load_json_object(eval_input.event_pack_path, label="Event pack")
        xml_infos_by_body = load_optional_texts_by_body(eval_input.xml_paths_by_body)
    except ValueError as exc:
        raise CriticEvaluationError(str(exc)) from exc

    video_duration_sec = probe_video_duration(eval_input.video_path)
    xml_texts_by_body = {body_name: info["text"] for body_name, info in xml_infos_by_body.items()}

    if not CONFIGS.optimization.critic_two_stage:
        try:
            input_digest, sampled_frames, content, prompt_cache_key, system_prompt = build_single_stage_inputs(
                task=eval_input.task,
                ir=ir,
                event_pack=event_pack,
                xml_infos_by_body=xml_infos_by_body,
                xml_texts_by_body=xml_texts_by_body,
                video_duration_sec=video_duration_sec,
                sample_every_sec=eval_input.sample_every_sec,
                max_frames=eval_input.max_frames,
                max_width=eval_input.max_width,
                video_path=eval_input.video_path,
                prompt_variant=prompt_variant,
            )
        except RuntimeError as exc:
            raise CriticEvaluationError(str(exc)) from exc
        stage_result = _run_critic_response(
            client=client,
            model=model,
            system_prompt=system_prompt,
            content=content,
            prompt_cache_key=prompt_cache_key,
            hosted_prompt_id=hosted_prompt_id,
            hosted_prompt_version=hosted_prompt_version,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            stage="single",
            prompt_variant=prompt_variant,
        )
        return CriticEvaluationResult(
            model=model,
            analysis_json=stage_result["analysis_json"],
            input_digest=input_digest,
            frames_used=len(sampled_frames),
            raw_response_text=stage_result["raw_response_text"],
            usage_summary=aggregate_usage_metrics([stage_result.get("usage")]),
            stage_logs=[stage_result],
        )

    try:
        stage1_frames = sample_video_frames(
            video_path=eval_input.video_path,
            sample_every_sec=eval_input.sample_every_sec,
            max_frames=CONFIGS.optimization.critic_stage1_max_frames,
            max_width=CONFIGS.optimization.critic_stage1_max_width,
        )
    except RuntimeError as exc:
        raise CriticEvaluationError(str(exc)) from exc
    stage1_digest, stage1_content = build_stage1_inputs(
        task=eval_input.task,
        ir=ir,
        event_pack=event_pack,
        xml_infos_by_body=xml_infos_by_body,
        xml_texts_by_body=xml_texts_by_body,
        video_duration_sec=video_duration_sec,
        sample_every_sec=eval_input.sample_every_sec,
        sampled_frames=stage1_frames,
    )
    stage1_result = _run_critic_response(
        client=client,
        model=model,
        system_prompt=CRITIC_STAGE1_SYSTEM_PROMPT,
        content=stage1_content,
        prompt_cache_key=build_stage1_critic_prompt_cache_key(),
        hosted_prompt_id=None,
        hosted_prompt_version=None,
        temperature=temperature,
        reasoning_effort=CONFIGS.optimization.critic_stage1_reasoning_effort,
        stage="critic_stage1",
        prompt_variant=CONFIGS.optimization.critic_stage1_prompt_variant,
    )
    emit_critic_progress(
        log_path=log_path,
        payload=build_critic_log_payload(
            model=model,
            input_digest=stage1_digest,
            frames_used=len(stage1_frames),
            raw_response_text=stage1_result["raw_response_text"],
            analysis_json=stage1_result["analysis_json"],
            stage_logs=[stage1_result],
        ),
    )

    if not _should_escalate(stage1_result["analysis_json"]):
        return CriticEvaluationResult(
            model=model,
            analysis_json=stage1_result["analysis_json"],
            input_digest=stage1_digest,
            frames_used=len(stage1_frames),
            raw_response_text=stage1_result["raw_response_text"],
            usage_summary=aggregate_usage_metrics([stage1_result.get("usage")]),
            stage_logs=[stage1_result],
        )

    try:
        stage2_frames = sample_video_frames(
            video_path=eval_input.video_path,
            sample_every_sec=eval_input.sample_every_sec,
            max_frames=eval_input.max_frames,
            max_width=eval_input.max_width,
        )
    except RuntimeError as exc:
        raise CriticEvaluationError(str(exc)) from exc
    stage2_result = _run_retrieval_critic(
        client=client,
        model=model,
        task=eval_input.task,
        compact_digest=stage1_digest,
        stage1_analysis=stage1_result["analysis_json"],
        sampled_frames=stage2_frames,
        ir=ir,
        event_pack=event_pack,
        xml_texts_by_body=xml_texts_by_body,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        progress_callback=(
            None
            if log_path is None
            else lambda stage_logs, raw_response_text, analysis_json: emit_critic_progress(
                log_path=log_path,
                payload=build_critic_log_payload(
                    model=model,
                    input_digest=stage1_digest,
                    frames_used=len(stage2_frames),
                    raw_response_text=stage1_result["raw_response_text"] if raw_response_text is None else raw_response_text,
                    analysis_json=stage1_result["analysis_json"] if analysis_json is None else analysis_json,
                    stage_logs=[stage1_result, *stage_logs],
                ),
            )
        ),
    )
    return CriticEvaluationResult(
        model=model,
        analysis_json=stage2_result["analysis_json"],
        input_digest=stage1_digest,
        frames_used=len(stage2_frames),
        raw_response_text=stage2_result["raw_response_text"],
        usage_summary=aggregate_usage_metrics([stage1_result.get("usage"), *[log.get("usage") for log in stage2_result["stage_logs"]]]),
        stage_logs=[stage1_result, *stage2_result["stage_logs"]],
    )

def _run_critic_response(
    *,
    client: OpenAIResponsesClient,
    model: str,
    system_prompt: str,
    content: list[dict[str, Any]],
    prompt_cache_key: str,
    hosted_prompt_id: str | None,
    hosted_prompt_version: str | None,
    temperature: float | None,
    reasoning_effort: str | None,
    stage: str,
    prompt_variant: str,
) -> dict[str, Any]:
    hosted_prompt = build_critic_hosted_prompt_ref(
        hosted_prompt_id=hosted_prompt_id,
        hosted_prompt_version=hosted_prompt_version,
    )
    try:
        message = client.responses_completion(
            model=model,
            messages=[{"role": "user", "content": content}] if hosted_prompt is not None else [
                {"role": "system", "content": system_prompt},
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
        analysis_json = parse_analysis_json(raw_text)
    except ValueError as exc:
        raise CriticEvaluationError(str(exc)) from exc
    return {
        "stage": stage,
        "prompt_variant": prompt_variant,
        "model": model,
        "raw_response_text": raw_text,
        "analysis_json": analysis_json,
        "usage": message_usage(message),
    }


def _run_retrieval_critic(
    *,
    client: OpenAIResponsesClient,
    model: str,
    task: str,
    compact_digest: dict[str, Any],
    stage1_analysis: dict[str, Any],
    sampled_frames: list[SampledFrame],
    ir: dict[str, Any],
    event_pack: dict[str, Any],
    xml_texts_by_body: dict[str, str],
    temperature: float | None,
    reasoning_effort: str | None,
    progress_callback: Callable[[list[dict[str, Any]], str | None, dict[str, Any] | None], None] | None = None,
) -> dict[str, Any]:
    tool_library = CriticToolLibrary(ir=ir, event_pack=event_pack, xml_texts_by_body=xml_texts_by_body)
    tool_specs = tool_library.tool_specs()
    previous_response_id: str | None = None
    stage_logs: list[dict[str, Any]] = []
    pending_messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": build_stage2_retrieval_user_content(
                task=task,
                compact_digest=compact_digest,
                stage1_analysis=stage1_analysis,
                sampled_frames=sampled_frames,
            ),
        }
    ]

    max_rounds = max(int(CONFIGS.optimization.critic_stage2_tool_max_rounds), 1)
    for round_idx in range(1, max_rounds + 1):
        message = client.responses_completion(
            model=model,
            messages=[{"role": "system", "content": CRITIC_RETRIEVAL_SYSTEM_PROMPT}, *pending_messages],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            prompt_cache_key=build_stage2_critic_prompt_cache_key(),
            previous_response_id=previous_response_id,
            tools=tool_specs,
            tool_choice="auto",
        )
        response_id = message.get("_response_id")
        if isinstance(response_id, str) and response_id:
            previous_response_id = response_id
        raw_text = coerce_content_to_text(message.get("content"))
        raw_tool_calls = message.get("tool_calls")
        usage = message_usage(message)

        stage_log: dict[str, Any] = {
            "stage": f"critic_stage2_round_{round_idx}",
            "prompt_variant": "retrieval",
            "model": model,
            "raw_response_text": raw_text,
            "usage": usage,
            "tool_results": [],
        }

        if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
            try:
                analysis_json = parse_analysis_json(raw_text)
            except ValueError as exc:
                raise CriticEvaluationError(str(exc)) from exc
            stage_logs.append(stage_log)
            emit_retrieval_progress(progress_callback, stage_logs, raw_text, analysis_json)
            return {"analysis_json": analysis_json, "raw_response_text": raw_text, "stage_logs": stage_logs}

        batch_calls: list[dict[str, Any]] = []
        for call_index, raw_call in enumerate(raw_tool_calls):
            function = raw_call.get("function", {}) if isinstance(raw_call, dict) else {}
            name = function.get("name") if isinstance(function, dict) else None
            arguments_json = function.get("arguments") if isinstance(function, dict) else None
            call_id = raw_call.get("id") if isinstance(raw_call, dict) else None
            if not isinstance(name, str):
                name = "unknown_tool"
            if not isinstance(arguments_json, str):
                arguments_json = "{}"
            if not isinstance(call_id, str):
                call_id = f"critic_tool_call_{round_idx}_{call_index}"
            batch_calls.append({"id": call_id, "name": name, "arguments_json": arguments_json})
        batch_results = tool_library.execute_tool_calls_batch(batch_calls)
        stage_logs.append(stage_log)
        emit_retrieval_progress(progress_callback, stage_logs, raw_text, None)
        for result in batch_results:
            stage_log["tool_results"].append(result)
            emit_retrieval_progress(progress_callback, stage_logs, raw_text, None)

        tool_messages: list[dict[str, Any]] = []
        for batch_call, result in zip(batch_calls, batch_results, strict=True):
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": batch_call["id"],
                    "name": batch_call["name"],
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

        if round_idx < max_rounds:
            pending_messages = tool_messages
            continue

        final_message = client.responses_completion(
            model=model,
            messages=[
                *tool_messages,
                {
                    "role": "user",
                    "content": "Use the evidence and tool outputs already gathered to return the final critique JSON now. Do not call more tools.",
                },
            ],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            prompt_cache_key=build_stage2_critic_prompt_cache_key(),
            previous_response_id=previous_response_id,
            response_format={"type": "json_object"},
        )
        final_raw_text = coerce_content_to_text(final_message.get("content"))
        final_usage = message_usage(final_message)
        stage_logs.append(
            {
                "stage": f"critic_stage2_round_{round_idx + 1}",
                "prompt_variant": "retrieval_finalize",
                "model": model,
                "raw_response_text": final_raw_text,
                "usage": final_usage,
                "tool_results": [],
            }
        )
        try:
            analysis_json = parse_analysis_json(final_raw_text)
        except ValueError as exc:
            raise CriticEvaluationError(str(exc)) from exc
        emit_retrieval_progress(progress_callback, stage_logs, final_raw_text, analysis_json)
        return {"analysis_json": analysis_json, "raw_response_text": final_raw_text, "stage_logs": stage_logs}

    raise CriticEvaluationError("Stage-2 retrieval critic exceeded configured tool rounds without returning final JSON.")


def _should_escalate(stage1_analysis: dict[str, Any]) -> bool:
    verdict = stage1_analysis.get("verdict")
    if CONFIGS.optimization.critic_force_escalate_on_stage1_pass and verdict == "pass":
        return True

    needs_escalation = stage1_analysis.get("needs_escalation")
    if isinstance(needs_escalation, bool):
        return needs_escalation
    score = stage1_analysis.get("overall_score")
    if verdict == "pass" and isinstance(score, int | float) and float(score) >= 90.0:
        return False
    return True
