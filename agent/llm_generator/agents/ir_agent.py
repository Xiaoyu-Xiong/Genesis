from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

from ...ir_schema import RigidIR
from ..client import OpenAIResponsesClient, coerce_content_to_text
from ..constraints.general_constraints import extract_first_json_object, parse_sanitize_validate
from ...tool_library import GeneralIRAgentToolLibrary
from ...tool_library.tool_specs import build_ir_agent_process_requirements
from .prompt_utils import truncate_prompt_text


class IRGenerationError(RuntimeError):
    pass


@dataclass(slots=True)
class IRGenerationRoundLog:
    round: int
    assistant_content: str | None
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    validation_error: str | None
    usage: dict[str, Any] | None


@dataclass(slots=True)
class IRGenerationResult:
    model: str
    rounds: int
    program: RigidIR
    ir_json: dict[str, Any]
    logs: list[IRGenerationRoundLog]


IR_SYSTEM_PROMPT = (
    "You are an IR planning agent for the Genesis rigid-scene IR. "
    "Use tools to fetch guide/schema and to validate draft candidates when useful. "
    "When done, output exactly one IR JSON object and nothing else."
)


def _compact_tool_call(raw_call: Any) -> dict[str, Any]:
    if not isinstance(raw_call, dict):
        return {"raw": raw_call}

    function = raw_call.get("function", {})
    name = function.get("name") if isinstance(function, dict) else None
    arguments = function.get("arguments") if isinstance(function, dict) else None
    return {"id": raw_call.get("id"), "name": name, "arguments": arguments}


def _tool_result_message(tool_call_id: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": json.dumps(result, ensure_ascii=False, default=str),
    }


def _build_process_requirements_prompt(
    task: str,
    *,
    mesh_generation_available: bool,
) -> str:
    lines = [
        "Process requirements:",
        "- Call get_generation_bootstrap first.",
        *build_ir_agent_process_requirements(mesh_generation_available=mesh_generation_available),
    ]
    return "\n".join(lines)


def _build_revision_prompt_sections(
    *,
    previous_ir_json: dict[str, Any] | None,
    previous_xml_texts_by_body: dict[str, str] | None,
    previous_xml_summaries_by_body: dict[str, dict[str, Any]] | None,
    previous_mesh_summaries_by_body: dict[str, dict[str, Any]] | None,
) -> list[str]:
    if previous_ir_json is None and not previous_xml_texts_by_body and not previous_xml_summaries_by_body and not previous_mesh_summaries_by_body:
        return []

    lines = [
        "",
        "Revision mode:",
        "- This is not a fresh generation pass.",
        "- You are revising the previous candidate based on critic feedback.",
        "- Preserve working parts of the previous candidate unless the feedback requires changing them.",
        "- Prefer targeted edits over broad regeneration.",
    ]
    if previous_ir_json is not None:
        lines.extend(
            [
                "",
                "Previous validated IR JSON to revise:",
                truncate_prompt_text(json.dumps(previous_ir_json, ensure_ascii=False, indent=2)),
            ]
        )
    if previous_xml_summaries_by_body:
        lines.extend(["", "Articulated asset summary for bodies that should usually stay unchanged:"])
        for body_name, summary in sorted(previous_xml_summaries_by_body.items()):
            lines.append(f"- {body_name}: {json.dumps(summary, ensure_ascii=False)}")
    if previous_mesh_summaries_by_body:
        lines.extend(["", "Existing mesh asset summary for bodies that may be reused:"])
        for body_name, summary in sorted(previous_mesh_summaries_by_body.items()):
            lines.append(f"- {body_name}: {json.dumps(summary, ensure_ascii=False)}")
    if previous_xml_texts_by_body:
        lines.extend(["", "Previous articulated XML assets to revise when XML changes are needed:"])
        for body_name, xml_text in sorted(previous_xml_texts_by_body.items()):
            if not xml_text.strip():
                continue
            lines.extend(
                [
                    "",
                    f"Body `{body_name}` previous XML:",
                    truncate_prompt_text(xml_text.strip()),
                ]
            )
    return lines


def _build_initial_messages(
    *,
    task: str,
    mesh_generation_available: bool,
    additional_requirements: str | None,
    previous_ir_json: dict[str, Any] | None,
    previous_xml_texts_by_body: dict[str, str] | None,
    previous_xml_summaries_by_body: dict[str, dict[str, Any]] | None,
    previous_mesh_summaries_by_body: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": _build_process_requirements_prompt(
                task,
                mesh_generation_available=mesh_generation_available,
            ),
        },
        {
            "role": "user",
            "content": "\n".join(["Task:", task.strip()]),
        },
    ]
    if additional_requirements and additional_requirements.strip():
        messages.append(
            {
                "role": "user",
                "content": "\n".join(["Additional hard requirements:", additional_requirements.strip()]),
            }
        )
    revision_sections = _build_revision_prompt_sections(
        previous_ir_json=previous_ir_json,
        previous_xml_texts_by_body=previous_xml_texts_by_body,
        previous_xml_summaries_by_body=previous_xml_summaries_by_body,
        previous_mesh_summaries_by_body=previous_mesh_summaries_by_body,
    )
    if revision_sections:
        messages.append({"role": "user", "content": "\n".join(revision_sections)})
    return messages


def _build_prompt_cache_key(tool_specs: list[dict[str, Any]]) -> str:
    signature = {
        "system": IR_SYSTEM_PROMPT,
        "tools": tool_specs,
    }
    digest = hashlib.sha1(json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"rigid_ir_agent:{digest}"


def _build_hosted_prompt_ref(
    *,
    hosted_prompt_id: str | None,
    hosted_prompt_version: str | None,
    task: str,
    additional_requirements: str | None,
    previous_ir_json: dict[str, Any] | None,
    previous_xml_texts_by_body: dict[str, str] | None,
    previous_xml_summaries_by_body: dict[str, dict[str, Any]] | None,
    previous_mesh_summaries_by_body: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if hosted_prompt_id is None:
        return None
    previous_xml_texts = ""
    if previous_xml_texts_by_body:
        sections: list[str] = []
        for body_name, xml_text in sorted(previous_xml_texts_by_body.items()):
            if not xml_text.strip():
                continue
            sections.append(f"Body `{body_name}` previous XML:\n{truncate_prompt_text(xml_text)}")
        previous_xml_texts = "\n\n".join(sections)
    previous_xml_summaries = ""
    if previous_xml_summaries_by_body:
        previous_xml_summaries = json.dumps(previous_xml_summaries_by_body, ensure_ascii=False, indent=2)
    previous_mesh_summaries = ""
    if previous_mesh_summaries_by_body:
        previous_mesh_summaries = json.dumps(previous_mesh_summaries_by_body, ensure_ascii=False, indent=2)
    prompt: dict[str, Any] = {
        "id": hosted_prompt_id,
        "variables": {
            "task": task,
            "additional_requirements": "" if additional_requirements is None else additional_requirements,
            "previous_ir_json": "" if previous_ir_json is None else truncate_prompt_text(
                json.dumps(previous_ir_json, ensure_ascii=False, indent=2)
            ),
            "previous_xml_texts_by_body": previous_xml_texts,
            "previous_xml_summaries_by_body": previous_xml_summaries,
            "previous_mesh_summaries_by_body": previous_mesh_summaries,
        },
    }
    if hosted_prompt_version is not None:
        prompt["version"] = hosted_prompt_version
    return prompt


def generate_ir_with_tool_agent(
    *,
    task: str,
    model: str,
    client: OpenAIResponsesClient,
    tool_library: GeneralIRAgentToolLibrary,
    max_rounds: int = 12,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    normalize: bool = True,
    additional_requirements: str | None = None,
    previous_ir_json: dict[str, Any] | None = None,
    previous_xml_texts_by_body: dict[str, str] | None = None,
    previous_xml_summaries_by_body: dict[str, dict[str, Any]] | None = None,
    previous_mesh_summaries_by_body: dict[str, dict[str, Any]] | None = None,
    hosted_prompt_id: str | None = None,
    hosted_prompt_version: str | None = None,
    progress_callback: Callable[[list[IRGenerationRoundLog]], None] | None = None,
) -> IRGenerationResult:
    if max_rounds < 1:
        raise ValueError("`max_rounds` must be >= 1.")

    system_message = {"role": "system", "content": IR_SYSTEM_PROMPT}
    pending_messages = _build_initial_messages(
        task=task,
        mesh_generation_available=tool_library.mesh_generation_fn is not None,
        additional_requirements=additional_requirements,
        previous_ir_json=previous_ir_json,
        previous_xml_texts_by_body=previous_xml_texts_by_body,
        previous_xml_summaries_by_body=previous_xml_summaries_by_body,
        previous_mesh_summaries_by_body=previous_mesh_summaries_by_body,
    )
    previous_response_id: str | None = None
    tool_specs = tool_library.tool_specs()
    prompt_cache_key = _build_prompt_cache_key(tool_specs)
    hosted_prompt = _build_hosted_prompt_ref(
        hosted_prompt_id=hosted_prompt_id,
        hosted_prompt_version=hosted_prompt_version,
        task=task,
        additional_requirements=additional_requirements,
        previous_ir_json=previous_ir_json,
        previous_xml_texts_by_body=previous_xml_texts_by_body,
        previous_xml_summaries_by_body=previous_xml_summaries_by_body,
        previous_mesh_summaries_by_body=previous_mesh_summaries_by_body,
    )

    if hosted_prompt is not None:
        pending_messages = []
    base_messages = [] if hosted_prompt is not None else [system_message]

    logs: list[IRGenerationRoundLog] = []
    for round_idx in range(1, max_rounds + 1):
        assistant_message = client.responses_completion(
            model=model,
            messages=[*base_messages, *pending_messages],
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            prompt=hosted_prompt,
            previous_response_id=previous_response_id,
            prompt_cache_key=prompt_cache_key,
            tools=tool_specs,
            tool_choice="auto",
        )
        response_id = assistant_message.get("_response_id")
        if isinstance(response_id, str) and response_id:
            previous_response_id = response_id

        assistant_content = coerce_content_to_text(assistant_message.get("content"))
        raw_tool_calls = assistant_message.get("tool_calls")
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        if isinstance(raw_tool_calls, list) and len(raw_tool_calls) > 0:
            next_messages: list[dict[str, Any]] = []
            batch_calls: list[dict[str, Any]] = []
            for call_index, raw_call in enumerate(raw_tool_calls):
                compact_call = _compact_tool_call(raw_call)
                tool_calls.append(compact_call)

                name = compact_call.get("name")
                arguments_json = compact_call.get("arguments")
                call_id = compact_call.get("id")

                if not isinstance(name, str):
                    name = "unknown_tool"
                if not isinstance(arguments_json, str):
                    arguments_json = "{}"
                if not isinstance(call_id, str):
                    call_id = f"synthetic_tool_call_{round_idx}_{call_index}"

                batch_calls.append({"id": call_id, "name": name, "arguments_json": arguments_json})

            round_log = IRGenerationRoundLog(
                round=round_idx,
                assistant_content=assistant_content or None,
                tool_calls=tool_calls,
                tool_results=tool_results,
                validation_error=None,
                usage=assistant_message.get("_usage") if isinstance(assistant_message.get("_usage"), dict) else None,
            )
            logs.append(round_log)
            _emit_progress(progress_callback, logs)

            batch_results = tool_library.execute_tool_calls_batch(batch_calls)
            for batch_call, result in zip(batch_calls, batch_results, strict=True):
                call_id = batch_call["id"]
                name = batch_call["name"]
                tool_results.append({"id": call_id, "name": name, "result": result})
                next_messages.append(_tool_result_message(call_id, name, result))
                _emit_progress(progress_callback, logs)

            successful_validate_result = None
            for tool_result in reversed(tool_results):
                if tool_result["name"] != "validate_ir":
                    continue
                result = tool_result["result"]
                if isinstance(result, dict) and result.get("ok") is True and isinstance(result.get("normalized_ir"), dict):
                    successful_validate_result = result
                    break

            if successful_validate_result is not None:
                try:
                    program = parse_sanitize_validate(successful_validate_result["normalized_ir"], normalize=normalize)
                    program = tool_library.apply_system_defaults(program)
                    constraint_errors = tool_library.validate_program_constraints(program)
                    if constraint_errors:
                        raise ValueError("; ".join(constraint_errors))
                    return IRGenerationResult(
                        model=model,
                        rounds=round_idx,
                        program=program,
                        ir_json=program.model_dump(mode="json"),
                        logs=logs,
                    )
                except Exception as exc:  # noqa: BLE001
                    pending_messages = [
                        {
                            "role": "user",
                            "content": (
                                "The validated candidate returned by validate_ir could not be finalized locally. "
                                f"Error: {exc}. Revise and return corrected JSON only."
                            ),
                        }
                    ]
                    continue

            pending_messages = next_messages
            continue

        validation_error: str | None = None
        try:
            payload = extract_first_json_object(assistant_content)
            program = parse_sanitize_validate(payload, normalize=normalize)
            program = tool_library.apply_system_defaults(program)
            constraint_errors = tool_library.validate_program_constraints(program)
            if constraint_errors:
                raise ValueError("; ".join(constraint_errors))
            logs.append(
                IRGenerationRoundLog(
                    round=round_idx,
                    assistant_content=assistant_content or None,
                    tool_calls=[],
                    tool_results=[],
                    validation_error=None,
                    usage=assistant_message.get("_usage") if isinstance(assistant_message.get("_usage"), dict) else None,
                )
            )
            _emit_progress(progress_callback, logs)
            return IRGenerationResult(
                model=model,
                rounds=round_idx,
                program=program,
                ir_json=program.model_dump(mode="json"),
                logs=logs,
            )
        except Exception as exc:  # noqa: BLE001
            validation_error = str(exc)

        logs.append(
            IRGenerationRoundLog(
                round=round_idx,
                assistant_content=assistant_content or None,
                tool_calls=[],
                tool_results=[],
                validation_error=validation_error,
                usage=assistant_message.get("_usage") if isinstance(assistant_message.get("_usage"), dict) else None,
            )
        )
        _emit_progress(progress_callback, logs)

        pending_messages = [
            {
                "role": "user",
                "content": (
                    "Your last response was not valid IR JSON. "
                    f"Validation error: {validation_error}. "
                    "Use tools and return corrected JSON only."
                ),
            }
        ]

    last_error = None
    for log in reversed(logs):
        if log.validation_error:
            last_error = log.validation_error
            break

    raise IRGenerationError(
        f"Failed to generate valid IR within {max_rounds} rounds. Last validation error: {last_error}"
    )


def _emit_progress(
    progress_callback: Callable[[list[IRGenerationRoundLog]], None] | None,
    logs: list[IRGenerationRoundLog],
) -> None:
    if progress_callback is not None:
        progress_callback(logs)
