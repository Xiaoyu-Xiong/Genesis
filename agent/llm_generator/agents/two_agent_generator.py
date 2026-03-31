from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..client import OpenAIResponsesClient
from ...tool_library import GeneralIRAgentToolLibrary, GeneratorParameterOverrides
from .ir_agent import IRGenerationResult, generate_ir_with_tool_agent
from .xml_agent import XMLGenerationResult, generate_articulated_xml_with_openai


@dataclass(slots=True)
class TwoAgentGenerationResult:
    model: str
    mode: str
    articulated_requested: bool
    ir_result: IRGenerationResult
    xml_results_by_body: dict[str, XMLGenerationResult]

    @property
    def ir_json(self) -> dict[str, Any]:
        return self.ir_result.ir_json


def _merge_xml_task(base_task: str, body_name: str, xml_feedback_requirements: str | None) -> str:
    body_header = f"Target articulated body name: `{body_name}`."
    if xml_feedback_requirements is None or not xml_feedback_requirements.strip():
        return "\n\n".join([base_task.strip(), body_header])
    return "\n\n".join(
        [
            base_task.strip(),
            body_header,
            "XML-specific repair requirements for this round:",
            xml_feedback_requirements.strip(),
        ]
    )


def _previous_xml_path_by_body(previous_ir_json: dict[str, Any] | None) -> dict[str, str]:
    if previous_ir_json is None:
        return {}
    bodies_any = previous_ir_json.get("bodies")
    if not isinstance(bodies_any, list):
        return {}
    paths_by_body: dict[str, str] = {}
    for body in bodies_any:
        if not isinstance(body, dict):
            continue
        body_name = body.get("name")
        shape = body.get("shape")
        if not isinstance(body_name, str) or not isinstance(shape, dict):
            continue
        if shape.get("kind") not in {"mjcf", "urdf"}:
            continue
        file_path = shape.get("file")
        if isinstance(file_path, str) and file_path.strip():
            paths_by_body[body_name] = file_path
    return paths_by_body


def _previous_xml_summaries_by_body(
    previous_ir_json: dict[str, Any] | None,
    previous_xml_texts_by_body: dict[str, str] | None,
    xml_feedback_requirements_by_body: dict[str, str] | None,
) -> dict[str, dict[str, Any]]:
    from .xml_agent import list_named_joint_names

    paths_by_body = _previous_xml_path_by_body(previous_ir_json)
    summaries: dict[str, dict[str, Any]] = {}
    for body_name, path_str in sorted(paths_by_body.items()):
        if xml_feedback_requirements_by_body and body_name in xml_feedback_requirements_by_body:
            continue
        summary: dict[str, Any] = {"xml_path": path_str}
        path = Path(path_str)
        if path.exists():
            try:
                summary["joint_names"] = list(list_named_joint_names(path))
            except Exception:  # noqa: BLE001
                pass
        text = None if previous_xml_texts_by_body is None else previous_xml_texts_by_body.get(body_name)
        if isinstance(text, str) and text.strip():
            summary["xml_chars"] = len(text)
        summaries[body_name] = summary
    return summaries


def generate_ir_two_agent(
    *,
    task: str,
    model: str,
    client: OpenAIResponsesClient,
    xml_model: str | None = None,
    max_rounds: int = 12,
    xml_max_attempts: int = 4,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    normalize: bool = True,
    assets_dir: str | Path = "agent/generated_assets",
    force_primitive_mode: bool = False,
    additional_requirements: str | None = None,
    xml_feedback_requirements_by_body: dict[str, str] | None = None,
    previous_ir_json: dict[str, Any] | None = None,
    previous_xml_texts_by_body: dict[str, str] | None = None,
    hosted_prompt_id: str | None = None,
    hosted_prompt_version: str | None = None,
    parameter_overrides: GeneratorParameterOverrides | None = None,
) -> TwoAgentGenerationResult:
    xml_out_dir = Path(assets_dir)
    xml_out_dir.mkdir(parents=True, exist_ok=True)
    previous_xml_paths_by_body = _previous_xml_path_by_body(previous_ir_json)

    def _xml_generation_fn(body_name: str, xml_task: str | None, file_stem: str | None) -> XMLGenerationResult:
        xml_feedback_requirements = None
        if xml_feedback_requirements_by_body is not None:
            xml_feedback_requirements = xml_feedback_requirements_by_body.get(body_name)
        previous_xml_text = None if previous_xml_texts_by_body is None else previous_xml_texts_by_body.get(body_name)
        previous_xml_path = previous_xml_paths_by_body.get(body_name)
        if xml_feedback_requirements is None and previous_xml_path:
            path = Path(previous_xml_path)
            if path.exists():
                xml_content = previous_xml_text if isinstance(previous_xml_text, str) and previous_xml_text else path.read_text(encoding="utf-8")
                return XMLGenerationResult(
                    model=xml_model or model,
                    attempts=0,
                    xml_path=str(path.as_posix()),
                    xml_content=xml_content,
                    logs=[],
                )
        xml_task_default = _merge_xml_task(task, body_name, xml_feedback_requirements)
        effective_xml_task = xml_task if isinstance(xml_task, str) and xml_task.strip() else xml_task_default
        if xml_feedback_requirements and xml_feedback_requirements.strip() not in effective_xml_task:
            effective_xml_task = _merge_xml_task(effective_xml_task, body_name, xml_feedback_requirements)
        result = generate_articulated_xml_with_openai(
            task=effective_xml_task,
            model=xml_model or model,
            client=client,
            output_dir=xml_out_dir,
            file_stem=file_stem or body_name,
            previous_xml_text=previous_xml_text,
            max_attempts=xml_max_attempts,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        return result

    tool_library = GeneralIRAgentToolLibrary(
        allowed_shape_kinds=(
            ("sphere", "box", "cylinder")
            if force_primitive_mode
            else ("sphere", "box", "cylinder", "mjcf")
        ),
        enforce_articulated_actuator_control=True,
        xml_generation_fn=None if force_primitive_mode else _xml_generation_fn,
        parameter_overrides=parameter_overrides,
    )

    requirement_lines: list[str] = []
    if previous_ir_json is not None or (previous_xml_texts_by_body and any(text.strip() for text in previous_xml_texts_by_body.values())):
        requirement_lines.extend(
            [
                "This is a refinement pass, not a fresh generation pass.",
                "Modify the previous candidate based on feedback instead of starting over from scratch.",
                "Preserve working parts of the previous IR/XML unless the feedback requires changing them.",
            ]
        )
    if force_primitive_mode:
        requirement_lines.append("Primitive-only mode: all bodies[].shape.kind must be one of sphere/box/cylinder.")
    else:
        requirement_lines.extend(
            [
                "You may generate multiple bodies in one IR.",
                "Multiple articulated bodies are allowed (`mjcf` or `urdf`) alongside primitive bodies.",
                "Use `fixed=true` on primitive or URDF bodies that should stay anchored in the world. For MJCF bodies, express a fixed base in the XML itself.",
                "If articulated motion is needed, call `generate_articulated_xml` once per articulated body and bind each returned xml_path to the matching body in `bodies` with `shape.kind='mjcf'`.",
                "Always include `body_name` when calling `generate_articulated_xml`.",
                "Do not regenerate articulated XML assets that do not need structural changes; reuse their existing xml_path when possible.",
                "Do not define actuators inside XML; define actuators only on the articulated body in `bodies[].actuators`.",
                "For articulated bodies, do not use `set_pose` / `set_dofs_position` / `set_dofs_velocity`; "
                "use that body's actuators plus actuator commands (`set_target_pos` for position actuators, "
                "`set_torque` for motor actuators).",
            ]
        )
    requirement_lines.append(
        "If the task specifies a target simulation duration, enforce it via "
        "validate_ir(target_sim_duration_sec=..., sim_duration_tolerance_sec=...)."
    )
    merged_requirements = "\n".join(requirement_lines)
    if additional_requirements:
        merged_requirements = "\n\n".join([merged_requirements, additional_requirements.strip()])

    ir_result = generate_ir_with_tool_agent(
        task=task,
        model=model,
        client=client,
        tool_library=tool_library,
        max_rounds=max_rounds,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        normalize=normalize,
        additional_requirements=merged_requirements,
        previous_ir_json=previous_ir_json,
        previous_xml_texts_by_body={
            body_name: xml_text
            for body_name, xml_text in (previous_xml_texts_by_body or {}).items()
            if xml_feedback_requirements_by_body and body_name in xml_feedback_requirements_by_body
        },
        previous_xml_summaries_by_body=_previous_xml_summaries_by_body(
            previous_ir_json,
            previous_xml_texts_by_body,
            xml_feedback_requirements_by_body,
        ),
        hosted_prompt_id=hosted_prompt_id,
        hosted_prompt_version=hosted_prompt_version,
    )

    xml_results_by_body = tool_library.generated_xml_results_by_body
    articulated_requested = any(body.shape.kind in {"mjcf", "urdf"} for body in ir_result.program.bodies)
    if force_primitive_mode:
        mode = "single_agent_primitive"
    elif xml_results_by_body:
        mode = "ir_agent_triggered_xml"
    else:
        mode = "ir_agent_no_xml"

    return TwoAgentGenerationResult(
        model=model,
        mode=mode,
        articulated_requested=articulated_requested,
        ir_result=ir_result,
        xml_results_by_body=xml_results_by_body,
    )
