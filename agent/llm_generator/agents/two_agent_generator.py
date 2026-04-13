from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from ...mesh.summary import estimate_scaled_bbox_size, load_mesh_asset_summary
from ..client import OpenAIResponsesClient
from ...tool_library import GeneralIRAgentToolLibrary
from .ir_agent import IRGenerationResult, generate_ir_with_tool_agent
from .mesh_agent import MeshGenerationResult, generate_mesh_asset_with_meshy, load_existing_mesh_generation_result
from .xml_agent import XMLGenerationResult, generate_articulated_xml_with_openai


@dataclass(slots=True)
class TwoAgentGenerationResult:
    model: str
    mode: str
    articulated_requested: bool
    ir_result: IRGenerationResult
    xml_results_by_body: dict[str, XMLGenerationResult]
    mesh_results_by_body: dict[str, MeshGenerationResult]

    @property
    def ir_json(self) -> dict[str, Any]:
        return self.ir_result.ir_json


def _prepare_asset_dir(path_like: str | Path) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _merge_mesh_task(base_task: str, body_name: str) -> str:
    return "\n\n".join(
        [
            base_task.strip(),
            f"Target non-articulated mesh body name: `{body_name}`.",
            "Generate one single-object simulation-friendly mesh asset for this body only.",
        ]
    )


def _previous_shape_path_by_body(
    previous_ir_json: dict[str, Any] | None,
    *,
    allowed_kinds: set[str],
) -> dict[str, str]:
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
        if shape.get("kind") not in allowed_kinds:
            continue
        file_path = shape.get("file")
        if isinstance(file_path, str) and file_path.strip():
            paths_by_body[body_name] = file_path
    return paths_by_body


def _previous_xml_path_by_body(previous_ir_json: dict[str, Any] | None) -> dict[str, str]:
    return _previous_shape_path_by_body(previous_ir_json, allowed_kinds={"mjcf", "urdf"})


def _previous_mesh_path_by_body(previous_ir_json: dict[str, Any] | None) -> dict[str, str]:
    return _previous_shape_path_by_body(previous_ir_json, allowed_kinds={"mesh"})


def _previous_mesh_summaries_by_body(previous_ir_json: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    paths_by_body = _previous_mesh_path_by_body(previous_ir_json)
    summaries: dict[str, dict[str, Any]] = {}
    bodies_any = previous_ir_json.get("bodies") if isinstance(previous_ir_json, dict) else None
    previous_bodies = [body for body in bodies_any if isinstance(body, dict)] if isinstance(bodies_any, list) else []
    body_payload_by_name = {
        body.get("name"): body for body in previous_bodies if isinstance(body.get("name"), str) and body.get("name")
    }
    for body_name, path_str in sorted(paths_by_body.items()):
        path = Path(path_str)
        if not path.exists():
            continue
        summary = load_mesh_asset_summary(path)
        previous_body = body_payload_by_name.get(body_name, {})
        shape = previous_body.get("shape") if isinstance(previous_body, dict) else {}
        scale = shape.get("scale") if isinstance(shape, dict) else None
        if isinstance(scale, int | float) and not isinstance(scale, bool):
            summary["applied_scale"] = float(scale)
            summary["estimated_bbox_size_after_scale"] = estimate_scaled_bbox_size(summary.get("bbox_size"), float(scale))
        existing = load_existing_mesh_generation_result(path)
        if existing is not None:
            summary["raw_manifold_ok"] = existing.raw_manifold_ok
            summary["repaired_manifold_ok"] = existing.repaired_manifold_ok
        summaries[body_name] = summary
    return summaries


def _filtered_previous_xml_texts_by_body(
    previous_xml_texts_by_body: dict[str, str] | None,
    xml_feedback_requirements_by_body: dict[str, str] | None,
) -> dict[str, str]:
    if not previous_xml_texts_by_body or not xml_feedback_requirements_by_body:
        return {}
    return {
        body_name: xml_text
        for body_name, xml_text in previous_xml_texts_by_body.items()
        if body_name in xml_feedback_requirements_by_body
    }


def _build_additional_requirements(
    *,
    force_primitive_mode: bool,
    additional_requirements: str | None,
) -> str | None:
    requirement_lines: list[str] = []
    if force_primitive_mode:
        requirement_lines.append("Primitive-only mode: all bodies[].shape.kind must be one of sphere/box/cylinder.")
    if additional_requirements and additional_requirements.strip():
        requirement_lines.append(additional_requirements.strip())
    if not requirement_lines:
        return None
    return "\n\n".join(requirement_lines)


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


def _build_xml_generation_fn(
    *,
    task: str,
    model: str,
    xml_model: str | None,
    client: OpenAIResponsesClient,
    output_dir: Path,
    xml_max_attempts: int,
    temperature: float | None,
    reasoning_effort: str | None,
    xml_feedback_requirements_by_body: dict[str, str] | None,
    previous_xml_texts_by_body: dict[str, str] | None,
    previous_xml_paths_by_body: dict[str, str],
):
    def _xml_generation_fn(body_name: str, xml_task: str | None, file_stem: str | None) -> XMLGenerationResult:
        xml_feedback_requirements = None
        if xml_feedback_requirements_by_body is not None:
            xml_feedback_requirements = xml_feedback_requirements_by_body.get(body_name)
        previous_xml_text = None if previous_xml_texts_by_body is None else previous_xml_texts_by_body.get(body_name)
        previous_xml_path = previous_xml_paths_by_body.get(body_name)
        if xml_feedback_requirements is None and previous_xml_path:
            path = Path(previous_xml_path)
            if path.exists():
                xml_content = (
                    previous_xml_text
                    if isinstance(previous_xml_text, str) and previous_xml_text
                    else path.read_text(encoding="utf-8")
                )
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
        return generate_articulated_xml_with_openai(
            task=effective_xml_task,
            model=xml_model or model,
            client=client,
            output_dir=output_dir,
            file_stem=file_stem or body_name,
            previous_xml_text=previous_xml_text,
            max_attempts=xml_max_attempts,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

    return _xml_generation_fn


def _build_mesh_generation_fn(
    *,
    task: str,
    output_dir: Path,
    previous_mesh_paths_by_body: dict[str, str],
):
    def _mesh_generation_fn(body_name: str, mesh_task: str | None, file_stem: str | None) -> MeshGenerationResult:
        previous_mesh_path = previous_mesh_paths_by_body.get(body_name)
        if previous_mesh_path is not None:
            reused = load_existing_mesh_generation_result(previous_mesh_path)
            if reused is not None and reused.repaired_manifold_ok:
                return reused
        mesh_task_default = _merge_mesh_task(task, body_name)
        effective_mesh_task = mesh_task if isinstance(mesh_task, str) and mesh_task.strip() else mesh_task_default
        return generate_mesh_asset_with_meshy(
            task=effective_mesh_task,
            output_dir=output_dir,
            file_stem=file_stem or body_name,
        )

    return _mesh_generation_fn


def _determine_generation_mode(
    *,
    force_primitive_mode: bool,
    xml_results_by_body: dict[str, XMLGenerationResult],
    mesh_results_by_body: dict[str, MeshGenerationResult],
) -> str:
    if force_primitive_mode:
        return "single_agent_primitive"
    if xml_results_by_body or mesh_results_by_body:
        return "ir_agent_triggered_assets"
    return "ir_agent_no_xml"


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
    mesh_assets_dir: str | Path = "agent/generated_meshes",
    force_primitive_mode: bool = False,
    additional_requirements: str | None = None,
    xml_feedback_requirements_by_body: dict[str, str] | None = None,
    previous_ir_json: dict[str, Any] | None = None,
    previous_xml_texts_by_body: dict[str, str] | None = None,
    hosted_prompt_id: str | None = None,
    hosted_prompt_version: str | None = None,
) -> TwoAgentGenerationResult:
    xml_out_dir = _prepare_asset_dir(assets_dir)
    mesh_out_dir = _prepare_asset_dir(mesh_assets_dir)
    previous_xml_paths_by_body = _previous_xml_path_by_body(previous_ir_json)
    previous_mesh_paths_by_body = _previous_mesh_path_by_body(previous_ir_json)
    mesh_generation_available = bool(os.getenv("MESHY_API_KEY"))
    xml_generation_fn = None if force_primitive_mode else _build_xml_generation_fn(
        task=task,
        model=model,
        xml_model=xml_model,
        client=client,
        output_dir=xml_out_dir,
        xml_max_attempts=xml_max_attempts,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        xml_feedback_requirements_by_body=xml_feedback_requirements_by_body,
        previous_xml_texts_by_body=previous_xml_texts_by_body,
        previous_xml_paths_by_body=previous_xml_paths_by_body,
    )
    mesh_generation_fn = None if force_primitive_mode or not mesh_generation_available else _build_mesh_generation_fn(
        task=task,
        output_dir=mesh_out_dir,
        previous_mesh_paths_by_body=previous_mesh_paths_by_body,
    )

    tool_library = GeneralIRAgentToolLibrary(
        allowed_shape_kinds=(
            ("sphere", "box", "cylinder")
            if force_primitive_mode
            else ("sphere", "box", "cylinder", "mesh", "mjcf")
        ),
        enforce_articulated_actuator_control=True,
        xml_generation_fn=xml_generation_fn,
        mesh_generation_fn=mesh_generation_fn,
    )

    merged_requirements = _build_additional_requirements(
        force_primitive_mode=force_primitive_mode,
        additional_requirements=additional_requirements,
    )

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
        previous_xml_texts_by_body=_filtered_previous_xml_texts_by_body(
            previous_xml_texts_by_body,
            xml_feedback_requirements_by_body,
        ),
        previous_xml_summaries_by_body=_previous_xml_summaries_by_body(
            previous_ir_json,
            previous_xml_texts_by_body,
            xml_feedback_requirements_by_body,
        ),
        previous_mesh_summaries_by_body=_previous_mesh_summaries_by_body(previous_ir_json),
        hosted_prompt_id=hosted_prompt_id,
        hosted_prompt_version=hosted_prompt_version,
    )

    xml_results_by_body = tool_library.generated_xml_results_by_body
    mesh_results_by_body = tool_library.generated_mesh_results_by_body
    articulated_requested = any(body.shape.kind in {"mjcf", "urdf"} for body in ir_result.program.bodies)
    mode = _determine_generation_mode(
        force_primitive_mode=force_primitive_mode,
        xml_results_by_body=xml_results_by_body,
        mesh_results_by_body=mesh_results_by_body,
    )

    return TwoAgentGenerationResult(
        model=model,
        mode=mode,
        articulated_requested=articulated_requested,
        ir_result=ir_result,
        xml_results_by_body=xml_results_by_body,
        mesh_results_by_body=mesh_results_by_body,
    )
