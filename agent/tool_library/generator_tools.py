from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from ..ir_schema import RigidIR
from ..llm_generator.constraints.general_constraints import parse_sanitize_validate
from .overrides import GeneratorParameterOverrides, apply_generator_parameter_overrides
from .program_constraints import validate_program_constraints
from .tool_specs import (
    build_generation_bootstrap_payload,
    build_tool_specs,
)

ToolFunc = Callable[[dict[str, Any]], dict[str, Any]]
if TYPE_CHECKING:
    from ..llm_generator.agents.xml_agent import XMLGenerationResult

XMLGenerationFunc = Callable[[str, str | None, str | None], "XMLGenerationResult"]


class GeneralIRAgentToolLibrary:
    """Tool registry for IR agent (supports primitive + articulated)."""

    def __init__(
        self,
        *,
        required_shape_kind: str | None = None,
        required_shape_file: str | None = None,
        allowed_shape_kinds: tuple[str, ...] | None = None,
        enforce_articulated_actuator_control: bool = False,
        xml_generation_fn: XMLGenerationFunc | None = None,
        xml_task_default: str | None = None,
        target_sim_duration_sec: float | None = None,
        sim_duration_tolerance_sec: float = 0.75,
        parameter_overrides: GeneratorParameterOverrides | None = None,
    ) -> None:
        self.required_shape_kind = required_shape_kind
        self.required_shape_file = required_shape_file
        self.allowed_shape_kinds = allowed_shape_kinds
        self.enforce_articulated_actuator_control = enforce_articulated_actuator_control
        self.xml_generation_fn = xml_generation_fn
        self.xml_task_default = xml_task_default
        self.target_sim_duration_sec = target_sim_duration_sec
        self.sim_duration_tolerance_sec = sim_duration_tolerance_sec
        self.parameter_overrides = parameter_overrides
        self._generated_xml_results_by_body: dict[str, XMLGenerationResult] = {}
        self.generated_xml_shape_files_by_body: dict[str, str] = {}
        self.allowed_articulated_joint_names_by_body: dict[str, tuple[str, ...]] = {}
        self._tool_funcs: dict[str, ToolFunc] = {
            "get_generation_bootstrap": self._get_generation_bootstrap,
            "validate_ir": self._validate_ir,
        }
        if self.xml_generation_fn is not None:
            self._tool_funcs["generate_articulated_xml"] = self._generate_articulated_xml

    def tool_specs(self) -> list[dict[str, Any]]:
        return build_tool_specs(xml_generation_enabled=self.xml_generation_fn is not None)

    def execute_tool_call(self, *, name: str, arguments_json: str | None) -> dict[str, Any]:
        if name not in self._tool_funcs:
            return {"ok": False, "error": f"Unknown tool `{name}`."}

        try:
            args = self._parse_arguments(arguments_json)
            return self._tool_funcs[name](args)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def execute_tool_calls_batch(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not calls:
            return []

        results: list[dict[str, Any] | None] = [None] * len(calls)
        xml_jobs: list[tuple[int, str, dict[str, Any]]] = []
        non_xml_jobs: list[tuple[int, str, str | None]] = []

        for index, call in enumerate(calls):
            name = call.get("name")
            arguments_json = call.get("arguments_json")
            if name == "generate_articulated_xml":
                try:
                    args = self._parse_arguments(arguments_json)
                    xml_jobs.append((index, name, args))
                    continue
                except Exception as exc:  # noqa: BLE001
                    results[index] = {"ok": False, "error": str(exc)}
                    continue
            non_xml_jobs.append((index, name, arguments_json))

        if xml_jobs:
            with ThreadPoolExecutor(max_workers=len(xml_jobs)) as executor:
                futures = {
                    executor.submit(self._run_xml_generation_job, args): (index, name)
                    for index, name, args in xml_jobs
                }
                completed: list[tuple[int, str, Any]] = []
                for future, (index, name) in futures.items():
                    try:
                        completed.append((index, name, future.result()))
                    except Exception as exc:  # noqa: BLE001
                        completed.append((index, name, {"ok": False, "error": str(exc)}))

            for index, _name, result in sorted(completed, key=lambda item: item[0]):
                if isinstance(result, tuple) and len(result) == 2:
                    body_name, xml_result = result
                    results[index] = self._register_xml_generation_result(body_name, xml_result)
                else:
                    results[index] = result

        for index, name, arguments_json in non_xml_jobs:
            results[index] = self.execute_tool_call(name=name, arguments_json=arguments_json)

        return [result if result is not None else {"ok": False, "error": "Tool execution failed."} for result in results]

    def validate_program_constraints(
        self,
        program: RigidIR,
        *,
        target_sim_duration_sec: float | None = None,
        sim_duration_tolerance_sec: float | None = None,
    ) -> list[str]:
        program = self.apply_parameter_overrides(program)
        return validate_program_constraints(
            program,
            required_shape_kind=self.required_shape_kind,
            required_shape_file=self.required_shape_file,
            allowed_shape_kinds=self.allowed_shape_kinds,
            allowed_articulated_joint_names_by_body=self.allowed_articulated_joint_names_by_body,
            enforce_articulated_actuator_control=self.enforce_articulated_actuator_control,
            xml_generation_enabled=self.xml_generation_fn is not None,
            generated_xml_shape_files_by_body=self.generated_xml_shape_files_by_body,
            target_sim_duration_sec=(
                self.target_sim_duration_sec if target_sim_duration_sec is None else target_sim_duration_sec
            ),
            sim_duration_tolerance_sec=(
                self.sim_duration_tolerance_sec if sim_duration_tolerance_sec is None else sim_duration_tolerance_sec
            ),
        )

    @property
    def generated_xml_results_by_body(self) -> dict[str, XMLGenerationResult]:
        return dict(self._generated_xml_results_by_body)

    def apply_parameter_overrides(self, program: RigidIR) -> RigidIR:
        return apply_generator_parameter_overrides(program, self.parameter_overrides)

    @staticmethod
    def _parse_arguments(arguments_json: str | None) -> dict[str, Any]:
        if arguments_json is None or arguments_json.strip() == "":
            return {}
        parsed = json.loads(arguments_json)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments root must be an object")
        return parsed

    def _get_generation_bootstrap(self, _: dict[str, Any]) -> dict[str, Any]:
        return build_generation_bootstrap_payload(
            required_shape_kind=self.required_shape_kind,
            required_shape_file=self.required_shape_file,
            allowed_shape_kinds=self.allowed_shape_kinds,
            allowed_articulated_joint_names_by_body=self.allowed_articulated_joint_names_by_body,
            enforce_articulated_actuator_control=self.enforce_articulated_actuator_control,
            target_sim_duration_sec=self.target_sim_duration_sec,
            duration_tolerance_sec=self.sim_duration_tolerance_sec,
            xml_generation_enabled=self.xml_generation_fn is not None,
            generated_xml_paths_by_body=self.generated_xml_shape_files_by_body,
            parameter_overrides=self.parameter_overrides,
        )

    def _generate_articulated_xml(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.xml_generation_fn is None:
            return {"ok": False, "errors": ["`generate_articulated_xml` is not available in current mode."]}

        body_name = args.get("body_name")
        xml_task = args.get("xml_task")
        file_stem = args.get("file_stem")
        if not isinstance(body_name, str) or not body_name.strip():
            return {"ok": False, "errors": ["`body_name` must be a non-empty string."]}
        body_name = body_name.strip()
        xml_task_value = xml_task if isinstance(xml_task, str) and xml_task.strip() else self.xml_task_default
        file_stem_value = file_stem if isinstance(file_stem, str) and file_stem.strip() else body_name

        result = self.xml_generation_fn(body_name, xml_task_value, file_stem_value)
        return self._register_xml_generation_result(body_name, result)

    def _run_xml_generation_job(self, args: dict[str, Any]) -> tuple[str, XMLGenerationResult] | dict[str, Any]:
        if self.xml_generation_fn is None:
            return {"ok": False, "errors": ["`generate_articulated_xml` is not available in current mode."]}
        body_name = args["body_name"].strip()
        xml_task = args.get("xml_task")
        file_stem = args.get("file_stem")
        xml_task_value = xml_task if isinstance(xml_task, str) and xml_task.strip() else self.xml_task_default
        file_stem_value = file_stem if isinstance(file_stem, str) and file_stem.strip() else body_name
        result = self.xml_generation_fn(body_name, xml_task_value, file_stem_value)
        return body_name, result

    def _register_xml_generation_result(self, body_name: str, result: XMLGenerationResult) -> dict[str, Any]:
        from ..llm_generator.agents.xml_agent import list_named_joint_names

        joint_names = list_named_joint_names(result.xml_path)
        self._generated_xml_results_by_body[body_name] = result
        self.generated_xml_shape_files_by_body[body_name] = result.xml_path
        self.allowed_articulated_joint_names_by_body[body_name] = joint_names
        return {
            "ok": True,
            "body_name": body_name,
            "xml_path": result.xml_path,
            "joint_names": list(joint_names),
            "attempts": result.attempts,
        }

    def _validate_ir(self, args: dict[str, Any]) -> dict[str, Any]:
        candidate = args.get("candidate_ir")
        normalize = args.get("normalize", True)
        target_sim_duration_sec = args.get("target_sim_duration_sec")
        sim_duration_tolerance_sec = args.get("sim_duration_tolerance_sec")

        if not isinstance(candidate, dict):
            return {
                "ok": False,
                "errors": [
                    "`candidate_ir` must be a JSON object. Call validate_ir only after drafting a complete candidate_ir object, or output final JSON directly if the draft is already ready."
                ],
            }
        if not isinstance(normalize, bool):
            return {"ok": False, "errors": ["`normalize` must be a boolean."]}
        if target_sim_duration_sec is not None:
            if not isinstance(target_sim_duration_sec, (int, float)) or float(target_sim_duration_sec) <= 0:
                return {"ok": False, "errors": ["`target_sim_duration_sec` must be a positive number."]}
            target_sim_duration_sec = float(target_sim_duration_sec)
        if sim_duration_tolerance_sec is not None:
            if not isinstance(sim_duration_tolerance_sec, (int, float)) or float(sim_duration_tolerance_sec) <= 0:
                return {"ok": False, "errors": ["`sim_duration_tolerance_sec` must be a positive number."]}
            sim_duration_tolerance_sec = float(sim_duration_tolerance_sec)

        try:
            program = parse_sanitize_validate(candidate, normalize=normalize)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "errors": [str(exc)]}

        errors = self.validate_program_constraints(
            program,
            target_sim_duration_sec=target_sim_duration_sec,
            sim_duration_tolerance_sec=sim_duration_tolerance_sec,
        )
        if errors:
            return {"ok": False, "errors": errors}

        program = self.apply_parameter_overrides(program)
        return {"ok": True, "errors": [], "normalized_ir": program.model_dump(mode="json")}
