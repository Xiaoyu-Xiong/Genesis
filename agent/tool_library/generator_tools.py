from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from ..ir_schema import RigidIR
from ..mesh.summary import load_mesh_asset_summary
from ..llm_generator.constraints.general_constraints import parse_sanitize_validate
from .overrides import apply_system_defaults
from .program_constraints import validate_program_constraints
from .tool_specs import (
    build_generation_bootstrap_payload,
    build_tool_specs,
)

ToolFunc = Callable[[dict[str, Any]], dict[str, Any]]
if TYPE_CHECKING:
    from ..llm_generator.agents.mesh_agent import MeshGenerationResult
    from ..llm_generator.agents.xml_agent import XMLGenerationResult

XMLGenerationFunc = Callable[[str, str | None, str | None], "XMLGenerationResult"]
MeshGenerationFunc = Callable[[str, str | None, str | None], "MeshGenerationResult"]


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
        mesh_generation_fn: MeshGenerationFunc | None = None,
        xml_task_default: str | None = None,
        mesh_task_default: str | None = None,
        target_sim_duration_sec: float | None = None,
        sim_duration_tolerance_sec: float = 0.75,
    ) -> None:
        self.required_shape_kind = required_shape_kind
        self.required_shape_file = required_shape_file
        self.allowed_shape_kinds = allowed_shape_kinds
        self.enforce_articulated_actuator_control = enforce_articulated_actuator_control
        self.xml_generation_fn = xml_generation_fn
        self.mesh_generation_fn = mesh_generation_fn
        self.xml_task_default = xml_task_default
        self.mesh_task_default = mesh_task_default
        self.target_sim_duration_sec = target_sim_duration_sec
        self.sim_duration_tolerance_sec = sim_duration_tolerance_sec
        self._generated_xml_results_by_body: dict[str, XMLGenerationResult] = {}
        self._generated_mesh_results_by_body: dict[str, MeshGenerationResult] = {}
        self.generated_xml_shape_files_by_body: dict[str, str] = {}
        self.generated_mesh_shape_files_by_body: dict[str, str] = {}
        self.failed_generated_mesh_shape_files_by_body: dict[str, str] = {}
        self._mesh_results_by_reuse_key: dict[str, MeshGenerationResult] = {}
        self.allowed_articulated_joint_names_by_body: dict[str, tuple[str, ...]] = {}
        self._tool_funcs: dict[str, ToolFunc] = {
            "get_generation_bootstrap": self._get_generation_bootstrap,
            "validate_ir": self._validate_ir,
        }
        if self.xml_generation_fn is not None:
            self._tool_funcs["generate_articulated_xml"] = self._generate_articulated_xml
        if self.mesh_generation_fn is not None:
            self._tool_funcs["generate_mesh_asset"] = self._generate_mesh_asset

    def tool_specs(self) -> list[dict[str, Any]]:
        return build_tool_specs(
            xml_generation_enabled=self.xml_generation_fn is not None,
            mesh_generation_enabled=self.mesh_generation_fn is not None,
        )

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
        mesh_jobs: list[tuple[int, str, dict[str, Any]]] = []
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
            if name == "generate_mesh_asset":
                try:
                    args = self._parse_arguments(arguments_json)
                    mesh_jobs.append((index, name, args))
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

        if mesh_jobs:
            with ThreadPoolExecutor(max_workers=len(mesh_jobs)) as executor:
                futures = {
                    executor.submit(self._run_mesh_generation_job, args): (index, name)
                    for index, name, args in mesh_jobs
                }
                completed_mesh: list[tuple[int, str, Any]] = []
                for future, (index, name) in futures.items():
                    try:
                        completed_mesh.append((index, name, future.result()))
                    except Exception as exc:  # noqa: BLE001
                        completed_mesh.append((index, name, {"ok": False, "error": str(exc)}))
            for index, _name, result in sorted(completed_mesh, key=lambda item: item[0]):
                if isinstance(result, tuple) and len(result) == 2:
                    body_name, mesh_result = result
                    results[index] = self._register_mesh_generation_result(body_name, mesh_result)
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
        program = self.apply_system_defaults(program)
        return validate_program_constraints(
            program,
            required_shape_kind=self.required_shape_kind,
            required_shape_file=self.required_shape_file,
            allowed_shape_kinds=self.allowed_shape_kinds,
            allowed_articulated_joint_names_by_body=self.allowed_articulated_joint_names_by_body,
            enforce_articulated_actuator_control=self.enforce_articulated_actuator_control,
            xml_generation_enabled=self.xml_generation_fn is not None,
            generated_xml_shape_files_by_body=self.generated_xml_shape_files_by_body,
            mesh_generation_enabled=self.mesh_generation_fn is not None,
            generated_mesh_shape_files_by_body=self.generated_mesh_shape_files_by_body,
            failed_generated_mesh_shape_files_by_body=self.failed_generated_mesh_shape_files_by_body,
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

    @property
    def generated_mesh_results_by_body(self) -> dict[str, MeshGenerationResult]:
        return dict(self._generated_mesh_results_by_body)

    def apply_system_defaults(self, program: RigidIR) -> RigidIR:
        return apply_system_defaults(program)

    @staticmethod
    def _parse_arguments(arguments_json: str | None) -> dict[str, Any]:
        if arguments_json is None or arguments_json.strip() == "":
            return {}
        parsed = json.loads(arguments_json)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments root must be an object")
        return parsed

    def _get_generation_bootstrap(self, _: dict[str, Any]) -> dict[str, Any]:
        generated_mesh_summaries_by_body = {
            body_name: load_mesh_asset_summary(result.mesh_path)
            for body_name, result in sorted(self._generated_mesh_results_by_body.items())
        }
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
            mesh_generation_enabled=self.mesh_generation_fn is not None,
            generated_mesh_paths_by_body=self.generated_mesh_shape_files_by_body,
            generated_mesh_summaries_by_body=generated_mesh_summaries_by_body,
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

    def _generate_mesh_asset(self, args: dict[str, Any]) -> dict[str, Any]:
        if self.mesh_generation_fn is None:
            return {"ok": False, "errors": ["`generate_mesh_asset` is not available in current mode."]}
        parsed = self._parse_mesh_generation_args(args)
        if isinstance(parsed, dict):
            return parsed
        body_name, mesh_task_value, file_stem_value, reuse_key = parsed
        reused_result = self._reuse_mesh_result(reuse_key)
        if reused_result is not None:
            return self._register_mesh_generation_result(body_name, reused_result)
        result = self.mesh_generation_fn(body_name, mesh_task_value, file_stem_value)
        self._store_mesh_reuse_result(reuse_key, result)
        return self._register_mesh_generation_result(body_name, result)

    def _run_mesh_generation_job(self, args: dict[str, Any]) -> tuple[str, MeshGenerationResult] | dict[str, Any]:
        if self.mesh_generation_fn is None:
            return {"ok": False, "errors": ["`generate_mesh_asset` is not available in current mode."]}
        parsed = self._parse_mesh_generation_args(args)
        if isinstance(parsed, dict):
            return parsed
        body_name, mesh_task_value, file_stem_value, reuse_key = parsed
        reused_result = self._reuse_mesh_result(reuse_key)
        if reused_result is not None:
            return body_name, reused_result
        result = self.mesh_generation_fn(body_name, mesh_task_value, file_stem_value)
        self._store_mesh_reuse_result(reuse_key, result)
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

    def _register_mesh_generation_result(self, body_name: str, result: MeshGenerationResult) -> dict[str, Any]:
        self._generated_mesh_results_by_body[body_name] = result
        payload = self._mesh_result_payload(body_name, result)
        if not result.repaired_manifold_ok:
            self.failed_generated_mesh_shape_files_by_body[body_name] = result.mesh_path
            payload["ok"] = False
            payload["errors"] = [f"Generated mesh asset for `{body_name}` is not manifold-ready."]
            return payload
        self.generated_mesh_shape_files_by_body[body_name] = result.mesh_path
        self.failed_generated_mesh_shape_files_by_body.pop(body_name, None)
        payload["ok"] = True
        return payload

    def _parse_mesh_generation_args(
        self,
        args: dict[str, Any],
    ) -> tuple[str, str | None, str, str | None] | dict[str, Any]:
        body_name = args.get("body_name")
        if not isinstance(body_name, str) or not body_name.strip():
            return {"ok": False, "errors": ["`body_name` must be a non-empty string."]}
        body_name = body_name.strip()
        mesh_task = args.get("mesh_task")
        file_stem = args.get("file_stem")
        reuse_key = args.get("reuse_key")
        if isinstance(reuse_key, str):
            reuse_key = reuse_key.strip() or None
        else:
            reuse_key = None
        mesh_task_value = mesh_task if isinstance(mesh_task, str) and mesh_task.strip() else self.mesh_task_default
        file_stem_value = file_stem if isinstance(file_stem, str) and file_stem.strip() else body_name
        return body_name, mesh_task_value, file_stem_value, reuse_key

    def _reuse_mesh_result(self, reuse_key: str | None) -> MeshGenerationResult | None:
        if reuse_key is None:
            return None
        return self._mesh_results_by_reuse_key.get(reuse_key)

    def _store_mesh_reuse_result(self, reuse_key: str | None, result: MeshGenerationResult) -> None:
        if reuse_key is not None and result.repaired_manifold_ok:
            self._mesh_results_by_reuse_key[reuse_key] = result

    @staticmethod
    def _mesh_result_payload(body_name: str, result: MeshGenerationResult) -> dict[str, Any]:
        return {
            "body_name": body_name,
            "mesh_path": result.mesh_path,
            "runtime_mesh_path": result.mesh_path,
            "runtime_mesh_path_note": (
                "Use this repaired runtime mesh path in bodies[].shape.file. "
                "Do not use textured_mesh_path in the main simulation IR."
            ),
            "attempts": result.attempts,
            "raw_manifold_ok": result.raw_manifold_ok,
            "repaired_manifold_ok": result.repaired_manifold_ok,
            "texture_requested": result.texture_requested,
            "texture_succeeded": result.texture_succeeded,
            "textured_mesh_path": result.textured_mesh_path,
            "textured_mtl_path": result.textured_mtl_path,
            "base_color_path": result.base_color_path,
            "centroid_at_origin": result.centroid_at_origin,
            "centroid_before_translation": (
                list(result.centroid_before_translation) if result.centroid_before_translation is not None else None
            ),
            "bbox_min": list(result.bbox_min) if result.bbox_min is not None else None,
            "bbox_max": list(result.bbox_max) if result.bbox_max is not None else None,
            "bbox_size": list(result.bbox_size) if result.bbox_size is not None else None,
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

        program = self.apply_system_defaults(program)
        return {"ok": True, "errors": [], "normalized_ir": program.model_dump(mode="json")}
