from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..compiler_backend.generator import CompiledRigidArtifact, compile_rigid_ir_to_file, compile_rigid_ir_to_source
from ..ir_schema.program import RigidIR, normalize_ir, parse_ir_payload
from ..runtime.event_pack import build_llm_event_pack
from ..runtime.runner import run_rigid_ir


class RigidToolLibrary:
    """
    Tool-style API surface for LLM agents.

    Stable APIs:
    - `ir_schema`: return JSON schema for strict output constraints.
    - `validate_ir`: parse + optional normalization.
    - `compile_ir`: lower IR to executable Genesis python code.
    - `run_ir`: execute IR directly and return structured events (+ render metadata if enabled).
    - `run_ir_llm`: execute IR and return an LLM-oriented event pack JSON.
    - `generate_ir_from_text`: two-agent generation that auto-supports articulated prompts (IR agent + XML agent).
    - `generation_tool_specs`: expose callable tool schemas for agentic IR generation.
    - `generation_call_tool`: execute one generation tool call locally.

    Scope of current IR:
    - Multiple bodies per program (`bodies`), including rigid primitive bodies, non-articulated mesh bodies,
      deformable bodies, and multiple articulated MJCF/URDF bodies.
    - Action space includes pose edits, dof writes, external wrench application, actuator controls,
      and deformable observation fields.
    """

    def ir_schema(self) -> dict[str, Any]:
        return RigidIR.model_json_schema()

    def validate_ir(
        self,
        payload: Mapping[str, Any] | RigidIR,
        *,
        normalize: bool = True,
    ) -> RigidIR:
        parsed = parse_ir_payload(payload)
        return normalize_ir(parsed) if normalize else parsed

    def compile_ir(
        self,
        payload: Mapping[str, Any] | RigidIR,
    ) -> CompiledRigidArtifact:
        return compile_rigid_ir_to_source(payload)

    def compile_ir_to_file(
        self,
        payload: Mapping[str, Any] | RigidIR,
        output_path: str | Path,
    ) -> CompiledRigidArtifact:
        return compile_rigid_ir_to_file(payload, output_path)

    def run_ir(
        self,
        payload: Mapping[str, Any] | RigidIR,
        *,
        normalize: bool = True,
    ) -> dict[str, Any]:
        return run_rigid_ir(payload, normalize=normalize)

    def run_ir_llm(
        self,
        payload: Mapping[str, Any] | RigidIR,
        *,
        normalize: bool = True,
    ) -> dict[str, Any]:
        program = parse_ir_payload(payload)
        if normalize:
            program = normalize_ir(program)
        raw = run_rigid_ir(program, normalize=False)
        return build_llm_event_pack(program, raw)

    def generate_ir_from_text(
        self,
        task: str,
        *,
        model: str = "gpt-4.1-mini",
        xml_model: str | None = None,
        max_rounds: int = 12,
        xml_max_attempts: int = 4,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        normalize: bool = True,
        hosted_prompt_id: str | None = None,
        hosted_prompt_version: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        timeout_sec: float = 120.0,
        assets_dir: str = "agent/generated_assets",
        mesh_assets_dir: str = "agent/generated_meshes",
        force_primitive_mode: bool = False,
        mesh_texture_enabled: bool | None = None,
    ) -> RigidIR:
        from ..llm_generator.agents.two_agent_generator import generate_ir_two_agent
        from ..llm_generator.client.openai_client import OpenAIResponsesClient

        client = OpenAIResponsesClient.from_env(
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            timeout_sec=timeout_sec,
        )
        result = generate_ir_two_agent(
            task=task,
            model=model,
            client=client,
            xml_model=xml_model,
            max_rounds=max_rounds,
            xml_max_attempts=xml_max_attempts,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            normalize=normalize,
            assets_dir=assets_dir,
            mesh_assets_dir=mesh_assets_dir,
            force_primitive_mode=force_primitive_mode,
            hosted_prompt_id=hosted_prompt_id,
            hosted_prompt_version=hosted_prompt_version,
            mesh_texture_enabled=mesh_texture_enabled,
        )
        return result.ir_result.program

    def generation_tool_specs(self) -> list[dict[str, Any]]:
        from .generator_tools import GeneralIRAgentToolLibrary

        return GeneralIRAgentToolLibrary().tool_specs()

    def generation_call_tool(
        self,
        *,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        import json

        from .generator_tools import GeneralIRAgentToolLibrary

        library = GeneralIRAgentToolLibrary()
        arguments_json = json.dumps(dict(arguments or {}), ensure_ascii=False)
        return library.execute_tool_call(name=name, arguments_json=arguments_json)


TOOLS = RigidToolLibrary()
