from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..configs import CONFIGS


@dataclass(slots=True)
class OptimizationConfig:
    model: str = CONFIGS.optimization.model
    xml_model: str | None = None
    critic_model: str | None = None
    hosted_prompt_id: str | None = None
    hosted_prompt_version: str | None = None
    critic_hosted_prompt_id: str | None = None
    critic_hosted_prompt_version: str | None = None
    critic_prompt_variant: str = CONFIGS.optimization.critic_prompt_variant
    temperature: float | None = None
    critic_temperature: float | None = None
    reasoning_effort: str | None = None
    critic_reasoning_effort: str | None = None
    backend: str = CONFIGS.optimization.backend
    max_opt_rounds: int = CONFIGS.optimization.max_opt_rounds
    generator_max_rounds: int = CONFIGS.optimization.max_attempts
    xml_max_attempts: int = CONFIGS.optimization.xml_max_attempts
    timeout_sec: float = CONFIGS.optimization.timeout_sec
    assets_dir: str = "agent/generated_assets"
    mesh_assets_dir: str = "agent/generated_meshes"
    sample_every_sec: float = CONFIGS.optimization.sample_every_sec
    max_frames: int = CONFIGS.optimization.max_frames
    max_width: int = CONFIGS.optimization.max_width
    output_root: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    mesh_texture_enabled: bool = CONFIGS.meshy_request.texture_enabled


@dataclass(slots=True)
class OptimizationRoundResult:
    round_index: int
    verdict: str | None
    passed: bool
    round_dir: str


@dataclass(slots=True)
class OptimizationResult:
    task: str
    status: str
    rounds: list[OptimizationRoundResult]
    final_round_dir: str
    final_verdict: str | None


@dataclass(slots=True)
class OptimizationTaskSpec:
    case_id: str
    task: str


@dataclass(slots=True)
class BatchOptimizationItemResult:
    case_id: str
    task: str
    status: str
    final_round_dir: str
    final_verdict: str | None
    rounds: list[OptimizationRoundResult]
    error: str | None = None


@dataclass(slots=True)
class BatchOptimizationResult:
    status: str
    run_root: str
    items: list[BatchOptimizationItemResult]


@dataclass(slots=True)
class RoundWorkspace:
    round_dir: Path
    assets_dir: Path
    mesh_assets_dir: Path
    ir_generated: Path
    generation_log: Path
    ir_run: Path
    ir_validated: Path
    run_result: Path
    event_pack: Path
    critic_json: Path
    critic_log: Path
    usage_json: Path
    task_txt: Path
    generator_feedback_txt: Path
    generator_feedback_json: Path
    video_path: Path
