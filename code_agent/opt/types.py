from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from code_agent.configs import CONFIGS


DEFAULT_PLANNER_INTENT = "Optimize generated behavior if bounded continuous parameters appear to limit success."


@dataclass(slots=True, frozen=True)
class OptAgentRequest:
    case_dir: Path
    original_prompt: str | None = None
    planner_intent: str = DEFAULT_PLANNER_INTENT
    allowed_edits: tuple[str, ...] = (
        "src/action.py for control schedules, target poses, controller gains, force limits, and action hooks",
        "src/body.py for material, contact, density, friction, initial setting, layout, and body-parameter hooks only",
        "src/scene.py for solver/contact/timestep hooks only",
        "assets/xml/**/*.xml for validated scalar actuator/joint/geom parameter patches only; no topology edits",
        "contracts/*.json",
        "reports/*.json",
        "artifacts/opt_*",
    )
    forbidden_changes: tuple[str, ...] = (
        "Do not change task semantics.",
        "Do not directly write dynamic object state after initialization.",
        "Do not add hidden constraints or attachments.",
        "Do not add/remove XML bodies, joints, geoms, actuators, meshes, or change XML topology during Opt.",
        "Do not edit src/rendering.py or optimize rendering/camera/visual-only variables.",
    )
    max_rollouts: int | None = None
    backend: str = CONFIGS.opt.agent_backend
    timeout_sec: float = CONFIGS.opt.agent_timeout_sec
    render_baseline: bool = CONFIGS.opt.agent_render_baseline
    render_best: bool = CONFIGS.opt.agent_render_best
    steps: int | None = None
    duration_sec: float | None = None
    render_fps: int | None = None
    target_video_frames: int | None = None
    success_criteria: tuple[str, ...] = ()


@dataclass(slots=True)
class OptAgentResult:
    status: str
    case_type: str | None
    edited_files: list[str] = field(default_factory=list)
    optimized_variables: list[str] = field(default_factory=list)
    baseline: dict[str, Any] = field(default_factory=dict)
    best: dict[str, Any] = field(default_factory=dict)
    diagnosis: str | None = None
    recommendation: str | None = None
    evidence: list[str] = field(default_factory=list)
    opt_report_path: str | None = None
    subagent_report_path: str | None = None
    failures: list[str] = field(default_factory=list)
