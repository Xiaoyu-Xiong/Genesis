from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from code_agent.configs import runtime_defaults_dict


@dataclass(slots=True, frozen=True)
class TimingPlan:
    """Resolved simulation and video timing for one generated case."""

    duration_sec: float | None
    steps: int
    render_fps: int
    target_video_frames: int | None
    sim_dt: float
    sim_substeps: int
    render_every_n_steps: int
    render_res: tuple[int, int]
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "duration_sec": self.duration_sec,
            "steps": self.steps,
            "render_fps": self.render_fps,
            "target_video_frames": self.target_video_frames,
            "sim_dt": self.sim_dt,
            "sim_substeps": self.sim_substeps,
            "render_every_n_steps": self.render_every_n_steps,
            "render_res": list(self.render_res),
            "source": self.source,
        }


def resolve_timing(
    *,
    planner_output: Mapping[str, Any] | None,
    steps: int | None = None,
    duration_sec: float | None = None,
    render_fps: int | None = None,
) -> TimingPlan:
    """Resolve planner/CLI timing into explicit step and video budgets."""

    execution_plan = _execution_plan(planner_output)
    defaults = runtime_defaults_dict(ipc_enabled=_planner_ipc_enabled(planner_output))
    default_sim_dt = float(defaults["sim_dt"])
    default_sim_substeps = int(defaults["sim_substeps"])
    default_render_every_n_steps = int(defaults["render_every_n_steps"])
    default_render_fps = int(defaults["render_fps"])
    default_render_res = _render_res(defaults["render_res"]) or (640, 480)

    sim_dt = _positive_float(execution_plan.get("sim_dt")) or default_sim_dt
    sim_substeps = _positive_int(execution_plan.get("sim_substeps")) or default_sim_substeps
    render_every_n_steps = _positive_int(execution_plan.get("render_every_n_steps")) or default_render_every_n_steps
    render_res = _render_res(execution_plan.get("render_res")) or default_render_res
    planned_duration = _positive_float(execution_plan.get("duration_sec"))
    planned_steps = _non_negative_int(execution_plan.get("step_budget"))
    planned_fps = _positive_int(execution_plan.get("render_fps"))
    planned_frames = _positive_int(execution_plan.get("render_budget"))

    source_parts: list[str] = []
    if "sim_dt" in execution_plan:
        source_parts.append("planner_sim_dt")
    if "sim_substeps" in execution_plan:
        source_parts.append("planner_sim_substeps")
    if "render_every_n_steps" in execution_plan:
        source_parts.append("planner_render_every_n_steps")
    if "render_res" in execution_plan:
        source_parts.append("planner_render_res")
    resolved_duration = planned_duration
    if resolved_duration is not None:
        source_parts.append("planner_duration")
    if resolved_duration is not None:
        _validate_finite_positive(resolved_duration, "duration_sec")
    if duration_sec is not None:
        resolved_duration = _require_positive_float(duration_sec, "duration_sec")
        source_parts.append("cli_duration")

    fps = planned_fps or default_render_fps
    if planned_fps is not None:
        source_parts.append("planner_fps")
    if render_fps is not None:
        fps = _require_positive_int(render_fps, "render_fps")
        source_parts.append("cli_fps")

    if steps is not None:
        resolved_steps = _require_non_negative_int(steps, "steps")
        source_parts.append("cli_steps")
    elif resolved_duration is not None:
        resolved_steps = max(1, math.ceil(resolved_duration / sim_dt))
        source_parts.append("duration_to_steps")
    elif planned_steps is not None:
        resolved_steps = planned_steps
        source_parts.append("planner_steps")
    else:
        raise ValueError("Timing requires planner execution_plan.step_budget or explicit --steps/--duration-sec")

    if resolved_duration is None and resolved_steps is not None:
        resolved_duration = resolved_steps * sim_dt
        source_parts.append("steps_to_duration")

    target_video_frames = planned_frames
    if target_video_frames is not None:
        source_parts.append("planner_render_budget")
    if resolved_duration is not None:
        target_video_frames = max(1, round(resolved_duration * fps))
        source_parts.append("duration_to_frames")

    return TimingPlan(
        duration_sec=resolved_duration,
        steps=resolved_steps,
        render_fps=fps,
        target_video_frames=target_video_frames,
        sim_dt=sim_dt,
        sim_substeps=sim_substeps,
        render_every_n_steps=render_every_n_steps,
        render_res=render_res,
        source="+".join(dict.fromkeys(source_parts)) or "explicit",
    )


def _execution_plan(planner_output: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(planner_output, Mapping):
        return {}
    execution_plan = planner_output.get("execution_plan")
    return execution_plan if isinstance(execution_plan, Mapping) else {}


def _planner_ipc_enabled(planner_output: Mapping[str, Any] | None) -> bool:
    if not isinstance(planner_output, Mapping):
        return False
    physics_plan = planner_output.get("physics_plan")
    if not isinstance(physics_plan, Mapping):
        return False
    if str(physics_plan.get("mode") or "") in {"rigid_ipc", "fem_ipc"}:
        return True
    return bool(physics_plan.get("ipc_enabled") or physics_plan.get("deformable_enabled"))


def _render_res(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        width = int(value[0])
        height = int(value[1])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def _positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _non_negative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _require_positive_float(value: Any, name: str) -> float:
    number = _positive_float(value)
    if number is None:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _require_positive_int(value: Any, name: str) -> int:
    number = _positive_int(value)
    if number is None:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _require_non_negative_int(value: Any, name: str) -> int:
    number = _non_negative_int(value)
    if number is None:
        raise ValueError(f"{name} must be a non-negative integer")
    return number


def _validate_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
