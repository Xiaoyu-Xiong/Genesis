from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from code_agent.opt.contracts import get_dotted


@dataclass(slots=True)
class ObjectiveScore:
    score: float | None
    success: bool
    terms: dict[str, Any]
    measured: dict[str, Any]
    failure_penalty: float | None = None
    failure_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_report(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "success": self.success,
            "terms": self.terms,
            "measured": self.measured,
            "failure_penalty": self.failure_penalty,
            "failure_reason": self.failure_reason,
        }


def evaluate_objective(
    *,
    target_spec: dict[str, Any],
    metrics: dict[str, Any] | None,
    execution_failed: bool = False,
    failure_reason: str | None = None,
) -> ObjectiveScore:
    objective = target_spec["objective"]
    direction = str(objective["direction"])
    penalty = _failure_penalty(objective.get("failure_penalty"), direction)
    if execution_failed:
        return _failed_score(penalty, failure_reason or "execution_failed")
    if metrics is None:
        return _failed_score(penalty, failure_reason or "missing_metrics")

    terms: dict[str, Any] = {}
    measured: dict[str, Any] = {}
    warnings: list[str] = []
    score = 0.0
    for term in objective.get("terms", []):
        term_name = str(term["name"])
        metric_path = str(term["metric_path"])
        value = get_metric(metrics, metric_path)
        if value is _MISSING:
            return _failed_score(penalty, f"missing metric: {metric_path}")
        measured[metric_path] = value
        try:
            transformed = _transform_value(value, term)
            contribution = float(term["weight"]) * transformed
        except ValueError as exc:
            return _failed_score(penalty, f"{term_name}: {exc}")
        score += contribution
        terms[term_name] = {
            "metric_path": metric_path,
            "raw": value,
            "target": term.get("target", term.get("success_threshold")),
            "transform": term["transform"],
            "value": transformed,
            "weight": term["weight"],
            "contribution": contribution,
        }
        if term["transform"] == "custom":
            warnings.append(f"custom transform for {term_name!r} is treated as zero in version 1")

    success = _success_from_criteria(target_spec.get("success_criteria", []), metrics, measured, warnings)
    return ObjectiveScore(
        score=float(score),
        success=success,
        terms=terms,
        measured=measured,
        warnings=warnings,
    )


def get_metric(metrics: dict[str, Any], metric_path: str) -> Any:
    value = get_dotted(metrics, metric_path, _MISSING)
    if value is not _MISSING:
        return value
    return _MISSING


def _failed_score(penalty: float, reason: str) -> ObjectiveScore:
    return ObjectiveScore(
        score=float(penalty),
        success=False,
        terms={},
        measured={},
        failure_penalty=float(penalty),
        failure_reason=reason,
    )


def _failure_penalty(raw_penalty: Any, direction: str) -> float:
    if raw_penalty is None:
        return -1.0e9 if direction == "maximize" else 1.0e9
    try:
        penalty = float(raw_penalty)
    except (TypeError, ValueError):
        return -1.0e9 if direction == "maximize" else 1.0e9
    if not math.isfinite(penalty):
        return -1.0e9 if direction == "maximize" else 1.0e9
    if direction == "maximize" and penalty > 0.0:
        return -penalty
    if direction == "minimize" and penalty < 0.0:
        return abs(penalty)
    return penalty


def _transform_value(value: Any, term: dict[str, Any]) -> float:
    transform = str(term["transform"])
    if transform == "custom":
        return 0.0
    if transform == "reward_if_true":
        return 1.0 if bool(value) else 0.0
    if transform == "penalty_if_true":
        return 1.0 if bool(value) else 0.0

    value_float = _as_number(value, str(term["metric_path"]))
    if transform == "identity":
        return value_float
    if transform == "absolute_error":
        return abs(value_float - _as_number(term.get("target"), f"{term['name']}.target"))
    if transform == "squared_error":
        error = value_float - _as_number(term.get("target"), f"{term['name']}.target")
        return error * error
    if transform == "threshold_min":
        threshold = _term_threshold(term)
        return max(0.0, threshold - value_float)
    if transform == "threshold_max":
        threshold = _term_threshold(term)
        return max(0.0, value_float - threshold)
    raise ValueError(f"unsupported transform {transform!r}")


def _term_threshold(term: dict[str, Any]) -> float:
    if term.get("success_threshold") is not None:
        return _as_number(term["success_threshold"], f"{term['name']}.success_threshold")
    return _as_number(term.get("target"), f"{term['name']}.target")


def _success_from_criteria(
    criteria: Any,
    metrics: dict[str, Any],
    measured: dict[str, Any],
    warnings: list[str],
) -> bool:
    if not criteria:
        return True
    all_passed = True
    for criterion in criteria:
        metric_path = str(criterion["metric_path"])
        value = get_metric(metrics, metric_path)
        if value is _MISSING:
            warnings.append(f"missing success criterion metric: {metric_path}")
            measured[metric_path] = None
            all_passed = False
            continue
        measured[metric_path] = value
        try:
            passed = _compare(value, criterion["op"], criterion["threshold"])
        except ValueError as exc:
            warnings.append(f"invalid success criterion {criterion['name']!r}: {exc}")
            passed = False
        if not passed:
            all_passed = False
    return all_passed


def _compare(value: Any, op: str, threshold: Any) -> bool:
    if isinstance(value, bool) or isinstance(threshold, bool):
        if op == "==":
            return value == threshold
        if op == "!=":
            return value != threshold
        raise ValueError(f"boolean success criterion does not support op {op!r}")
    left = _as_number(value, "success criterion value")
    right = _as_number(threshold, "success criterion threshold")
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    raise ValueError(f"unsupported success criterion op {op!r}")


def _as_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} is not numeric")
    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{label} is not finite")
    return value_float


_MISSING = object()
