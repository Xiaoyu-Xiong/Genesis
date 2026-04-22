from __future__ import annotations

from typing import Any


def usage_to_metrics(usage: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}

    input_tokens = _as_int(usage.get("input_tokens"))
    output_tokens = _as_int(usage.get("output_tokens"))
    total_tokens = _as_int(usage.get("total_tokens"))
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    cached_tokens = _as_int(usage.get("cached_tokens"))
    if cached_tokens is None and isinstance(input_details, dict):
        cached_tokens = _as_int(input_details.get("cached_tokens"))
    reasoning_tokens = _as_int(usage.get("reasoning_tokens"))
    if reasoning_tokens is None and isinstance(output_details, dict):
        reasoning_tokens = _as_int(output_details.get("reasoning_tokens"))

    metrics = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
    }
    return {key: value for key, value in metrics.items() if value is not None}


def aggregate_usage_metrics(entries: list[dict[str, Any] | None]) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    for entry in entries:
        metrics = usage_to_metrics(entry)
        for key in totals:
            totals[key] += int(metrics.get(key, 0) or 0)
    return totals


def _as_int(value: Any) -> int | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return int(value)
