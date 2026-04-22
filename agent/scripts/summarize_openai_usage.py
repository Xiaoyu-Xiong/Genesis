from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _metrics(entry: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(entry, dict):
        return {}
    out: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens"):
        value = entry.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            out[key] = int(value)
    return out


def _add_row(rows: list[dict[str, Any]], *, case_id: str, round_name: str, component: str, metrics: dict[str, int]) -> None:
    rows.append(
        {
            "case_id": case_id,
            "round": round_name,
            "component": component,
            "input_tokens": metrics.get("input_tokens", 0),
            "output_tokens": metrics.get("output_tokens", 0),
            "total_tokens": metrics.get("total_tokens", 0),
            "cached_tokens": metrics.get("cached_tokens", 0),
            "reasoning_tokens": metrics.get("reasoning_tokens", 0),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"run_root": str(args.run_root), "cases": {}, "totals": {}}

    for case_dir in sorted(path for path in args.run_root.iterdir() if path.is_dir()):
        case_rows: list[dict[str, Any]] = []
        for round_dir in sorted(path for path in case_dir.iterdir() if path.is_dir() and path.name.startswith("round_")):
            usage_payload = _load_json(round_dir / "llm_usage.json")
            if usage_payload is None:
                continue
            generator = usage_payload.get("generator") if isinstance(usage_payload, dict) else None
            critic = usage_payload.get("critic") if isinstance(usage_payload, dict) else None

            if isinstance(generator, dict):
                before_len = len(rows)
                _add_row(
                    rows,
                    case_id=case_dir.name,
                    round_name=round_dir.name,
                    component="generator_ir",
                    metrics=_metrics(generator.get("generator_ir") if isinstance(generator.get("generator_ir"), dict) else {}),
                )
                _add_row(
                    rows,
                    case_id=case_dir.name,
                    round_name=round_dir.name,
                    component="generator_xml",
                    metrics=_metrics(generator.get("generator_xml") if isinstance(generator.get("generator_xml"), dict) else {}),
                )
                case_rows.extend(rows[before_len:])
            if isinstance(critic, dict):
                before_len = len(rows)
                _add_row(
                    rows,
                    case_id=case_dir.name,
                    round_name=round_dir.name,
                    component="critic_total",
                    metrics=_metrics(critic.get("critic_total") if isinstance(critic.get("critic_total"), dict) else {}),
                )
                stages = critic.get("stages")
                if isinstance(stages, list):
                    for stage in stages:
                        if not isinstance(stage, dict):
                            continue
                        stage_name = stage.get("stage")
                        if not isinstance(stage_name, str):
                            continue
                        _add_row(
                            rows,
                            case_id=case_dir.name,
                            round_name=round_dir.name,
                            component=stage_name,
                            metrics=_metrics(stage.get("usage") if isinstance(stage.get("usage"), dict) else {}),
                        )
                case_rows.extend(rows[before_len:])
        if case_rows:
            summary["cases"][case_dir.name] = case_rows

    totals = {
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "cached_tokens": sum(row["cached_tokens"] for row in rows),
        "reasoning_tokens": sum(row["reasoning_tokens"] for row in rows),
    }
    summary["rows"] = rows
    summary["totals"] = totals

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "case_id",
                "round",
                "component",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "cached_tokens",
                "reasoning_tokens",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(args.out_json)
    print(args.out_tsv)


if __name__ == "__main__":
    main()
