from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from agent.usage import usage_to_metrics

TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens", "cached_tokens", "reasoning_tokens")
GENERATOR_COMPONENTS = ("generator_ir", "generator_xml")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _add_row(rows: list[dict[str, Any]], *, case_id: str, round_name: str, component: str, metrics: dict[str, int]) -> None:
    rows.append(
        {
            "case_id": case_id,
            "round": round_name,
            "component": component,
            **{key: metrics.get(key, 0) for key in TOKEN_KEYS},
        }
    )


def _append_component_row(
    rows: list[dict[str, Any]],
    *,
    case_rows: list[dict[str, Any]],
    case_id: str,
    round_name: str,
    component: str,
    usage: dict[str, Any] | None,
) -> None:
    before_len = len(rows)
    _add_row(
        rows,
        case_id=case_id,
        round_name=round_name,
        component=component,
        metrics=usage_to_metrics(usage if isinstance(usage, dict) else {}),
    )
    case_rows.extend(rows[before_len:])


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
                for component in GENERATOR_COMPONENTS:
                    _append_component_row(
                        rows,
                        case_rows=case_rows,
                        case_id=case_dir.name,
                        round_name=round_dir.name,
                        component=component,
                        usage=generator.get(component),
                    )
            if isinstance(critic, dict):
                _append_component_row(
                    rows,
                    case_rows=case_rows,
                    case_id=case_dir.name,
                    round_name=round_dir.name,
                    component="critic_total",
                    usage=critic.get("critic_total"),
                )
                stages = critic.get("stages")
                if isinstance(stages, list):
                    for stage in stages:
                        if not isinstance(stage, dict):
                            continue
                        stage_name = stage.get("stage")
                        if not isinstance(stage_name, str):
                            continue
                        _append_component_row(
                            rows,
                            case_rows=case_rows,
                            case_id=case_dir.name,
                            round_name=round_dir.name,
                            component=stage_name,
                            usage=stage.get("usage"),
                        )
        if case_rows:
            summary["cases"][case_dir.name] = case_rows

    totals = {key: sum(int(row.get(key, 0) or 0) for row in rows) for key in TOKEN_KEYS}
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
                *TOKEN_KEYS,
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(args.out_json)
    print(args.out_tsv)


if __name__ == "__main__":
    main()
