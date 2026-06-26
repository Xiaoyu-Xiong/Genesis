from __future__ import annotations

import argparse
from pathlib import Path

from code_agent.configs import CONFIGS
from code_agent.scores.physical.agent import PhysicalScoreRequest, run_physical_score
from code_agent.scores.physical.suite import score_physical_suite


def _cmd_score_case(args: argparse.Namespace) -> None:
    report = run_physical_score(
        PhysicalScoreRequest(
            run_dir=args.run_dir,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            code_root=args.code_root,
            case_id=args.case_id,
            output_path=args.output,
            model=args.model,
            timeout_sec=args.timeout_sec,
            force=args.force,
        )
    )
    report_path = args.output or args.run_dir.resolve() / "reports" / "physical_score_report.json"
    print(
        "SBAR-v1 "
        f"{report.get('scorer_status', 'unknown')}: "
        f"overall={_fmt(report.get('overall_score'))}, "
        f"scene={_fmt(report.get('scene_score'))}, "
        f"body={_fmt(report.get('body_score'))}, "
        f"action={_fmt(report.get('action_score'))}, "
        f"render={_fmt(report.get('render_score'))}. "
        f"Report: {report_path}"
    )


def _cmd_score_suite(args: argparse.Namespace) -> None:
    summary = score_physical_suite(
        suite_dir=args.suite_dir,
        tasks_file=args.tasks_file,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        max_cases=args.max_cases,
        model=args.model,
        timeout_sec=args.timeout_sec,
        force=args.force,
    )
    summary_path = (args.output_dir or args.suite_dir.resolve() / "physical_scores") / "summary.json"
    averages = summary.get("averages") if isinstance(summary.get("averages"), dict) else {}
    print(
        "SBAR-v1 suite "
        f"{summary.get('num_succeeded', 0)}/{summary.get('num_cases', 0)} scored. "
        f"avg_overall={_fmt(averages.get('overall_score'))}. "
        f"Summary: {summary_path}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m code_agent.scores.physical.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    case_parser = sub.add_parser("score-case", aliases=["case"], help="Score one generated simulation folder.")
    case_parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Folder containing code, artifacts, reports, and/or video.",
    )
    case_parser.add_argument("--code-root", type=Path, default=None, help="Optional separate generated code root.")
    case_parser.add_argument("--case-id", default=None)
    prompt_group = case_parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None)
    prompt_group.add_argument("--prompt-file", type=Path, default=None)
    case_parser.add_argument("--output", type=Path, default=None)
    case_parser.add_argument("--model", default=CONFIGS.codex.critic_model)
    case_parser.add_argument("--timeout-sec", type=float, default=CONFIGS.codex.critic_timeout_sec)
    case_parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if a physical score report already exists.",
    )
    case_parser.set_defaults(func=_cmd_score_case)

    suite_parser = sub.add_parser("score-suite", aliases=["suite"], help="Score every case in a suite in parallel.")
    suite_parser.add_argument("--suite-dir", type=Path, required=True)
    suite_parser.add_argument(
        "--tasks-file",
        type=Path,
        default=None,
        help="Optional suite tasks file for prompt mapping.",
    )
    suite_parser.add_argument("--output-dir", type=Path, default=None)
    suite_parser.add_argument("--max-workers", type=int, default=2)
    suite_parser.add_argument("--max-cases", type=int, default=None)
    suite_parser.add_argument("--model", default=CONFIGS.codex.critic_model)
    suite_parser.add_argument("--timeout-sec", type=float, default=CONFIGS.codex.critic_timeout_sec)
    suite_parser.add_argument("--force", action="store_true", help="Rerun cases even if score reports already exist.")
    suite_parser.set_defaults(func=_cmd_score_suite)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def _fmt(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.2f}"
    return "n/a"


if __name__ == "__main__":
    main()
