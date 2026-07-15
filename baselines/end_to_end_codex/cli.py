from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from code_agent.utils.codex import CodexAuthFreshnessError, ensure_configured_codex_accounts_fresh

from baselines.end_to_end_codex.configs import (
    DEFAULT_BACKEND,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    DEFAULT_CODEX_SERVICE_TIER,
    DEFAULT_CODEX_TIMEOUT_SEC,
    DEFAULT_EXECUTION_TIMEOUT_SEC,
    DEFAULT_MAX_PARALLEL_CASES,
)
from baselines.end_to_end_codex.runner import EndToEndBaselineConfig, run_end_to_end_suite


def _cmd_run_suite(args: argparse.Namespace) -> None:
    try:
        ensure_configured_codex_accounts_fresh()
    except CodexAuthFreshnessError as exc:
        raise SystemExit(str(exc)) from exc

    summary = run_end_to_end_suite(
        EndToEndBaselineConfig(
            tasks_file=args.tasks_file.resolve(),
            out_dir=args.out_dir.resolve(),
            backend=args.backend,
            max_cases=args.max_cases,
            max_parallel_cases=args.max_parallel_cases,
            execution_timeout_sec=args.execution_timeout_sec,
            codex_timeout_sec=args.codex_timeout_sec,
            render=args.render,
            steps=args.steps,
            duration_sec=args.duration_sec,
            render_fps=args.render_fps,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            service_tier=args.service_tier,
        )
    )
    summary_path = args.out_dir.resolve() / "summary.json"
    print(f"Done. {summary['num_passed']}/{summary['num_cases']} cases executed successfully. Summary: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m baselines.end_to_end_codex.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    run_suite_parser = sub.add_parser("run-suite", help="Run the single-Codex-agent end-to-end baseline.")
    run_suite_parser.add_argument("--tasks-file", type=Path, required=True)
    run_suite_parser.add_argument("--out-dir", type=Path, required=True)
    run_suite_parser.add_argument("--backend", choices=("cpu", "gpu"), default=DEFAULT_BACKEND)
    run_suite_parser.add_argument("--cpu", action="store_const", const="cpu", dest="backend")
    run_suite_parser.add_argument("--gpu", action="store_const", const="gpu", dest="backend")
    run_suite_parser.add_argument("--max-cases", type=int, default=None)
    run_suite_parser.add_argument("--max-parallel-cases", type=int, default=DEFAULT_MAX_PARALLEL_CASES)
    run_suite_parser.add_argument(
        "--execution-timeout-sec",
        type=float,
        default=DEFAULT_EXECUTION_TIMEOUT_SEC,
    )
    run_suite_parser.add_argument("--codex-timeout-sec", type=float, default=DEFAULT_CODEX_TIMEOUT_SEC)
    run_suite_parser.add_argument("--render", action="store_true", default=True)
    run_suite_parser.add_argument("--no-render", action="store_false", dest="render")
    run_suite_parser.add_argument("--steps", type=int, default=None)
    run_suite_parser.add_argument("--duration-sec", type=float, default=None)
    run_suite_parser.add_argument("--render-fps", type=int, default=None)
    run_suite_parser.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    run_suite_parser.add_argument("--reasoning-effort", default=DEFAULT_CODEX_REASONING_EFFORT)
    run_suite_parser.add_argument("--service-tier", choices=("fast", "standard"), default=DEFAULT_CODEX_SERVICE_TIER)
    run_suite_parser.set_defaults(func=_cmd_run_suite)
    return parser


def main(*, hard_exit_after_success: bool = False) -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    if hard_exit_after_success and args.command == "run-suite":
        # Match the main code_agent CLI batch path: long Genesis runs can finish
        # successfully and then abort during native teardown. The summary has
        # already been written, so bypass late destructors for supervised suites.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main(hard_exit_after_success=True)
