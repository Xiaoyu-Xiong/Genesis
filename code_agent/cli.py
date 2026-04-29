from __future__ import annotations

import argparse
from pathlib import Path

from code_agent.configs import CONFIGS
from code_agent.utils.suite import run_suite


def _cmd_run_suite(args: argparse.Namespace) -> None:
    summary = run_suite(
        tasks_file=args.tasks_file,
        out_dir=args.out_dir,
        backend=args.backend,
        max_cases=args.max_cases,
        timeout_sec=args.timeout_sec,
        render=args.render,
        repair_rounds=args.repair_rounds,
        steps=args.steps,
        duration_sec=args.duration_sec,
        render_fps=args.render_fps,
    )
    print(f"Done. {summary['num_passed']}/{summary['num_cases']} cases passed. Summary: {args.out_dir / 'summary.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m code_agent.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    run_suite_parser = sub.add_parser("run-suite", help="Run a code-agent prompt suite.")
    run_suite_parser.add_argument("--tasks-file", type=Path, required=True)
    run_suite_parser.add_argument("--out-dir", type=Path, required=True)
    run_suite_parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    run_suite_parser.add_argument("--cpu", action="store_const", const="cpu", dest="backend")
    run_suite_parser.add_argument("--gpu", action="store_const", const="gpu", dest="backend")
    run_suite_parser.add_argument("--max-cases", type=int, default=None)
    run_suite_parser.add_argument("--timeout-sec", type=float, default=CONFIGS.harness.execution_timeout_sec)
    run_suite_parser.add_argument("--render", action="store_true", default=True)
    run_suite_parser.add_argument("--no-render", action="store_false", dest="render")
    run_suite_parser.add_argument("--steps", type=int, default=None)
    run_suite_parser.add_argument("--duration-sec", type=float, default=None)
    run_suite_parser.add_argument("--render-fps", type=int, default=None)
    run_suite_parser.add_argument("--repair-rounds", type=int, default=1)
    run_suite_parser.set_defaults(func=_cmd_run_suite)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
