from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from code_agent.configs import CONFIGS
from code_agent.context.genesis import build_genesis_context_pack
from code_agent.opt.runner import RunOptConfig, run_optimization
from code_agent.utils.suite import run_suite


def _cmd_run_suite(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    summary = run_suite(
        tasks_file=args.tasks_file.resolve(),
        out_dir=out_dir,
        backend=args.backend,
        max_cases=args.max_cases,
        timeout_sec=args.timeout_sec,
        render=args.render,
        repair_rounds=args.repair_rounds,
        max_parallel_cases=args.max_parallel_cases,
        steps=args.steps,
        duration_sec=args.duration_sec,
        render_fps=args.render_fps,
        deformable_enabled=args.deformable_enabled,
        ipc_enabled=args.ipc_enabled,
        opt_enabled=args.opt_enabled,
    )
    summary_path = out_dir / "summary.json"
    print(f"Done. {summary['num_passed']}/{summary['num_cases']} cases passed. Summary: {summary_path}")


def _cmd_build_genesis_context(args: argparse.Namespace) -> None:
    pack = build_genesis_context_pack(args.out_dir, refresh=args.refresh)
    print(f"Genesis context: {pack.markdown_path}")
    print(f"Official docs cache: {pack.docs_dir}")


def _cmd_run_opt(args: argparse.Namespace) -> None:
    report = run_optimization(
        RunOptConfig(
            case_dir=args.case_dir.resolve(),
            target_spec_path=args.target_spec.resolve() if args.target_spec else None,
            opt_space_path=args.opt_space.resolve() if args.opt_space else None,
            default_params_path=args.default_params.resolve() if args.default_params else None,
            backend=args.backend,
            max_trials=args.max_trials,
            population_size=args.population_size,
            seed=args.seed,
            timeout_sec=args.timeout_sec,
            steps=args.steps,
            duration_sec=args.duration_sec,
            render_fps=args.render_fps,
            render_best=args.render_best,
            main_file=args.main_file,
        )
    )
    report_path = args.case_dir.resolve() / "reports" / "opt_report.json"
    best_score = report.get("best_score")
    best_score_text = "none" if best_score is None else f"{float(best_score):.6g}"
    print(f"Opt {report['status']}. Best score: {best_score_text}. Report: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m code_agent.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    run_suite_parser = sub.add_parser("run-suite", help="Run a code-agent prompt suite.")
    run_suite_parser.add_argument("--tasks-file", type=Path, required=True)
    run_suite_parser.add_argument("--out-dir", type=Path, required=True)
    run_suite_parser.add_argument("--backend", choices=("cpu", "gpu"), default=CONFIGS.harness.default_backend)
    run_suite_parser.add_argument("--cpu", action="store_const", const="cpu", dest="backend")
    run_suite_parser.add_argument("--gpu", action="store_const", const="gpu", dest="backend")
    run_suite_parser.add_argument("--max-cases", type=int, default=None)
    run_suite_parser.add_argument("--max-parallel-cases", type=int, default=CONFIGS.harness.max_parallel_cases)
    run_suite_parser.add_argument("--timeout-sec", type=float, default=CONFIGS.harness.execution_timeout_sec)
    run_suite_parser.add_argument("--render", action="store_true", default=True)
    run_suite_parser.add_argument("--no-render", action="store_false", dest="render")
    run_suite_parser.add_argument("--steps", type=int, default=None)
    run_suite_parser.add_argument("--duration-sec", type=float, default=None)
    run_suite_parser.add_argument("--render-fps", type=int, default=None)
    run_suite_parser.add_argument("--repair-rounds", type=int, default=CONFIGS.harness.max_repair_rounds)
    deformable_group = run_suite_parser.add_mutually_exclusive_group()
    deformable_group.add_argument("--enable-deformable", action="store_true", dest="deformable_enabled", default=None)
    deformable_group.add_argument("--disable-deformable", action="store_false", dest="deformable_enabled")
    ipc_group = run_suite_parser.add_mutually_exclusive_group()
    ipc_group.add_argument("--enable-ipc", action="store_true", dest="ipc_enabled", default=None)
    ipc_group.add_argument("--disable-ipc", action="store_false", dest="ipc_enabled")
    opt_group = run_suite_parser.add_mutually_exclusive_group()
    opt_group.add_argument("--enable-opt", action="store_true", dest="opt_enabled", default=None)
    opt_group.add_argument("--disable-opt", action="store_false", dest="opt_enabled")
    run_suite_parser.set_defaults(func=_cmd_run_suite)

    context_parser = sub.add_parser("build-genesis-context", help="Fetch/cache Genesis docs context for subagents.")
    context_parser.add_argument("--out-dir", type=Path, required=True)
    context_parser.add_argument("--refresh", action="store_true")
    context_parser.set_defaults(func=_cmd_build_genesis_context)

    run_opt_parser = sub.add_parser("run-opt", help="Run the optimization agent on a generated case workspace.")
    run_opt_parser.add_argument("--case-dir", type=Path, required=True)
    run_opt_parser.add_argument("--target-spec", type=Path, default=None)
    run_opt_parser.add_argument("--opt-space", type=Path, default=None)
    run_opt_parser.add_argument("--default-params", type=Path, default=None)
    run_opt_parser.add_argument("--main-file", default=CONFIGS.opt.runner_main_file)
    run_opt_parser.add_argument("--backend", choices=("cpu", "gpu"), default=None)
    run_opt_parser.add_argument("--cpu", action="store_const", const="cpu", dest="backend")
    run_opt_parser.add_argument("--gpu", action="store_const", const="gpu", dest="backend")
    run_opt_parser.add_argument("--max-trials", type=int, default=None)
    run_opt_parser.add_argument("--population-size", type=int, default=None)
    run_opt_parser.add_argument("--seed", type=int, default=None)
    run_opt_parser.add_argument("--timeout-sec", type=float, default=None)
    run_opt_parser.add_argument("--steps", type=int, default=None)
    run_opt_parser.add_argument("--duration-sec", type=float, default=None)
    run_opt_parser.add_argument("--render-fps", type=int, default=None)
    render_best_group = run_opt_parser.add_mutually_exclusive_group()
    render_best_group.add_argument("--render-best", action="store_true", dest="render_best", default=None)
    render_best_group.add_argument("--no-render-best", action="store_false", dest="render_best")
    run_opt_parser.set_defaults(func=_cmd_run_opt)
    return parser


def main(*, hard_exit_after_success: bool = False) -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    if hard_exit_after_success and args.command == "run-suite":
        # Some Genesis/native dependencies can abort during interpreter teardown
        # after the suite summary has already been written. For the CLI batch
        # path, flush user-visible output and bypass late native destructors so
        # supervisors see the real suite completion status.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main(hard_exit_after_success=True)
