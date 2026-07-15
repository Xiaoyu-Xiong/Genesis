from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

from code_agent.configs import CONFIGS
from code_agent.context.genesis import build_genesis_context_pack
from code_agent.dataset.store import DEFAULT_DATA_ROOT, DatasetStore
from code_agent.opt.runner import RunOptConfig, run_optimization
from code_agent.scores.physical.agent import PhysicalScoreRequest, run_physical_score
from code_agent.scores.physical.suite import score_physical_suite
from code_agent.utils.codex import CodexAuthFreshnessError, ensure_configured_codex_accounts_fresh
from code_agent.utils.suite import run_suite


def _cmd_run_suite(args: argparse.Namespace) -> None:
    _ensure_codex_auth_fresh()
    out_dir = args.out_dir.resolve()
    summary_path = out_dir / "summary.json"
    summary_callback = _dataset_train_summary_callback(args, out_dir=out_dir, summary_path=summary_path)
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
        opt_enabled=args.opt_enabled,
        summary_callback=summary_callback,
    )
    print(f"Done. {summary['num_passed']}/{summary['num_cases']} cases passed. Summary: {summary_path}")


def _dataset_train_summary_callback(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    summary_path: Path,
):
    if args.no_dataset_train_sync or not _looks_like_dataset_train_batch(args.tasks_file, out_dir):
        return None

    store = DatasetStore(args.dataset_data_root)

    def sync(summary: dict[str, Any]) -> None:
        result = store.mark_train_results_from_suite(summary, summary_path=summary_path)
        if result.get("changed"):
            print(
                "[dataset] marked train passes: "
                f"{result['changed']} changed; run_id={result['run_id']}",
                flush=True,
            )

    return sync


def _looks_like_dataset_train_batch(tasks_file: Path, out_dir: Path) -> bool:
    return out_dir.name.startswith("dataset_train_batch_") or tasks_file.resolve().parent.name.startswith(
        "dataset_train_batch_"
    )


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


def _cmd_score_physical_case(args: argparse.Namespace) -> None:
    _ensure_codex_auth_fresh()
    report = run_physical_score(
        PhysicalScoreRequest(
            run_dir=args.run_dir.resolve(),
            prompt=args.prompt,
            prompt_file=args.prompt_file.resolve() if args.prompt_file else None,
            code_root=args.code_root.resolve() if args.code_root else None,
            case_id=args.case_id,
            output_path=args.output.resolve() if args.output else None,
            model=args.model,
            timeout_sec=args.timeout_sec,
            force=args.force,
        )
    )
    report_path = (
        args.output.resolve()
        if args.output
        else args.run_dir.resolve() / "reports" / "physical_score_report.json"
    )
    print(
        "SBAR-v1 "
        f"{report.get('scorer_status', 'unknown')}: "
        f"overall={_format_score(report.get('overall_score'))}, "
        f"scene={_format_score(report.get('scene_score'))}, "
        f"body={_format_score(report.get('body_score'))}, "
        f"action={_format_score(report.get('action_score'))}, "
        f"render={_format_score(report.get('render_score'))}. "
        f"Report: {report_path}"
    )


def _cmd_score_physical_suite(args: argparse.Namespace) -> None:
    _ensure_codex_auth_fresh()
    summary = score_physical_suite(
        suite_dir=args.suite_dir.resolve(),
        tasks_file=args.tasks_file.resolve() if args.tasks_file else None,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        max_workers=args.max_workers,
        max_cases=args.max_cases,
        model=args.model,
        timeout_sec=args.timeout_sec,
        force=args.force,
    )
    summary_path = (
        args.output_dir.resolve()
        if args.output_dir
        else args.suite_dir.resolve() / "physical_scores"
    ) / "summary.json"
    averages = summary.get("averages") if isinstance(summary.get("averages"), dict) else {}
    print(
        "SBAR-v1 suite "
        f"{summary.get('num_succeeded', 0)}/{summary.get('num_cases', 0)} scored. "
        f"avg_overall={_format_score(averages.get('overall_score'))}. "
        f"Summary: {summary_path}"
    )


def _format_score(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.2f}"
    return "n/a"


def _ensure_codex_auth_fresh() -> None:
    try:
        ensure_configured_codex_accounts_fresh()
    except CodexAuthFreshnessError as exc:
        raise SystemExit(str(exc)) from exc


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
    run_suite_parser.add_argument(
        "--dataset-data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset manifest root to sync pass-only train markers for dataset_train_batch_* suites.",
    )
    run_suite_parser.add_argument(
        "--no-dataset-train-sync",
        action="store_true",
        help="Disable automatic pass-only dataset trained marker sync for dataset_train_batch_* suites.",
    )
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

    physical_case_parser = sub.add_parser("score-physical-case", help="Run SBAR-v1 on one generated case folder.")
    physical_case_parser.add_argument("--run-dir", type=Path, required=True)
    physical_case_parser.add_argument("--code-root", type=Path, default=None)
    physical_case_parser.add_argument("--case-id", default=None)
    prompt_group = physical_case_parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=None)
    prompt_group.add_argument("--prompt-file", type=Path, default=None)
    physical_case_parser.add_argument("--output", type=Path, default=None)
    physical_case_parser.add_argument("--model", default=CONFIGS.codex.critic_model)
    physical_case_parser.add_argument("--timeout-sec", type=float, default=CONFIGS.codex.critic_timeout_sec)
    physical_case_parser.add_argument("--force", action="store_true")
    physical_case_parser.set_defaults(func=_cmd_score_physical_case)

    physical_suite_parser = sub.add_parser("score-physical-suite", help="Run SBAR-v1 over a suite in parallel.")
    physical_suite_parser.add_argument("--suite-dir", type=Path, required=True)
    physical_suite_parser.add_argument("--tasks-file", type=Path, default=None)
    physical_suite_parser.add_argument("--output-dir", type=Path, default=None)
    physical_suite_parser.add_argument("--max-workers", type=int, default=2)
    physical_suite_parser.add_argument("--max-cases", type=int, default=None)
    physical_suite_parser.add_argument("--model", default=CONFIGS.codex.critic_model)
    physical_suite_parser.add_argument("--timeout-sec", type=float, default=CONFIGS.codex.critic_timeout_sec)
    physical_suite_parser.add_argument("--force", action="store_true")
    physical_suite_parser.set_defaults(func=_cmd_score_physical_suite)
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
