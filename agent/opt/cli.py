from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from ..configs import CONFIGS
from ..io_utils import dump_json
from .pipeline import (
    OptimizationConfig,
    OptimizationTaskSpec,
    optimize_prompt,
    optimize_prompts_batch,
)


def _cmd_optimize(args: argparse.Namespace) -> None:
    config = _build_config(args)
    result = optimize_prompt(task=args.task, config=config)
    if args.out is not None:
        dump_json(
            {
                "task": result.task,
                "status": result.status,
                "final_round_dir": result.final_round_dir,
                "final_verdict": result.final_verdict,
                "rounds": [asdict(item) for item in result.rounds],
            },
            args.out,
        )


def _parse_task_specs(task_specs: list[str] | None, tasks_file: Path | None) -> list[OptimizationTaskSpec]:
    parsed: list[OptimizationTaskSpec] = []

    for item in task_specs or []:
        if "=" not in item:
            raise ValueError("Each --task-spec must use the form CASE_ID=TASK.")
        case_id, task = item.split("=", 1)
        case_id = case_id.strip()
        task = task.strip()
        if not case_id or not task:
            raise ValueError("Each --task-spec must use the form CASE_ID=TASK.")
        parsed.append(OptimizationTaskSpec(case_id=case_id, task=task))

    if tasks_file is not None:
        for raw_line in tasks_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" not in line:
                raise ValueError("Each line in --tasks-file must use the form CASE_ID|TASK.")
            case_id, task = line.split("|", 1)
            case_id = case_id.strip()
            task = task.strip()
            if not case_id or not task:
                raise ValueError("Each line in --tasks-file must use the form CASE_ID|TASK.")
            parsed.append(OptimizationTaskSpec(case_id=case_id, task=task))

    if not parsed:
        raise ValueError("Provide at least one --task-spec or a non-empty --tasks-file.")

    seen: set[str] = set()
    for spec in parsed:
        if spec.case_id in seen:
            raise ValueError(f"Duplicate case_id `{spec.case_id}` in batch input.")
        seen.add(spec.case_id)
    return parsed


def _build_config(args: argparse.Namespace) -> OptimizationConfig:
    mesh_texture_enabled = CONFIGS.meshy_request.texture_enabled
    if args.mesh_texture_enabled is not None:
        mesh_texture_enabled = args.mesh_texture_enabled
    return OptimizationConfig(
        model=CONFIGS.optimization.model,
        xml_model=args.xml_model,
        critic_model=CONFIGS.optimization.critic_model or None,
        hosted_prompt_id=args.hosted_prompt_id,
        hosted_prompt_version=args.hosted_prompt_version,
        critic_hosted_prompt_id=args.critic_hosted_prompt_id,
        critic_hosted_prompt_version=args.critic_hosted_prompt_version,
        critic_prompt_variant=CONFIGS.optimization.critic_prompt_variant,
        temperature=args.temperature,
        critic_temperature=args.critic_temperature,
        reasoning_effort=CONFIGS.optimization.reasoning_effort,
        critic_reasoning_effort=(CONFIGS.optimization.critic_reasoning_effort or None),
        backend=CONFIGS.optimization.backend,
        max_opt_rounds=CONFIGS.optimization.max_opt_rounds,
        generator_max_rounds=CONFIGS.optimization.max_attempts,
        xml_max_attempts=CONFIGS.optimization.xml_max_attempts,
        timeout_sec=CONFIGS.optimization.timeout_sec,
        assets_dir=str(args.assets_dir),
        mesh_assets_dir=str(args.mesh_assets_dir),
        sample_every_sec=CONFIGS.optimization.sample_every_sec,
        max_frames=CONFIGS.optimization.max_frames,
        max_width=CONFIGS.optimization.max_width,
        output_root=str(args.out_dir) if args.out_dir is not None else None,
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        mesh_texture_enabled=mesh_texture_enabled,
    )


def _cmd_optimize_batch(args: argparse.Namespace) -> None:
    config = _build_config(args)
    task_specs = _parse_task_specs(args.task_spec, args.tasks_file)
    max_parallel = CONFIGS.optimization.max_parallel if args.max_parallel is None else args.max_parallel
    result = optimize_prompts_batch(
        task_specs=task_specs,
        config=config,
        max_parallel=max_parallel,
    )
    if args.out is not None:
        dump_json(
            {
                "status": result.status,
                "run_root": result.run_root,
                "items": [asdict(item) for item in result.items],
            },
            args.out,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iterative generator->critic optimization loop.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_opt = subparsers.add_parser("optimize", help="Generate, run, critique, and refine until pass.")
    parser_opt_batch = subparsers.add_parser(
        "optimize-batch",
        help="Optimize multiple tasks in parallel while serializing simulation execution.",
    )

    common_parsers = (parser_opt, parser_opt_batch)
    parser_opt.add_argument("--task", type=str, required=True, help="Original task prompt.")
    parser_opt_batch.add_argument(
        "--task-spec",
        action="append",
        default=None,
        help="Batch task in the form CASE_ID=TASK. Repeat for multiple tasks.",
    )
    parser_opt_batch.add_argument(
        "--tasks-file",
        type=Path,
        default=None,
        help="Optional file with one CASE_ID|TASK entry per line.",
    )
    for parser_variant in common_parsers:
        parser_variant.add_argument("--xml-model", type=str, default=None, help="Optional XML generator model override.")
        parser_variant.add_argument(
            "--hosted-prompt-id",
            type=str,
            default=None,
            help="Optional generator Hosted Prompt ID.",
        )
        parser_variant.add_argument(
            "--hosted-prompt-version",
            type=str,
            default=None,
            help="Optional generator Hosted Prompt version.",
        )
        parser_variant.add_argument(
            "--critic-hosted-prompt-id",
            type=str,
            default=None,
            help="Optional critic Hosted Prompt ID.",
        )
        parser_variant.add_argument(
            "--critic-hosted-prompt-version",
            type=str,
            default=None,
            help="Optional critic Hosted Prompt version.",
        )
        parser_variant.add_argument(
            "--temperature",
            type=float,
            default=None,
            help="Optional generator sampling temperature.",
        )
        parser_variant.add_argument(
            "--critic-temperature",
            type=float,
            default=None,
            help="Optional critic sampling temperature.",
        )
        parser_variant.add_argument(
            "--assets-dir",
            type=Path,
            default=Path("agent/generated_assets"),
            help="Directory for generated articulated XML assets.",
        )
        parser_variant.add_argument(
            "--mesh-assets-dir",
            type=Path,
            default=Path("agent/generated_meshes"),
            help="Directory for generated non-articulated mesh assets.",
        )
        parser_variant.add_argument("--out-dir", type=Path, default=None, help="Optional optimization run directory.")
        parser_variant.add_argument("--out", type=Path, default=None, help="Optional summary JSON output path.")
        parser_variant.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="API key env var name.")
        parser_variant.add_argument("--base-url-env", type=str, default="OPENAI_BASE_URL", help="Base URL env var name.")
        parser_variant.add_argument(
            "--mesh-texture-enabled",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable Meshy texture generation for non-articulated mesh assets in this optimization run.",
        )
    parser_opt_batch.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Override batch worker parallelism. Useful for memory-heavy mesh/texture suites.",
    )
    parser_opt.set_defaults(func=_cmd_optimize)
    parser_opt_batch.set_defaults(func=_cmd_optimize_batch)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
