from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from ..io_utils import dump_json
from ..tool_library import GeneratorParameterOverrides
from .pipeline import (
    OptimizationConfig,
    OptimizationTaskSpec,
    optimize_prompt,
    optimize_prompts_batch,
)


def _cmd_optimize(args: argparse.Namespace) -> None:
    parameter_overrides = GeneratorParameterOverrides(
        sim_dt=args.sim_dt,
        render_every_n_steps=args.render_every_n_steps,
        render_res=(args.render_width, args.render_height),
        primitive_density=args.primitive_density,
        ground_friction=args.ground_friction,
    )
    config = OptimizationConfig(
        model=args.model,
        xml_model=args.xml_model,
        critic_model=args.critic_model,
        hosted_prompt_id=args.hosted_prompt_id,
        hosted_prompt_version=args.hosted_prompt_version,
        critic_hosted_prompt_id=args.critic_hosted_prompt_id,
        critic_hosted_prompt_version=args.critic_hosted_prompt_version,
        critic_prompt_variant=args.critic_prompt_variant,
        temperature=args.temperature,
        critic_temperature=args.critic_temperature,
        reasoning_effort=args.reasoning_effort,
        critic_reasoning_effort=args.critic_reasoning_effort,
        backend=args.backend,
        max_opt_rounds=args.max_opt_rounds,
        generator_max_rounds=args.max_attempts,
        xml_max_attempts=args.xml_max_attempts,
        timeout_sec=args.timeout_sec,
        assets_dir=str(args.assets_dir),
        mesh_assets_dir=str(args.mesh_assets_dir),
        generator_parameter_overrides=parameter_overrides,
        sample_every_sec=args.sample_every_sec,
        max_frames=args.max_frames,
        max_width=args.max_width,
        output_root=str(args.out_dir) if args.out_dir is not None else None,
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
    )
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
    parameter_overrides = GeneratorParameterOverrides(
        sim_dt=args.sim_dt,
        render_every_n_steps=args.render_every_n_steps,
        render_res=(args.render_width, args.render_height),
        primitive_density=args.primitive_density,
        ground_friction=args.ground_friction,
    )
    return OptimizationConfig(
        model=args.model,
        xml_model=args.xml_model,
        critic_model=args.critic_model,
        hosted_prompt_id=args.hosted_prompt_id,
        hosted_prompt_version=args.hosted_prompt_version,
        critic_hosted_prompt_id=args.critic_hosted_prompt_id,
        critic_hosted_prompt_version=args.critic_hosted_prompt_version,
        critic_prompt_variant=args.critic_prompt_variant,
        temperature=args.temperature,
        critic_temperature=args.critic_temperature,
        reasoning_effort=args.reasoning_effort,
        critic_reasoning_effort=args.critic_reasoning_effort,
        backend=args.backend,
        max_opt_rounds=args.max_opt_rounds,
        generator_max_rounds=args.max_attempts,
        xml_max_attempts=args.xml_max_attempts,
        timeout_sec=args.timeout_sec,
        assets_dir=str(args.assets_dir),
        mesh_assets_dir=str(args.mesh_assets_dir),
        generator_parameter_overrides=parameter_overrides,
        sample_every_sec=args.sample_every_sec,
        max_frames=args.max_frames,
        max_width=args.max_width,
        output_root=str(args.out_dir) if args.out_dir is not None else None,
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
    )


def _cmd_optimize_batch(args: argparse.Namespace) -> None:
    config = _build_config(args)
    task_specs = _parse_task_specs(args.task_spec, args.tasks_file)
    result = optimize_prompts_batch(
        task_specs=task_specs,
        config=config,
        max_parallel=args.max_parallel,
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
    parser_opt_batch.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Maximum number of tasks allowed to overlap in generator/critic stages.",
    )
    for parser_variant in common_parsers:
        parser_variant.add_argument("--model", type=str, default="gpt-5.4", help="Generator model.")
        parser_variant.add_argument("--xml-model", type=str, default=None, help="Optional XML generator model override.")
        parser_variant.add_argument("--critic-model", type=str, default=None, help="Optional critic model override.")
        parser_variant.add_argument("--hosted-prompt-id", type=str, default=None, help="Optional generator Hosted Prompt ID.")
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
        "--critic-prompt-variant",
        type=str,
        default="full",
        choices=("full", "compact"),
        help="Choose the original or compact critic prompt path.",
        )
        parser_variant.add_argument("--temperature", type=float, default=None, help="Optional generator sampling temperature.")
        parser_variant.add_argument(
        "--critic-temperature",
        type=float,
        default=None,
        help="Optional critic sampling temperature.",
        )
        parser_variant.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Optional generator reasoning effort.",
        )
        parser_variant.add_argument(
        "--critic-reasoning-effort",
        type=str,
        default=None,
        help="Optional critic reasoning effort.",
        )
        parser_variant.add_argument("--backend", type=str, default="cpu", choices=("cpu", "gpu"), help="Simulation backend.")
        parser_variant.add_argument("--max-opt-rounds", type=int, default=3, help="Maximum optimization rounds.")
        parser_variant.add_argument("--max-attempts", type=int, default=12, help="Generator max IR-agent rounds per round.")
        parser_variant.add_argument("--xml-max-attempts", type=int, default=4, help="Max XML generation attempts per round.")
        parser_variant.add_argument("--timeout-sec", type=float, default=600.0, help="OpenAI request timeout.")
        parser_variant.add_argument("--sim-dt", type=float, default=0.001, help="Fixed simulation dt passed to generator.")
        parser_variant.add_argument(
        "--render-every-n-steps",
        type=int,
        default=10,
        help="Fixed render cadence passed to generator.",
        )
        parser_variant.add_argument("--render-width", type=int, default=640, help="Fixed render width passed to generator.")
        parser_variant.add_argument(
        "--render-height",
        type=int,
        default=480,
        help="Fixed render height passed to generator.",
        )
        parser_variant.add_argument(
        "--primitive-density",
        type=float,
        default=1e3,
        help="Fixed primitive density passed to generator.",
        )
        parser_variant.add_argument(
        "--ground-friction",
        type=float,
        default=0.8,
        help="Fixed ground friction passed to generator.",
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
        parser_variant.add_argument("--sample-every-sec", type=float, default=0.5, help="Critic video sampling interval.")
        parser_variant.add_argument("--max-frames", type=int, default=24, help="Critic hard cap on sampled frames.")
        parser_variant.add_argument("--max-width", type=int, default=640, help="Critic max sampled frame width.")
        parser_variant.add_argument("--out-dir", type=Path, default=None, help="Optional optimization run directory.")
        parser_variant.add_argument("--out", type=Path, default=None, help="Optional summary JSON output path.")
        parser_variant.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="API key env var name.")
        parser_variant.add_argument("--base-url-env", type=str, default="OPENAI_BASE_URL", help="Base URL env var name.")
    parser_opt.set_defaults(func=_cmd_optimize)
    parser_opt_batch.set_defaults(func=_cmd_optimize_batch)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
