from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from ..io_utils import dump_json
from .agents import generate_ir_two_agent
from .client import OpenAIResponsesClient, REASONING_EFFORT_VALUES


def _cmd_generate(args: argparse.Namespace) -> None:
    client = OpenAIResponsesClient.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        timeout_sec=args.timeout_sec,
    )

    result = generate_ir_two_agent(
        task=args.task,
        model=args.model,
        client=client,
        xml_model=args.xml_model,
        max_rounds=args.max_attempts,
        xml_max_attempts=args.xml_max_attempts,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        normalize=not args.no_normalize,
        assets_dir=args.assets_dir,
        mesh_assets_dir=args.mesh_assets_dir,
        force_primitive_mode=args.primitive_only,
        hosted_prompt_id=args.hosted_prompt_id,
        hosted_prompt_version=args.hosted_prompt_version,
        mesh_texture_enabled=args.mesh_texture_enabled,
    )
    dump_json(result.ir_json, args.out)

    if args.log_out is not None:
        ir_logs = [asdict(log) for log in result.ir_result.logs]
        dump_json(
            {
                "model": result.model,
                "reasoning_effort": args.reasoning_effort,
                "mode": result.mode,
                "articulated_requested": result.articulated_requested,
                "ir_rounds": result.ir_result.rounds,
                "xml_results_by_body": {
                    body_name: {
                        "xml_path": xml_result.xml_path,
                        "attempts": xml_result.attempts,
                    }
                    for body_name, xml_result in sorted(result.xml_results_by_body.items())
                },
                "mesh_results_by_body": {
                    body_name: {
                        "mesh_path": mesh_result.mesh_path,
                        "raw_manifold_ok": mesh_result.raw_manifold_ok,
                        "repaired_manifold_ok": mesh_result.repaired_manifold_ok,
                        "texture_requested": mesh_result.texture_requested,
                        "texture_succeeded": mesh_result.texture_succeeded,
                        "textured_mesh_path": mesh_result.textured_mesh_path,
                        "base_color_path": mesh_result.base_color_path,
                    }
                    for body_name, mesh_result in sorted(result.mesh_results_by_body.items())
                },
                "ir_logs": ir_logs,
                "xml_logs_by_body": {
                    body_name: [asdict(log) for log in xml_result.logs]
                    for body_name, xml_result in sorted(result.xml_results_by_body.items())
                },
                "mesh_logs_by_body": {
                    body_name: [asdict(log) for log in mesh_result.logs]
                    for body_name, mesh_result in sorted(result.mesh_results_by_body.items())
                },
            },
            args.log_out,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate rigid-scene IR with agentic LLM workflow (multiple primitive and articulated bodies)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_generate = subparsers.add_parser(
        "generate",
        help="Generate IR JSON from natural language (primitive or articulated).",
    )
    parser_generate.add_argument("--task", type=str, required=True, help="Natural-language simulation request.")
    parser_generate.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model name.")
    parser_generate.add_argument(
        "--xml-model",
        type=str,
        default=None,
        help="Optional model override for articulated XML agent (defaults to --model).",
    )
    parser_generate.add_argument(
        "--max-attempts",
        type=int,
        default=12,
        help="Max IR-agent rounds (tool-calling + repair iterations).",
    )
    parser_generate.add_argument(
        "--xml-max-attempts",
        type=int,
        default=4,
        help="Max attempts for XML agent when articulated prompt is detected.",
    )
    parser_generate.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. Omitted by default.",
    )
    parser_generate.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=REASONING_EFFORT_VALUES,
        help="Optional Responses API reasoning effort.",
    )
    parser_generate.add_argument(
        "--hosted-prompt-id",
        type=str,
        default=None,
        help="Optional Hosted Prompt ID for the fixed generator instructions.",
    )
    parser_generate.add_argument(
        "--hosted-prompt-version",
        type=str,
        default=None,
        help="Optional Hosted Prompt version.",
    )
    parser_generate.add_argument("--timeout-sec", type=float, default=120.0, help="OpenAI request timeout in seconds.")
    parser_generate.add_argument("--out", type=Path, default=None, help="Output IR JSON path. Defaults to stdout.")
    parser_generate.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("agent/generated_assets"),
        help="Directory for generated articulated XML assets.",
    )
    parser_generate.add_argument(
        "--mesh-assets-dir",
        type=Path,
        default=Path("agent/generated_meshes"),
        help="Directory for generated non-articulated mesh assets.",
    )
    parser_generate.add_argument(
        "--primitive-only",
        action="store_true",
        help="Force primitive-only mode in unified generator (no articulated XML route).",
    )
    parser_generate.add_argument(
        "--mesh-texture-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Meshy texture generation for non-articulated mesh assets in this generation run.",
    )
    parser_generate.add_argument(
        "--log-out",
        type=Path,
        default=None,
        help="Optional debug log path with per-round IR-agent and XML-agent interactions.",
    )
    parser_generate.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="API key env var name.")
    parser_generate.add_argument("--base-url-env", type=str, default="OPENAI_BASE_URL", help="Base URL env var name.")
    parser_generate.add_argument("--no-normalize", action="store_true", help="Disable quaternion normalization.")
    parser_generate.set_defaults(func=_cmd_generate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
