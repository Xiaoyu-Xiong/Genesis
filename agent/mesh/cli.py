from __future__ import annotations

import argparse
from pathlib import Path

from ..defaults import DEFAULTS
from ..io_utils import dump_json
from .models import (
    MESH_FORMAT_VALUES,
    MESHY_AI_MODEL_VALUES,
    MESHY_ART_STYLE_VALUES,
    MESHY_SYMMETRY_VALUES,
    MESHY_TOPOLOGY_VALUES,
    MeshRepairConfig,
    MeshyApiConfig,
    MeshyGenerationConfig,
)
from .pipeline import default_mesh_output_dir, generate_meshy_mesh_from_text, parse_extra_payload
from .sanity import run_mesh_manifold_check


def _cmd_generate(args: argparse.Namespace) -> None:
    repair_defaults = DEFAULTS.mesh_repair
    output_dir = args.out_dir or default_mesh_output_dir(args.prompt, root=args.root_dir)
    api_config = MeshyApiConfig.from_env(
        api_key_env=args.api_key_env,
        base_url_env=args.base_url_env,
        timeout_sec=args.timeout_sec,
    )
    if args.base_url is not None:
        api_config.base_url = args.base_url.rstrip("/")
    if args.text_to_3d_path is not None:
        api_config.text_to_3d_path = args.text_to_3d_path

    generation_config = MeshyGenerationConfig(
        prompt=args.prompt,
        output_dir=output_dir,
        mesh_format=args.mesh_format,
        ai_model=args.ai_model,
        art_style=args.art_style,
        should_remesh=args.should_remesh,
        topology=args.topology,
        target_polycount=args.target_polycount,
        symmetry_mode=args.symmetry_mode,
        moderation=args.moderation,
        negative_prompt=args.negative_prompt,
        auto_size=args.auto_size,
        origin_at=args.origin_at,
        poll_interval_sec=args.poll_interval_sec,
        max_wait_sec=args.max_wait_sec,
        extra_payload=parse_extra_payload(args.extra_payload),
    )

    repair_config = None
    if not args.skip_postprocess:
        repair_config = MeshRepairConfig(
            component_count_face_cap=repair_defaults.component_count_face_cap,
            min_component_faces=args.min_component_faces,
            max_repair_attempts=args.max_repair_attempts,
            merge_vertices=repair_defaults.merge_vertices,
            merge_digits_vertex=repair_defaults.merge_digits_vertex,
            fix_normals=repair_defaults.fix_normals,
            process_validate=repair_defaults.process_validate,
            keep_largest_component=args.keep_largest_component,
            ftetwild_edge_length_fac=args.ftetwild_edge_length_fac,
            ftetwild_edge_length_abs=args.ftetwild_edge_length_abs,
            ftetwild_optimize=not args.ftetwild_no_optimize,
            ftetwild_simplify=not args.ftetwild_no_simplify,
            ftetwild_epsilon=args.ftetwild_epsilon,
            ftetwild_stop_energy=args.ftetwild_stop_energy,
            ftetwild_coarsen=args.ftetwild_coarsen,
            ftetwild_num_threads=args.ftetwild_num_threads,
            ftetwild_num_opt_iter=args.ftetwild_num_opt_iter,
            ftetwild_quiet=not args.ftetwild_verbose,
            ftetwild_disable_filtering=args.ftetwild_disable_filtering,
        )

    bundle = generate_meshy_mesh_from_text(
        prompt=args.prompt,
        api_config=api_config,
        generation_config=generation_config,
        repair_config=repair_config,
    )
    dump_json(bundle.to_dict(), args.out)


def _cmd_manifold_check(args: argparse.Namespace) -> None:
    result = run_mesh_manifold_check(args.mesh)
    dump_json(result.to_dict(), args.out)


def build_parser() -> argparse.ArgumentParser:
    request_defaults = DEFAULTS.meshy_request
    repair_defaults = DEFAULTS.mesh_repair
    parser = argparse.ArgumentParser(
        description="Standalone text-to-mesh utilities (Meshy preview generation, mesh repair, and manifold checks)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_generate = subparsers.add_parser(
        "generate",
        help="Generate a single mesh asset from a text prompt with the Meshy preview API.",
    )
    parser_generate.add_argument("--prompt", type=str, required=True, help="Text prompt for a single object mesh.")
    parser_generate.add_argument(
        "--mesh-format",
        type=str,
        default=request_defaults.mesh_format,
        choices=MESH_FORMAT_VALUES,
        help="Requested geometry format for the downloaded mesh asset.",
    )
    parser_generate.add_argument(
        "--ai-model",
        type=str,
        default=request_defaults.ai_model,
        choices=MESHY_AI_MODEL_VALUES,
        help="Meshy AI model for the preview stage.",
    )
    parser_generate.add_argument(
        "--art-style",
        type=str,
        default=request_defaults.art_style,
        choices=MESHY_ART_STYLE_VALUES,
        help="Meshy preview art style. `realistic` is the best default for simulation geometry.",
    )
    parser_generate.add_argument(
        "--should-remesh",
        action=argparse.BooleanOptionalAction,
        default=request_defaults.should_remesh,
        help="Enable Meshy remeshing so topology and target polycount are applied.",
    )
    parser_generate.add_argument(
        "--min-component-faces",
        type=int,
        default=repair_defaults.min_component_faces,
        help="Discard tiny connected components smaller than this face count before the final single-component pass.",
    )
    parser_generate.add_argument(
        "--max-repair-attempts",
        type=int,
        default=repair_defaults.max_repair_attempts,
        help="Maximum number of ftetwild repair attempts with progressively more robust parameter sets.",
    )
    parser_generate.add_argument(
        "--topology",
        type=str,
        default=request_defaults.topology,
        choices=MESHY_TOPOLOGY_VALUES,
        help="Requested topology when remeshing is enabled.",
    )
    parser_generate.add_argument(
        "--target-polycount",
        type=int,
        default=request_defaults.target_polycount,
        help="Optional Meshy target polycount when remeshing is enabled.",
    )
    parser_generate.add_argument(
        "--symmetry-mode",
        type=str,
        default=request_defaults.symmetry_mode,
        choices=MESHY_SYMMETRY_VALUES,
        help="Meshy symmetry mode for preview generation.",
    )
    parser_generate.add_argument(
        "--negative-prompt",
        type=str,
        default=request_defaults.negative_prompt,
        help="Optional Meshy negative prompt.",
    )
    parser_generate.add_argument(
        "--moderation",
        action=argparse.BooleanOptionalAction,
        default=request_defaults.moderation,
        help="Enable Meshy moderation screening.",
    )
    parser_generate.add_argument(
        "--auto-size",
        action=argparse.BooleanOptionalAction,
        default=request_defaults.auto_size,
        help="Let Meshy estimate real-world size automatically.",
    )
    parser_generate.add_argument(
        "--origin-at",
        type=str,
        default=request_defaults.origin_at,
        choices=("bottom", "center"),
        help="Origin placement when auto-size is enabled.",
    )
    parser_generate.add_argument(
        "--root-dir",
        type=Path,
        default=Path("agent/generated_meshes"),
        help="Root directory used when --out-dir is omitted.",
    )
    parser_generate.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Explicit output directory for prompt, raw API responses, downloaded asset, repair artifact, and manifold reports.",
    )
    parser_generate.add_argument(
        "--extra-payload",
        type=str,
        default=None,
        help="Optional extra JSON object, either inline or via a path to a JSON file.",
    )
    parser_generate.add_argument(
        "--poll-interval-sec",
        type=float,
        default=request_defaults.poll_interval_sec,
        help="Polling interval in seconds.",
    )
    parser_generate.add_argument(
        "--max-wait-sec",
        type=float,
        default=request_defaults.max_wait_sec,
        help="Maximum time to wait for the provider job to finish.",
    )
    parser_generate.add_argument(
        "--timeout-sec",
        type=float,
        default=request_defaults.timeout_sec,
        help="HTTP request timeout.",
    )
    parser_generate.add_argument(
        "--api-key-env",
        type=str,
        default="MESHY_API_KEY",
        help="Environment variable containing the Meshy API key.",
    )
    parser_generate.add_argument(
        "--base-url-env",
        type=str,
        default="MESHY_API_BASE_URL",
        help="Environment variable containing the Meshy API base URL.",
    )
    parser_generate.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Direct Meshy API base URL override. Takes precedence over --base-url-env.",
    )
    parser_generate.add_argument(
        "--text-to-3d-path",
        type=str,
        default=None,
        help="Direct Meshy text-to-3D path override. Defaults to /openapi/v2/text-to-3d.",
    )
    parser_generate.add_argument(
        "--skip-postprocess",
        action="store_true",
        help="Skip the repair-style post-processing step and only report the raw manifold check.",
    )
    parser_generate.add_argument(
        "--keep-largest-component",
        action=argparse.BooleanOptionalAction,
        default=repair_defaults.keep_largest_component,
        help="During repair, keep only the largest connected component if the mesh contains multiple parts. This is enabled by default in the current repair config; pass-through retained for explicitness.",
    )
    parser_generate.add_argument(
        "--ftetwild-edge-length-fac",
        type=float,
        default=repair_defaults.ftetwild_edge_length_fac,
        help="Relative target tet edge length for pytetwild.",
    )
    parser_generate.add_argument(
        "--ftetwild-edge-length-abs",
        type=float,
        default=repair_defaults.ftetwild_edge_length_abs,
        help="Absolute target tet edge length for pytetwild. Overrides edge-length-fac when set.",
    )
    parser_generate.add_argument(
        "--ftetwild-no-optimize",
        action="store_true",
        default=not repair_defaults.ftetwild_optimize,
        help="Disable fTetWild optimization passes.",
    )
    parser_generate.add_argument(
        "--ftetwild-no-simplify",
        action="store_true",
        default=not repair_defaults.ftetwild_simplify,
        help="Disable fTetWild internal surface simplification.",
    )
    parser_generate.add_argument(
        "--ftetwild-epsilon",
        type=float,
        default=repair_defaults.ftetwild_epsilon,
        help="Relative envelope size for pytetwild.",
    )
    parser_generate.add_argument(
        "--ftetwild-stop-energy",
        type=float,
        default=repair_defaults.ftetwild_stop_energy,
        help="Optimization stop energy for pytetwild.",
    )
    parser_generate.add_argument(
        "--ftetwild-coarsen",
        action=argparse.BooleanOptionalAction,
        default=repair_defaults.ftetwild_coarsen,
        help="Allow pytetwild to coarsen as much as possible while maintaining quality.",
    )
    parser_generate.add_argument(
        "--ftetwild-num-threads",
        type=int,
        default=repair_defaults.ftetwild_num_threads,
        help="Thread count for pytetwild. 0 uses all available cores.",
    )
    parser_generate.add_argument(
        "--ftetwild-num-opt-iter",
        type=int,
        default=repair_defaults.ftetwild_num_opt_iter,
        help="Maximum optimization iterations for pytetwild.",
    )
    parser_generate.add_argument(
        "--ftetwild-disable-filtering",
        action=argparse.BooleanOptionalAction,
        default=repair_defaults.ftetwild_disable_filtering,
        help="Disable pytetwild filtering and keep the background mesh.",
    )
    parser_generate.add_argument(
        "--ftetwild-verbose",
        action=argparse.BooleanOptionalAction,
        default=not repair_defaults.ftetwild_quiet,
        help="Enable pytetwild console output.",
    )
    parser_generate.add_argument("--out", type=Path, default=None, help="Optional summary JSON output path.")
    parser_generate.set_defaults(func=_cmd_generate)

    parser_check = subparsers.add_parser(
        "manifold-check",
        help="Check whether an existing mesh is watertight and winding-consistent.",
    )
    parser_check.add_argument("--mesh", type=Path, required=True, help="Existing mesh file path.")
    parser_check.add_argument("--out", type=Path, default=None, help="Optional summary JSON output path.")
    parser_check.set_defaults(func=_cmd_manifold_check)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
