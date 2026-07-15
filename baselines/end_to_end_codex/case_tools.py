from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from code_agent.assets.mesh.episode import generate_mesh_assets_for_episode
from code_agent.assets.mesh.request_adapter import select_mesh_requests
from code_agent.io_utils import dump_json, load_json_object
from code_agent.utils.adaptive_ipc import apply_adaptive_contact_d_hat
from code_agent.utils.execution import run_generated_simulation

from baselines.end_to_end_codex.configs import (
    DEFAULT_BACKEND,
    DEFAULT_EXECUTION_TIMEOUT_SEC,
    deformable_config_from_planner_output,
)
from baselines.end_to_end_codex.timing import BaselineTimingPlan, resolve_baseline_timing


TOOL_HISTORY_NAME = "baseline_agent_tool_history.jsonl"
REPO_ROOT = Path(__file__).resolve().parents[2]


def prepare_contracts(
    *,
    case_dir: Path,
    steps: int | None = None,
    duration_sec: float | None = None,
    render_fps: int | None = None,
) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    reports_dir = case_dir / "reports"
    contracts_dir = case_dir / "contracts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir.mkdir(parents=True, exist_ok=True)

    planner_output_path = contracts_dir / "planner_output.json"
    planner_output = load_json_object(planner_output_path)
    if planner_output is None:
        report = _tool_report(
            "prepare_contracts",
            ok=False,
            status="planner_output_missing_or_invalid",
            case_dir=case_dir,
            errors=[f"Missing or invalid JSON object: {planner_output_path}"],
        )
        return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_contract_report.json")

    physics_mode = _physics_mode_from_planner_output(planner_output)
    deformable_config = deformable_config_from_planner_output(planner_output)
    dump_json(deformable_config, contracts_dir / "deformable_config.json")

    try:
        timing = resolve_baseline_timing(
            planner_output=planner_output,
            steps=steps,
            duration_sec=duration_sec,
            render_fps=render_fps,
        )
    except Exception as exc:
        report = _tool_report(
            "prepare_contracts",
            ok=False,
            status="timing_resolution_failed",
            case_dir=case_dir,
            errors=[f"{type(exc).__name__}: {exc}"],
            physics_mode=physics_mode,
            deformable_config_path=str(contracts_dir / "deformable_config.json"),
        )
        return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_contract_report.json")

    dump_json(timing.to_dict(), contracts_dir / "timing.json")
    report = _tool_report(
        "prepare_contracts",
        ok=True,
        status="contracts_ready",
        case_dir=case_dir,
        physics_mode=physics_mode,
        planner_output_path=str(planner_output_path),
        deformable_config_path=str(contracts_dir / "deformable_config.json"),
        timing_path=str(contracts_dir / "timing.json"),
        timing=timing.to_dict(),
    )
    return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_contract_report.json")


def generate_mesh_assets(
    *,
    case_dir: Path,
    asset_names: list[str] | None = None,
) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    reports_dir = case_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    planner_output = load_json_object(case_dir / "contracts" / "planner_output.json") or {}
    selected, skipped = select_mesh_requests(planner_output, asset_names=asset_names)
    if not selected and not skipped:
        report = _tool_report(
            "generate_mesh_assets",
            ok=True,
            status="no_mesh_requests",
            case_dir=case_dir,
            selected_asset_names=[],
            skipped_asset_names=[],
        )
        return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_mesh_tool_report.json")

    report = generate_mesh_assets_for_episode(
        case_dir=case_dir,
        task=_load_task(case_dir),
        planner_output=planner_output,
        asset_names=asset_names,
    )
    tool_report = _tool_report(
        "generate_mesh_assets",
        ok=bool(report.get("ok")),
        status=str(report.get("status") or "mesh_asset_generation_finished"),
        case_dir=case_dir,
        asset_generation_report=report,
    )
    return _write_tool_report(case_dir, tool_report, reports_dir / "baseline_agent_mesh_tool_report.json")


def apply_adaptive_ipc_d_hat(*, case_dir: Path, out_dir: Path | None = None) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    reports_dir = case_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    config_path = case_dir / "contracts" / "deformable_config.json"
    deformable_config = load_json_object(config_path)
    if deformable_config is None:
        report = _tool_report(
            "apply_adaptive_ipc_d_hat",
            ok=False,
            status="deformable_config_missing_or_invalid",
            case_dir=case_dir,
            errors=[f"Missing or invalid JSON object: {config_path}"],
        )
        return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_adaptive_ipc_report.json")

    if not bool(deformable_config.get("ipc_enabled")):
        report = _tool_report(
            "apply_adaptive_ipc_d_hat",
            ok=True,
            status="ipc_disabled",
            case_dir=case_dir,
            deformable_config_path=str(config_path),
        )
        return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_adaptive_ipc_report.json")
    if not bool(deformable_config.get("ipc_contact_d_hat_adaptive", False)):
        report = _tool_report(
            "apply_adaptive_ipc_d_hat",
            ok=True,
            status="adaptive_disabled",
            case_dir=case_dir,
            deformable_config_path=str(config_path),
        )
        return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_adaptive_ipc_report.json")

    adaptive_report = apply_adaptive_contact_d_hat(
        deformable_config,
        out_dir or case_dir / "artifacts",
        case_root=case_dir,
        default_deformable_cfg=deformable_config,
        repo_root=REPO_ROOT,
    )
    dump_json(deformable_config, config_path)
    status = "adaptive_applied" if adaptive_report is not None else "no_adaptive_candidate"
    report = _tool_report(
        "apply_adaptive_ipc_d_hat",
        ok=True,
        status=status,
        case_dir=case_dir,
        deformable_config_path=str(config_path),
        ipc_contact_d_hat=deformable_config.get("ipc_contact_d_hat"),
        adaptive_report=adaptive_report,
    )
    return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_adaptive_ipc_report.json")


def run_simulation(
    *,
    case_dir: Path,
    backend: str,
    timeout_sec: float,
    render: bool,
    steps: int | None = None,
    duration_sec: float | None = None,
    render_fps: int | None = None,
    ensure_mesh_assets: bool = False,
) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    reports_dir = case_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    contract_report = prepare_contracts(
        case_dir=case_dir,
        steps=steps,
        duration_sec=duration_sec,
        render_fps=render_fps,
    )
    if not contract_report.get("ok"):
        report = _tool_report(
            "run_simulation",
            ok=False,
            status="contracts_not_ready",
            case_dir=case_dir,
            contract_report=contract_report,
        )
        return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_simulation_report.json")

    mesh_report = generate_mesh_assets(case_dir=case_dir) if ensure_mesh_assets else None
    if mesh_report is not None and not mesh_report.get("ok"):
        report = _tool_report(
            "run_simulation",
            ok=False,
            status="mesh_assets_not_ready",
            case_dir=case_dir,
            contract_report=contract_report,
            mesh_report=mesh_report,
        )
        return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_simulation_report.json")

    adaptive_report = apply_adaptive_ipc_d_hat(case_dir=case_dir)

    timing = _timing_from_contract(contract_report)
    if timing is None:
        report = _tool_report(
            "run_simulation",
            ok=False,
            status="timing_contract_invalid",
            case_dir=case_dir,
            contract_report=contract_report,
        )
        return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_simulation_report.json")

    main_py = case_dir / "src" / "main.py"
    if not main_py.is_file():
        report = _tool_report(
            "run_simulation",
            ok=False,
            status="main_missing",
            case_dir=case_dir,
            errors=[f"Generated entrypoint not found: {main_py}"],
        )
        return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_simulation_report.json")

    execution = run_generated_simulation(
        main_py=main_py,
        run_dir=case_dir,
        backend=backend,
        timeout_sec=timeout_sec,
        steps=timing.steps,
        render_fps=timing.render_fps,
        sim_dt=timing.sim_dt,
        sim_substeps=timing.sim_substeps,
        render_every_n_steps=timing.render_every_n_steps,
        render_res=timing.render_res,
        render=render,
        duration_sec=timing.duration_sec,
        target_video_frames=timing.target_video_frames,
    )
    report = _tool_report(
        "run_simulation",
        ok=execution.ok,
        status="execution_passed" if execution.ok else "execution_failed",
        case_dir=case_dir,
        contract_report=contract_report,
        mesh_report=mesh_report,
        adaptive_ipc_report=adaptive_report,
        execution=execution.to_dict(),
    )
    return _write_tool_report(case_dir, report, reports_dir / "baseline_agent_simulation_report.json")


def _physics_mode_from_planner_output(planner_output: dict[str, Any]) -> str:
    physics_plan = planner_output.get("physics_plan")
    if isinstance(physics_plan, dict):
        mode = str(physics_plan.get("mode") or "")
        if mode in {"rigid", "rigid_ipc", "fem_ipc"}:
            return mode
    return "rigid"


def _timing_from_contract(report: dict[str, Any]) -> BaselineTimingPlan | None:
    timing = report.get("timing")
    if not isinstance(timing, dict):
        return None
    render_res = timing.get("render_res")
    if not isinstance(render_res, list | tuple) or len(render_res) != 2:
        return None
    try:
        return BaselineTimingPlan(
            duration_sec=None if timing.get("duration_sec") is None else float(timing["duration_sec"]),
            steps=int(timing["steps"]),
            render_fps=int(timing["render_fps"]),
            target_video_frames=None
            if timing.get("target_video_frames") is None
            else int(timing["target_video_frames"]),
            sim_dt=float(timing["sim_dt"]),
            sim_substeps=int(timing["sim_substeps"]),
            render_every_n_steps=int(timing["render_every_n_steps"]),
            render_res=(int(render_res[0]), int(render_res[1])),
            source=str(timing.get("source") or "contract"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _load_task(case_dir: Path) -> str:
    path = case_dir / "inputs" / "user_prompt.md"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _tool_report(command: str, *, ok: bool, status: str, case_dir: Path, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "command": command,
        "ok": ok,
        "status": status,
        "case_dir": str(case_dir),
        "created_at_unix": time.time(),
        **fields,
    }


def _write_tool_report(case_dir: Path, report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    report["report_path"] = str(report_path)
    dump_json(report, report_path)
    history_path = case_dir / "reports" / TOOL_HISTORY_NAME
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(report, ensure_ascii=False) + "\n")
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, ensure_ascii=False))


def _cmd_prepare_contracts(args: argparse.Namespace) -> None:
    report = prepare_contracts(
        case_dir=args.case_dir,
        steps=args.steps,
        duration_sec=args.duration_sec,
        render_fps=args.render_fps,
    )
    _print_report(report)
    raise SystemExit(0 if report.get("ok") else 1)


def _cmd_generate_mesh_assets(args: argparse.Namespace) -> None:
    report = generate_mesh_assets(case_dir=args.case_dir, asset_names=args.asset_name or None)
    _print_report(report)
    raise SystemExit(0 if report.get("ok") else 1)


def _cmd_apply_adaptive_ipc(args: argparse.Namespace) -> None:
    report = apply_adaptive_ipc_d_hat(case_dir=args.case_dir)
    _print_report(report)
    raise SystemExit(0 if report.get("ok") else 1)


def _cmd_run_simulation(args: argparse.Namespace) -> None:
    report = run_simulation(
        case_dir=args.case_dir,
        backend=args.backend,
        timeout_sec=args.timeout_sec,
        render=args.render,
        steps=args.steps,
        duration_sec=args.duration_sec,
        render_fps=args.render_fps,
        ensure_mesh_assets=args.ensure_mesh_assets,
    )
    _print_report(report)
    raise SystemExit(0 if report.get("ok") else 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m baselines.end_to_end_codex.case_tools")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-contracts", help="Resolve planner_output into timing/deformable contracts.")
    _add_case_and_timing_args(prepare)
    prepare.set_defaults(func=_cmd_prepare_contracts)

    mesh = sub.add_parser("generate-mesh-assets", help="Run Meshy generation for current planner_output requests.")
    mesh.add_argument("--case-dir", type=Path, required=True)
    mesh.add_argument("--asset-name", action="append", default=[])
    mesh.set_defaults(func=_cmd_generate_mesh_assets)

    adaptive = sub.add_parser("apply-adaptive-ipc", help="Apply adaptive IPC d-hat to deformable_config.json.")
    adaptive.add_argument("--case-dir", type=Path, required=True)
    adaptive.set_defaults(func=_cmd_apply_adaptive_ipc)

    sim = sub.add_parser("run-simulation", help="Run generated src/main.py through the locked Genesis harness.")
    _add_case_and_timing_args(sim)
    sim.add_argument("--backend", choices=("cpu", "gpu"), default=DEFAULT_BACKEND)
    sim.add_argument("--timeout-sec", type=float, default=DEFAULT_EXECUTION_TIMEOUT_SEC)
    sim.add_argument("--render", action="store_true", default=True)
    sim.add_argument("--no-render", action="store_false", dest="render")
    sim.add_argument("--ensure-mesh-assets", action="store_true")
    sim.set_defaults(func=_cmd_run_simulation)
    return parser


def _add_case_and_timing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--render-fps", type=int, default=None)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
