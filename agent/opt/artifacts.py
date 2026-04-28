from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import fcntl
from pathlib import Path
import traceback
from typing import Any

from ..io_utils import dump_json
from ..usage import aggregate_usage_metrics
from .models import BatchOptimizationItemResult, OptimizationTaskSpec, RoundWorkspace

SIMULATION_LOCK_PATH = Path(__file__).resolve().parents[1] / "runs" / ".simulation.lock"


class SimulationFileLock:
    def __enter__(self):
        SIMULATION_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._file = SIMULATION_LOCK_PATH.open("a+", encoding="utf-8")
        fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()


def resolve_run_root(output_root: str | None) -> Path:
    if output_root is not None:
        path = Path(output_root)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("agent/runs/opt") / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_round_workspace(*, run_root: Path, config, round_index: int) -> RoundWorkspace:
    round_name = f"round_{round_index:02d}"
    round_dir = run_root / round_name
    round_dir.mkdir(parents=True, exist_ok=True)

    assets_dir = Path(config.assets_dir) / run_root.name / round_name
    assets_dir.mkdir(parents=True, exist_ok=True)
    mesh_assets_dir = Path(config.mesh_assets_dir) / run_root.name / round_name
    mesh_assets_dir.mkdir(parents=True, exist_ok=True)

    return RoundWorkspace(
        round_dir=round_dir,
        assets_dir=assets_dir,
        mesh_assets_dir=mesh_assets_dir,
        ir_generated=round_dir / "ir.generated.json",
        generation_log=round_dir / "generation.log.json",
        ir_run=round_dir / "ir.run.json",
        ir_validated=round_dir / "ir.validated.json",
        run_result=round_dir / "run_result.json",
        event_pack=round_dir / "event_pack.json",
        critic_json=round_dir / "critic.json",
        critic_log=round_dir / "critic.log.json",
        usage_json=round_dir / "llm_usage.json",
        task_txt=round_dir / "task.txt",
        generator_feedback_txt=round_dir / "generator_feedback.txt",
        generator_feedback_json=round_dir / "generator_feedback.json",
        video_path=round_dir / "render.mp4",
    )


def build_batch_failure_result(
    *,
    spec: OptimizationTaskSpec,
    case_root: Path,
    exc: Exception,
) -> BatchOptimizationItemResult:
    failure_payload = {
        "case_id": spec.case_id,
        "task": spec.task,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    dump_json(failure_payload, case_root / "failure.json")
    return BatchOptimizationItemResult(
        case_id=spec.case_id,
        task=spec.task,
        status="failed",
        final_round_dir=str(case_root),
        final_verdict=None,
        rounds=[],
        error=str(exc),
    )


def prepare_run_payload(
    ir_json: dict[str, Any],
    *,
    backend: str,
    video_path: Path,
) -> dict[str, Any]:
    payload = dict(ir_json)
    scene_any = payload.get("scene")
    scene = dict(scene_any) if isinstance(scene_any, dict) else {}
    scene["backend"] = backend
    scene["show_viewer"] = False

    render_any = scene.get("render")
    render = dict(render_any) if isinstance(render_any, dict) else {}
    render["output_video"] = str(video_path)
    render["gui"] = False
    scene["render"] = render
    payload["scene"] = scene
    return payload


def resolve_articulated_asset_paths_by_body(program) -> dict[str, Path]:
    paths_by_body: dict[str, Path] = {}
    for body in program.bodies:
        if body.shape.kind not in {"mjcf", "urdf"}:
            continue
        file_path = getattr(body.shape, "file", None)
        if not isinstance(file_path, str):
            continue
        path = Path(file_path)
        if path.exists():
            paths_by_body[body.name] = path
    return paths_by_body


def load_articulated_asset_texts_by_body(program) -> dict[str, str]:
    texts_by_body: dict[str, str] = {}
    for body_name, path in resolve_articulated_asset_paths_by_body(program).items():
        texts_by_body[body_name] = path.read_text(encoding="utf-8")
    return texts_by_body


def build_generation_log_payload(result) -> dict[str, Any]:
    xml_results_by_body = result.xml_results_by_body
    mesh_results_by_body = result.mesh_results_by_body
    return {
        "model": result.model,
        "mode": result.mode,
        "articulated_requested": result.articulated_requested,
        "ir_rounds": result.ir_result.rounds,
        "xml_results_by_body": {
            body_name: {
                "xml_path": xml_result.xml_path,
                "attempts": xml_result.attempts,
            }
            for body_name, xml_result in sorted(xml_results_by_body.items())
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
            for body_name, mesh_result in sorted(mesh_results_by_body.items())
        },
        "ir_logs": [asdict(log) for log in result.ir_result.logs],
        "xml_logs_by_body": {
            body_name: [asdict(log) for log in xml_result.logs]
            for body_name, xml_result in sorted(xml_results_by_body.items())
        },
        "mesh_logs_by_body": {
            body_name: [asdict(log) for log in mesh_result.logs]
            for body_name, mesh_result in sorted(mesh_results_by_body.items())
        },
        "usage_summary": build_generator_usage_summary(result),
    }


def build_generator_usage_summary(result) -> dict[str, Any]:
    ir_entries = [getattr(log, "usage", None) for log in result.ir_result.logs]
    xml_entries = [
        getattr(log, "usage", None)
        for xml_result in result.xml_results_by_body.values()
        for log in xml_result.logs
    ]
    return {
        "generator_ir": aggregate_usage_metrics(ir_entries),
        "generator_xml": aggregate_usage_metrics(xml_entries),
        "generator_total": aggregate_usage_metrics([*ir_entries, *xml_entries]),
    }


def build_round_usage_payload(generator_result, critic_log_payload: dict[str, Any]) -> dict[str, Any]:
    generator_usage = build_generator_usage_summary(generator_result)
    critic_stage_logs = critic_log_payload.get("stage_logs")
    critic_entries = [
        log.get("usage")
        for log in critic_stage_logs
        if isinstance(log, dict)
    ] if isinstance(critic_stage_logs, list) else []
    critic_usage = aggregate_usage_metrics(critic_entries)
    return {
        "generator": generator_usage,
        "critic": critic_usage,
        "total": aggregate_usage_metrics([generator_usage["generator_total"], critic_usage]),
        "by_component": {
            **generator_usage,
            "critic_total": critic_usage,
        },
    }
