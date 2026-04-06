from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import fcntl
import multiprocessing as mp
from pathlib import Path
from typing import Any

from ..io_utils import dump_json
from ..llm_critic import CriticEvaluationInput, evaluate_prompt_event_video
from ..llm_generator import OpenAIResponsesClient, generate_ir_two_agent
from ..llm_generator.constraints import parse_sanitize_validate
from ..runtime import build_llm_event_pack, run_rigid_ir
from ..tool_library import GeneratorParameterOverrides
from .feedback import build_generator_feedback_package

_SIMULATION_LOCK_PATH = Path(__file__).resolve().parents[1] / "runs" / ".simulation.lock"


@dataclass(slots=True)
class OptimizationConfig:
    model: str = "gpt-5.4"
    xml_model: str | None = None
    critic_model: str | None = None
    hosted_prompt_id: str | None = None
    hosted_prompt_version: str | None = None
    critic_hosted_prompt_id: str | None = None
    critic_hosted_prompt_version: str | None = None
    critic_prompt_variant: str = "full"
    temperature: float | None = None
    critic_temperature: float | None = None
    reasoning_effort: str | None = None
    critic_reasoning_effort: str | None = None
    backend: str = "cpu"
    max_opt_rounds: int = 3
    generator_max_rounds: int = 12
    xml_max_attempts: int = 4
    timeout_sec: float = 600.0
    assets_dir: str = "agent/generated_assets"
    mesh_assets_dir: str = "agent/generated_meshes"
    generator_parameter_overrides: GeneratorParameterOverrides = field(
        default_factory=lambda: GeneratorParameterOverrides(
            sim_dt=0.001,
            render_every_n_steps=10,
            render_res=(640, 480),
        )
    )
    sample_every_sec: float = 0.5
    max_frames: int = 24
    max_width: int = 640
    output_root: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"


@dataclass(slots=True)
class OptimizationRoundResult:
    round_index: int
    verdict: str | None
    passed: bool
    round_dir: str


@dataclass(slots=True)
class OptimizationResult:
    task: str
    status: str
    rounds: list[OptimizationRoundResult]
    final_round_dir: str
    final_verdict: str | None


@dataclass(slots=True)
class OptimizationTaskSpec:
    case_id: str
    task: str


@dataclass(slots=True)
class BatchOptimizationItemResult:
    case_id: str
    task: str
    status: str
    final_round_dir: str
    final_verdict: str | None
    rounds: list[OptimizationRoundResult]


@dataclass(slots=True)
class BatchOptimizationResult:
    status: str
    run_root: str
    items: list[BatchOptimizationItemResult]


def optimize_prompt(
    *,
    task: str,
    config: OptimizationConfig,
) -> OptimizationResult:
    client = OpenAIResponsesClient.from_env(
        api_key_env=config.api_key_env,
        base_url_env=config.base_url_env,
        timeout_sec=config.timeout_sec,
    )

    run_root = _resolve_run_root(config.output_root)
    rounds: list[OptimizationRoundResult] = []
    feedback_package: dict[str, Any] | None = None
    previous_ir_json: dict[str, Any] | None = None
    previous_xml_texts_by_body: dict[str, str] = {}

    final_round_dir = run_root
    final_verdict: str | None = None

    for round_index in range(1, config.max_opt_rounds + 1):
        round_dir = run_root / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        final_round_dir = round_dir
        assets_root = Path(config.assets_dir) / run_root.name
        round_assets_dir = assets_root / f"round_{round_index:02d}"
        round_assets_dir.mkdir(parents=True, exist_ok=True)
        mesh_assets_root = Path(config.mesh_assets_dir) / run_root.name
        round_mesh_assets_dir = mesh_assets_root / f"round_{round_index:02d}"
        round_mesh_assets_dir.mkdir(parents=True, exist_ok=True)

        generator_result = generate_ir_two_agent(
            task=task,
            model=config.model,
            client=client,
            xml_model=config.xml_model,
            max_rounds=config.generator_max_rounds,
            xml_max_attempts=config.xml_max_attempts,
            temperature=config.temperature,
            reasoning_effort=config.reasoning_effort,
            normalize=True,
            assets_dir=str(round_assets_dir),
            mesh_assets_dir=str(round_mesh_assets_dir),
            force_primitive_mode=False,
            additional_requirements=(
                None if feedback_package is None else feedback_package["generator_requirements"]
            ),
            xml_feedback_requirements_by_body=(
                None if feedback_package is None else feedback_package.get("xml_requirements_by_body")
            ),
            previous_ir_json=previous_ir_json,
            previous_xml_texts_by_body=previous_xml_texts_by_body,
            hosted_prompt_id=config.hosted_prompt_id,
            hosted_prompt_version=config.hosted_prompt_version,
            parameter_overrides=config.generator_parameter_overrides,
        )

        ir_generated = round_dir / "ir.generated.json"
        generation_log = round_dir / "generation.log.json"
        ir_run = round_dir / "ir.run.json"
        ir_validated = round_dir / "ir.validated.json"
        run_result_path = round_dir / "run_result.json"
        event_pack_path = round_dir / "event_pack.json"
        critic_json = round_dir / "critic.json"
        critic_log = round_dir / "critic.log.json"
        task_path = round_dir / "task.txt"
        refinement_path = round_dir / "generator_feedback.txt"
        refinement_json_path = round_dir / "generator_feedback.json"
        video_path = round_dir / "render.mp4"

        task_path.write_text(task + "\n", encoding="utf-8")
        if feedback_package is not None:
            refinement_path.write_text(feedback_package["generator_requirements"] + "\n", encoding="utf-8")
            dump_json(feedback_package, refinement_json_path)

        dump_json(generator_result.ir_json, ir_generated)
        dump_json(_generation_log_payload(generator_result, config), generation_log)

        run_payload = _prepare_run_payload(
            generator_result.ir_json,
            backend=config.backend,
            video_path=video_path,
        )
        dump_json(run_payload, ir_run)

        validated_program = parse_sanitize_validate(run_payload, normalize=True)
        dump_json(validated_program.model_dump(mode="json"), ir_validated)
        previous_ir_json = validated_program.model_dump(mode="json")
        previous_xml_texts_by_body = _load_articulated_asset_texts_by_body(validated_program)

        with _simulation_file_lock():
            raw_result = run_rigid_ir(validated_program, normalize=False)
        dump_json(raw_result, run_result_path)

        event_pack = build_llm_event_pack(validated_program, raw_result)
        dump_json(event_pack, event_pack_path)

        try:
            critic_result = evaluate_prompt_event_video(
                client=client,
                model=config.critic_model or config.model,
                eval_input=CriticEvaluationInput(
                    task=task,
                    ir_path=ir_validated,
                    event_pack_path=event_pack_path,
                    video_path=video_path,
                    xml_paths_by_body=_resolve_articulated_asset_paths_by_body(validated_program),
                    sample_every_sec=config.sample_every_sec,
                    max_frames=config.max_frames,
                    max_width=config.max_width,
                    generator_parameter_overrides=config.generator_parameter_overrides,
                ),
                temperature=config.critic_temperature,
                reasoning_effort=config.critic_reasoning_effort or config.reasoning_effort,
                hosted_prompt_id=config.critic_hosted_prompt_id,
                hosted_prompt_version=config.critic_hosted_prompt_version,
                prompt_variant=config.critic_prompt_variant,
            )
            analysis_json = critic_result.analysis_json
            critic_log_payload = {
                "mode": "critic_evaluation",
                "model": critic_result.model,
                "input_digest": critic_result.input_digest,
                "frames_used": critic_result.frames_used,
                "raw_response_text": critic_result.raw_response_text,
                "analysis_json": analysis_json,
            }
        except Exception as exc:  # noqa: BLE001
            analysis_json = _build_synthetic_critic_analysis(raw_result=raw_result, event_pack=event_pack, error_text=str(exc))
            critic_log_payload = {
                "mode": "synthetic_critic_failure",
                "error": str(exc),
                "analysis_json": analysis_json,
            }

        dump_json(analysis_json, critic_json)
        dump_json(critic_log_payload, critic_log)

        verdict = analysis_json.get("verdict")
        passed = verdict == "pass"
        final_verdict = verdict if isinstance(verdict, str) else None
        rounds.append(
            OptimizationRoundResult(
                round_index=round_index,
                verdict=final_verdict,
                passed=passed,
                round_dir=str(round_dir),
            )
        )
        if passed:
            return OptimizationResult(
                task=task,
                status="passed",
                rounds=rounds,
                final_round_dir=str(round_dir),
                final_verdict=final_verdict,
            )

        feedback_package = build_generator_feedback_package(analysis_json)

    return OptimizationResult(
        task=task,
        status="max_rounds_exhausted",
        rounds=rounds,
        final_round_dir=str(final_round_dir),
        final_verdict=final_verdict,
    )


def _run_batch_task(
    spec: OptimizationTaskSpec,
    config: OptimizationConfig,
    run_root_str: str,
) -> BatchOptimizationItemResult:
    run_root = Path(run_root_str)
    case_root = run_root / spec.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "task.txt").write_text(spec.task + "\n", encoding="utf-8")
    case_config = replace(config, output_root=str(case_root))
    result = optimize_prompt(task=spec.task, config=case_config)
    return BatchOptimizationItemResult(
        case_id=spec.case_id,
        task=spec.task,
        status=result.status,
        final_round_dir=result.final_round_dir,
        final_verdict=result.final_verdict,
        rounds=result.rounds,
    )


def optimize_prompts_batch(
    *,
    task_specs: list[OptimizationTaskSpec],
    config: OptimizationConfig,
    max_parallel: int = 4,
) -> BatchOptimizationResult:
    if not task_specs:
        raise ValueError("`task_specs` must contain at least one task.")
    if max_parallel < 1:
        raise ValueError("`max_parallel` must be >= 1.")

    run_root = _resolve_run_root(config.output_root)
    ordered_results: list[BatchOptimizationItemResult | None] = [None] * len(task_specs)

    executor_kwargs: dict[str, Any] = {
        "max_workers": min(max_parallel, len(task_specs)),
    }
    if config.backend == "gpu":
        # GPU + forked workers is fragile because CUDA/Taichi state can leak across
        # process boundaries. Use spawn and recycle workers after one task to keep each
        # case isolated.
        executor_kwargs["mp_context"] = mp.get_context("spawn")
        executor_kwargs["max_tasks_per_child"] = 1

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        futures = [
            executor.submit(_run_batch_task, spec, config, str(run_root))
            for spec in task_specs
        ]
        for index, future in enumerate(futures):
            item = future.result()
            ordered_results[index] = item

    items = [item for item in ordered_results if item is not None]
    overall_status = "passed" if items and all(item.status == "passed" for item in items) else "completed"
    return BatchOptimizationResult(
        status=overall_status,
        run_root=str(run_root),
        items=items,
    )


class _simulation_file_lock:
    def __enter__(self):
        _SIMULATION_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._file = _SIMULATION_LOCK_PATH.open("a+", encoding="utf-8")
        fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()


def _resolve_run_root(output_root: str | None) -> Path:
    if output_root is not None:
        path = Path(output_root)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("agent/runs/opt") / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_run_payload(
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


def _resolve_articulated_asset_paths_by_body(program) -> dict[str, Path]:
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


def _load_articulated_asset_texts_by_body(program) -> dict[str, str]:
    texts_by_body: dict[str, str] = {}
    for body_name, path in _resolve_articulated_asset_paths_by_body(program).items():
        texts_by_body[body_name] = path.read_text(encoding="utf-8")
    return texts_by_body


def _generation_log_payload(result, config: OptimizationConfig) -> dict[str, Any]:
    xml_results_by_body = result.xml_results_by_body
    mesh_results_by_body = result.mesh_results_by_body
    return {
        "model": result.model,
        "mode": result.mode,
        "articulated_requested": result.articulated_requested,
        "generator_parameter_overrides": config.generator_parameter_overrides.as_dict(),
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
    }


def _build_synthetic_critic_analysis(
    *,
    raw_result: dict[str, Any],
    event_pack: dict[str, Any],
    error_text: str,
) -> dict[str, Any]:
    crash_any = raw_result.get("crash")
    crash = dict(crash_any) if isinstance(crash_any, dict) else None
    crash_summary = "Simulation crashed during execution." if crash is not None else "Critic evaluation could not complete."
    crash_evidence: list[str] = []
    if crash is not None:
        crash_stage = crash.get("stage")
        crash_step = crash.get("step")
        attempted_step = crash.get("attempted_step")
        crash_time = crash.get("time_sec")
        attempted_time = crash.get("attempted_time_sec")
        crash_message = crash.get("error_message")
        if isinstance(crash_stage, str):
            crash_evidence.append(f"Crash stage: {crash_stage}")
        if isinstance(crash_step, int | float) and not isinstance(crash_step, bool):
            crash_evidence.append(f"Last completed step: {int(crash_step)}")
        if isinstance(attempted_step, int | float) and not isinstance(attempted_step, bool):
            crash_evidence.append(f"Attempted crash step: {int(attempted_step)}")
        if isinstance(crash_time, int | float) and not isinstance(crash_time, bool):
            crash_evidence.append(f"Last completed time_sec: {float(crash_time):.6f}")
        if isinstance(attempted_time, int | float) and not isinstance(attempted_time, bool):
            crash_evidence.append(f"Attempted crash time_sec: {float(attempted_time):.6f}")
        if isinstance(crash_message, str) and crash_message:
            crash_evidence.append(f"Crash error: {crash_message}")
    crash_evidence.append(f"Critic fallback reason: {error_text}")

    scene_fix = (
        "Keep the fixed simulation/render overrides unchanged. Revise scene setup to avoid unstable contact "
        "configurations, interpenetration, or impossible initial placements."
    )
    body_fix = (
        "Revise robot geometry, joint layout, wheel axis orientation, or mass distribution so the model can step "
        "stably under the fixed simulation parameters."
    )
    actions_fix = (
        "Reduce abrupt control changes and overly aggressive torques or target jumps. Add more gradual action "
        "transitions and enough settling time between commands."
    )

    return {
        "verdict": "fail",
        "overall_score": 0,
        "summary": crash_summary,
        "by_section": {
            "scene": {
                "score": 0,
                "summary": "The round did not complete a stable executable simulation.",
                "strengths": [],
                "issues": [
                    {
                        "severity": "high",
                        "title": "Scene became unstable before a full evaluation could complete",
                        "evidence": crash_evidence,
                        "fix": scene_fix,
                    }
                ],
            },
            "actions": {
                "score": 0,
                "summary": "Action sequence or control aggressiveness likely exceeded stable operating limits.",
                "strengths": [],
                "issues": [
                    {
                        "severity": "high",
                        "title": "Action program needs more conservative dynamics",
                        "evidence": crash_evidence,
                        "fix": actions_fix,
                    }
                ],
            },
        },
        "by_body": {
            entity_name: {
                "score": 0,
                "summary": "This body likely contributed to the failed or unstable execution.",
                "strengths": [],
                "issues": [
                    {
                        "severity": "high",
                        "title": f"Revise body `{entity_name}` for stability",
                        "evidence": crash_evidence,
                        "fix": body_fix,
                    }
                ],
            }
            for entity_name in event_pack.get("entities", {})
            if isinstance(entity_name, str)
        },
        "cross_checks": {
            "ir_vs_event": "Execution metadata indicates a failed or incomplete run.",
            "event_vs_video": (
                "Cross-check is incomplete because the round crashed or critic evaluation had to fall back."
            ),
            "ir_vs_video": (
                "Cross-check is incomplete because the round crashed or critic evaluation had to fall back."
            ),
        },
        "priority_fixes": [scene_fix, body_fix, actions_fix],
        "runtime_failure": {
            "raw_result_status": raw_result.get("status"),
            "crash": crash,
            "execution": event_pack.get("execution"),
        },
    }
