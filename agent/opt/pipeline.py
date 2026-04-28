from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing as mp
from pathlib import Path
from typing import Any

from ..io_utils import dump_json
from ..llm_critic.critic import CriticEvaluationInput, evaluate_prompt_event_video
from ..llm_generator.agents.two_agent_generator import generate_ir_two_agent
from ..llm_generator.client.openai_client import OpenAIResponsesClient
from ..llm_generator.constraints.general_constraints import parse_sanitize_validate
from ..runtime.event_pack import build_llm_event_pack
from ..runtime.runner import run_rigid_ir
from .artifacts import (
    SimulationFileLock,
    build_batch_failure_result,
    build_generation_log_payload,
    build_round_usage_payload,
    load_articulated_asset_texts_by_body,
    prepare_round_workspace,
    prepare_run_payload,
    resolve_articulated_asset_paths_by_body,
    resolve_run_root,
)
from .feedback import build_generator_feedback_package
from .models import (
    BatchOptimizationItemResult,
    BatchOptimizationResult,
    OptimizationConfig,
    OptimizationResult,
    OptimizationRoundResult,
    OptimizationTaskSpec,
)


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

    run_root = resolve_run_root(config.output_root)
    rounds: list[OptimizationRoundResult] = []
    feedback_package: dict[str, Any] | None = None
    previous_ir_json: dict[str, Any] | None = None
    previous_xml_texts_by_body: dict[str, str] = {}

    final_round_dir = run_root
    final_verdict: str | None = None

    for round_index in range(1, config.max_opt_rounds + 1):
        workspace = prepare_round_workspace(run_root=run_root, config=config, round_index=round_index)
        final_round_dir = workspace.round_dir

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
            assets_dir=str(workspace.assets_dir),
            mesh_assets_dir=str(workspace.mesh_assets_dir),
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
            mesh_texture_enabled=config.mesh_texture_enabled,
            log_path=workspace.generation_log,
        )

        workspace.task_txt.write_text(task + "\n", encoding="utf-8")
        if feedback_package is not None:
            workspace.generator_feedback_txt.write_text(
                feedback_package["generator_requirements"] + "\n",
                encoding="utf-8",
            )
            dump_json(feedback_package, workspace.generator_feedback_json)

        dump_json(generator_result.ir_json, workspace.ir_generated)
        dump_json(build_generation_log_payload(generator_result), workspace.generation_log)

        run_payload = prepare_run_payload(
            generator_result.ir_json,
            backend=config.backend,
            video_path=workspace.video_path,
        )
        dump_json(run_payload, workspace.ir_run)

        validated_program = parse_sanitize_validate(run_payload, normalize=True)
        dump_json(validated_program.model_dump(mode="json"), workspace.ir_validated)
        previous_ir_json = validated_program.model_dump(mode="json")
        previous_xml_texts_by_body = load_articulated_asset_texts_by_body(validated_program)

        with SimulationFileLock():
            raw_result = run_rigid_ir(validated_program, normalize=False)
        dump_json(raw_result, workspace.run_result)

        event_pack = build_llm_event_pack(validated_program, raw_result)
        dump_json(event_pack, workspace.event_pack)

        try:
            critic_result = evaluate_prompt_event_video(
                client=client,
                model=config.critic_model or config.model,
                eval_input=CriticEvaluationInput(
                    task=task,
                    ir_path=workspace.ir_validated,
                    event_pack_path=workspace.event_pack,
                    video_path=workspace.video_path,
                    xml_paths_by_body=resolve_articulated_asset_paths_by_body(validated_program),
                    sample_every_sec=config.sample_every_sec,
                    max_frames=config.max_frames,
                    max_width=config.max_width,
                ),
                temperature=config.critic_temperature,
                reasoning_effort=config.critic_reasoning_effort or config.reasoning_effort,
                hosted_prompt_id=config.critic_hosted_prompt_id,
                hosted_prompt_version=config.critic_hosted_prompt_version,
                prompt_variant=config.critic_prompt_variant,
                log_path=workspace.critic_log,
            )
            analysis_json = critic_result.analysis_json
            critic_log_payload = {
                "mode": "critic_evaluation",
                "model": critic_result.model,
                "input_digest": critic_result.input_digest,
                "frames_used": critic_result.frames_used,
                "raw_response_text": critic_result.raw_response_text,
                "analysis_json": analysis_json,
                "usage_summary": critic_result.usage_summary,
                "stage_logs": critic_result.stage_logs,
            }
        except Exception as exc:  # noqa: BLE001
            analysis_json = _build_synthetic_critic_analysis(raw_result=raw_result, event_pack=event_pack, error_text=str(exc))
            critic_log_payload = {
                "mode": "synthetic_critic_failure",
                "error": str(exc),
                "analysis_json": analysis_json,
                "usage_summary": {},
                "stage_logs": [],
            }

        dump_json(analysis_json, workspace.critic_json)
        dump_json(critic_log_payload, workspace.critic_log)
        dump_json(build_round_usage_payload(generator_result, critic_log_payload), workspace.usage_json)

        verdict = analysis_json.get("verdict")
        passed = verdict == "pass"
        final_verdict = verdict if isinstance(verdict, str) else None
        rounds.append(
            OptimizationRoundResult(
                round_index=round_index,
                verdict=final_verdict,
                passed=passed,
                round_dir=str(workspace.round_dir),
            )
        )
        if passed:
            return OptimizationResult(
                task=task,
                status="passed",
                rounds=rounds,
                final_round_dir=str(workspace.round_dir),
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
    try:
        result = optimize_prompt(task=spec.task, config=case_config)
        return BatchOptimizationItemResult(
            case_id=spec.case_id,
            task=spec.task,
            status=result.status,
            final_round_dir=result.final_round_dir,
            final_verdict=result.final_verdict,
            rounds=result.rounds,
        )
    except Exception as exc:  # noqa: BLE001
        return build_batch_failure_result(spec=spec, case_root=case_root, exc=exc)


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

    run_root = resolve_run_root(config.output_root)
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
            try:
                item = future.result()
            except Exception as exc:  # noqa: BLE001
                spec = task_specs[index]
                case_root = run_root / spec.case_id
                case_root.mkdir(parents=True, exist_ok=True)
                item = build_batch_failure_result(spec=spec, case_root=case_root, exc=exc)
            ordered_results[index] = item

    items = [item for item in ordered_results if item is not None]
    overall_status = "passed" if items and all(item.status == "passed" for item in items) else "completed"
    return BatchOptimizationResult(
        status=overall_status,
        run_root=str(run_root),
        items=items,
    )


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
        "Revise scene setup to avoid unstable contact configurations, interpenetration, or impossible initial "
        "placements. Do not rely on changing simulator timing or render cadence to fix the failure."
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
