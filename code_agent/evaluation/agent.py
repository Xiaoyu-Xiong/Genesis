from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from code_agent.configs import CONFIGS
from code_agent.io_utils import load_json_object
from code_agent.prompts import prompt_mode
from code_agent.prompts.common import BUILTIN_ASSET_POLICY_GUIDE, SOURCE_AWARE_REPAIR_GUIDE
from code_agent.prompts.critic import (
    CRITIC_ASSET_EVALUATION_GUIDE,
    CRITIC_DECISION_GUIDE,
    CRITIC_EVIDENCE_READING_GUIDE,
    CRITIC_GENERAL_RULES,
    CRITIC_VISUAL_EVIDENCE_GUIDE,
)
from code_agent.utils.codex import DEFAULT_REPO_ROOT, CodexExecRequest, run_codex_exec
from code_agent.utils.state_cache import build_state_cache_source_consistency_report

PROMPT_TEXT_LIMITS = {
    "execution_report": 120_000,
    "generated_source_file": 60_000,
    "source_file": 50_000,
    "json_report": 50_000,
    "stdout_stderr": 40_000,
}


def run_codex_critic(
    *,
    run_dir: Path,
    task: str,
    artifact_report: dict[str, Any],
    simdebug_card_context: str = "",
) -> dict[str, Any]:
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"
    source_report = _write_state_cache_source_report(run_dir)
    evidence_index = _write_critic_evidence_index(
        run_dir=run_dir,
        artifact_report=artifact_report,
        source_report=source_report,
    )
    image_paths = _critic_image_paths(run_dir)
    max_attempts = max(1, int(CONFIGS.critic.max_attempts))
    attempts: list[dict[str, Any]] = []
    previous_failure: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    result = None

    for attempt_index in range(max_attempts):
        prompt = _critic_prompt(
            run_dir=run_dir,
            task=task,
            artifact_report=artifact_report,
            evidence_index=evidence_index,
            attempt_index=attempt_index,
            previous_failure=previous_failure,
            simdebug_card_context=simdebug_card_context,
        )
        jsonl_path = logs_dir / ("codex_critic.jsonl" if attempt_index == 0 else f"codex_critic_retry_{attempt_index:02d}.jsonl")
        result = run_codex_exec(
            CodexExecRequest(
                role="critic",
                prompt=prompt,
                cwd=DEFAULT_REPO_ROOT,
                sandbox=CONFIGS.codex.critic_sandbox,
                model=CONFIGS.codex.critic_model,
                output_schema_path=Path("code_agent/specs/critic_report.schema.json"),
                image_paths=image_paths,
                output_jsonl_path=jsonl_path,
                final_message_path=logs_dir / "codex_critic.final.json",
                timeout_sec=CONFIGS.codex.critic_timeout_sec,
                writable_roots=(run_dir,),
            )
        )
        report = load_json_object(Path(result.final_message_path))
        attempt_record = _critic_attempt_record(result=result, attempt_index=attempt_index)
        attempts.append(attempt_record)
        if report is not None:
            break
        previous_failure = attempt_record
        if not _should_retry_critic(result=result, attempt_index=attempt_index, max_attempts=max_attempts):
            break

    if report is None:
        assert result is not None
        if result.error_type == "codex_usage_limit":
            observations = ["Codex critic was blocked by usage limits and did not evaluate the run."]
            failure_modes = ["critic.codex_usage_limit"]
        else:
            observations = ["Codex critic did not return parseable JSON."]
            failure_modes = ["critic.parse_failed"]
        report = {
            "verdict": "inconclusive",
            "score": 0.0,
            "observations": observations,
            "failure_modes": failure_modes,
            "recommended_owner": "none",
            "repair_summary": None,
            "asset_diagnostics": None,
            "cache_source_consistency": _default_cache_source_consistency(source_report),
            "evidence": {"metrics": [], "frames": [], "video": None, "event_logs": []},
        }
    assert result is not None
    report["cache_source_consistency"] = _normalize_cache_source_consistency(
        report.get("cache_source_consistency"),
        source_report,
    )
    report["state_cache_source_report"] = source_report
    report["codex_result"] = {
        "returncode": result.exit_code,
        "ok": result.success,
        "duration_sec": result.duration_sec,
        "final_message_path": result.final_message_path,
        "stderr_path": result.stderr_path,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }
    report["critic_attempts"] = attempts
    report["critic_infra_status"] = _critic_infra_status(report)
    (reports_dir / "codex_critic_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _critic_prompt(
    *,
    run_dir: Path,
    task: str,
    artifact_report: dict[str, Any],
    evidence_index: dict[str, Any],
    attempt_index: int,
    previous_failure: dict[str, Any] | None,
    simdebug_card_context: str = "",
) -> str:
    metrics = _compact_file(run_dir / "artifacts" / "metrics.json")
    render_stats = _compact_file(run_dir / "artifacts" / "render_stats.json")
    visual_evaluation = _visual_digest(run_dir / "reports" / "visual_evaluation.json")
    execution_report = _execution_digest(run_dir / "reports" / "execution_report.json")
    planner_output = _compact_file(run_dir / "contracts" / "planner_output.json")
    timing_contract = _compact_file(run_dir / "contracts" / "timing.json")
    deformable_config = _compact_file(run_dir / "contracts" / "deformable_config.json")
    opt_subagent_report = _compact_file(run_dir / "reports" / "opt_subagent_report.json")
    opt_report = _compact_file(run_dir / "reports" / "opt_report.json")
    opt_verification = _compact_file(run_dir / "reports" / "verification_report.json")
    current_opt_params = _compact_file(run_dir / "contracts" / "current_opt_params.json")
    best_opt_params = _compact_file(run_dir / "contracts" / "best_opt_params.json")
    asset_manifest = _compact_file(run_dir / "assets" / "asset_manifest.json")
    genesis_context = _genesis_context_pointer(run_dir)
    summary = _compact_file(run_dir / "artifacts" / "summary.json")
    run_result = _compact_file(run_dir / "artifacts" / "run_result.json")
    stdout = _read_text(run_dir / "reports" / "stdout.txt", max_chars=CONFIGS.critic.prompt_inline_text_chars)
    stderr = _read_text(run_dir / "reports" / "stderr.txt", max_chars=CONFIGS.critic.prompt_inline_text_chars)
    source_paths = _source_paths_for_prompt(run_dir)
    source_consistency = evidence_index.get("state_cache_source_consistency")
    asset_evidence_paths = _asset_evidence_paths_for_prompt(run_dir)
    retry_note = ""
    if previous_failure is not None:
        retry_note = "\nPrevious critic attempt failed; retry with this compact evidence packet:\n" + json.dumps(
            previous_failure, indent=2
        )
    simdebug_section = (
        f"""
        Planner-dispatched SimDebug cards for Critic:
        {simdebug_card_context or "No SimDebug cards were dispatched for this critic call."}
        """
        if prompt_mode() != "legacy"
        else ""
    )
    return textwrap.dedent(
        f"""
        {CRITIC_GENERAL_RULES}

        Critic attempt: {attempt_index + 1} of {max(1, int(CONFIGS.critic.max_attempts))}
        {retry_note}

        Original task prompt:
        {task}

        {simdebug_section}

        Case workspace:
        {run_dir}

        Evidence index:
        {json.dumps(evidence_index, indent=2)}

        Important: the evidence index contains file paths and sizes. Read files on demand from disk instead of assuming
        every detail is inlined here. The inlined sections below are compact digests to keep this critic call reliable.

        {CRITIC_EVIDENCE_READING_GUIDE}

        Deterministic artifact report digest:
        {json.dumps(_artifact_report_digest(artifact_report), indent=2)}

        Execution report digest:
        {execution_report}

        Metrics digest:
        {metrics}

        Event log:
        Full event log is available at {run_dir / "artifacts" / "event_log.json"}. Read only relevant event slices.

        Render stats digest:
        {render_stats}

        State-cache source consistency report:
        {json.dumps(source_consistency, indent=2)}

        When this report has status=mismatch, inspect every listed diff, cached snapshot, current source file, and any
        relevant worker logs before deciding whether physics must be rerun. Classify the mismatch as:
        - render_only: all changes are limited to renderer setup, camera, lights, visual-only materials/background,
          video encoding, or replay plumbing and cannot alter physical geometry, collision, physical material values,
          solver settings, initialization, forces, actions, or simulated state.
        - physics_affecting: any change can alter geometry/topology, entities, collision, solver/coupler, dt/substeps,
          physical material values, initial state, forces, controls, or actions.
        - indeterminate: the available snapshot/diff/log evidence is insufficient to prove the change is render-only.
        Judge source content, not filenames: a rendering-only edit may live in scene.py or body.py. Set
        physics_rerun_required=false only for not_applicable, match, or a well-supported render_only classification.

        Visual evidence digest:
        {visual_evaluation}

        Planner output digest:
        {planner_output}

        Timing contract:
        {timing_contract}

        FEM/IPC capability/config contract:
        {deformable_config}

        Optimization evidence, if Opt was used:
        Opt subagent report:
        {opt_subagent_report}

        Low-level opt report:
        {opt_report}

        Opt verification report:
        {opt_verification}

        Current opt params:
        {current_opt_params}

        Best opt params:
        {best_opt_params}

        Asset manifest digest:
        {asset_manifest}

        Generated asset source and preview paths:
        {json.dumps(asset_evidence_paths, indent=2)}

        Genesis documentation and local-code context:
        {genesis_context}

        Generated source paths:
        {json.dumps(source_paths, indent=2)}

        Summary artifact:
        {summary}

        Run result:
        {run_result}

        stdout:
        {stdout}

        stderr:
        {stderr}

        {CRITIC_DECISION_GUIDE}

        {CRITIC_ASSET_EVALUATION_GUIDE}

        {CRITIC_VISUAL_EVIDENCE_GUIDE}

        {BUILTIN_ASSET_POLICY_GUIDE}

        If repair is needed, use `repair_summary` for this guidance:
        {SOURCE_AWARE_REPAIR_GUIDE}

        Return JSON matching critic_report.schema.json. `recommended_owner` must be one of:
        planner, scene, body, action, rendering, integrator, execution, none.
        When a generated asset itself is the likely source of failure, set `recommended_owner` to `planner` and populate
        the optional `asset_diagnostics` object with the affected asset names, asset family, evidence, and the Planner
        asset action that should be used next.
        Use `needs_repair` when there is a clear owner-routed fix.
        """
    ).strip()


def _critic_attempt_record(*, result: Any, attempt_index: int) -> dict[str, Any]:
    stderr_tail = ""
    stderr_path = Path(result.stderr_path) if result.stderr_path else None
    if stderr_path is not None and stderr_path.exists():
        stderr_tail = _read_text(stderr_path, max_chars=4000)
    return {
        "attempt": attempt_index + 1,
        "returncode": result.exit_code,
        "ok": result.success,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "output_jsonl_path": result.output_jsonl_path,
        "final_message_path": result.final_message_path,
        "stderr_path": result.stderr_path,
        "stderr_tail": stderr_tail,
        "infra_status": _infra_status_from_error(
            error_type=result.error_type,
            error_message=result.error_message,
            stderr_tail=stderr_tail,
        ),
    }


def _should_retry_critic(*, result: Any, attempt_index: int, max_attempts: int) -> bool:
    if attempt_index + 1 >= max_attempts:
        return False
    if result.error_type == "codex_usage_limit":
        return False
    return True


def _critic_infra_status(report: dict[str, Any]) -> str:
    codex_result = report.get("codex_result")
    attempts = report.get("critic_attempts")
    latest_attempt = attempts[-1] if isinstance(attempts, list) and attempts else None
    if isinstance(latest_attempt, dict):
        status = latest_attempt.get("infra_status")
        if isinstance(status, str) and status != "ok":
            return status
    if isinstance(codex_result, dict):
        status = _infra_status_from_error(
            error_type=codex_result.get("error_type") if isinstance(codex_result.get("error_type"), str) else None,
            error_message=codex_result.get("error_message") if isinstance(codex_result.get("error_message"), str) else None,
            stderr_tail="",
        )
        if status != "ok":
            return status
    failure_modes = report.get("failure_modes")
    if isinstance(failure_modes, list) and "critic.parse_failed" in failure_modes:
        return "critic_parse_failed"
    return "ok"


def _infra_status_from_error(*, error_type: str | None, error_message: str | None, stderr_tail: str) -> str:
    text = "\n".join(item for item in (error_type or "", error_message or "", stderr_tail) if item).lower()
    if "usage limit" in text or "purchase more credits" in text:
        return "quota_blocked"
    if error_type == "codex_auth_failed" or "401 unauthorized" in text:
        return "auth_failed"
    if error_type == "codex_input_too_large" or "input exceeds the maximum length" in text:
        return "critic_prompt_too_large"
    if error_type == "timeout":
        return "critic_timeout"
    if error_type:
        return "critic_exec_failed"
    return "ok"


def _artifact_report_digest(artifact_report: dict[str, Any]) -> dict[str, Any]:
    checks = artifact_report.get("checks")
    failed_checks = []
    if isinstance(checks, list):
        failed_checks = [item for item in checks if isinstance(item, dict) and item.get("status") == "fail"]
    visual_report = artifact_report.get("visual_report")
    visual_digest = None
    if isinstance(visual_report, dict):
        sampled = visual_report.get("sampled_frames")
        visual_digest = {
            "contact_sheet_path": visual_report.get("contact_sheet_path"),
            "num_sampled_frames": len(sampled) if isinstance(sampled, list) else 0,
            "warnings": visual_report.get("warnings"),
            "sampling": visual_report.get("sampling"),
        }
    return {
        "passed": artifact_report.get("passed"),
        "failure_classes": artifact_report.get("failure_classes"),
        "recommended_owner": artifact_report.get("recommended_owner"),
        "repair_summary": artifact_report.get("repair_summary"),
        "failed_checks": failed_checks[:8],
        "visual_report": visual_digest,
    }


def _compact_file(path: Path) -> str:
    payload = load_json_object(path)
    if isinstance(payload, dict):
        return json.dumps(_compact_json_payload(payload), indent=2)
    return _read_text(path, max_chars=CONFIGS.critic.prompt_inline_text_chars)


def _compact_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"artifacts", "frames", "sampled_frames", "frame_summaries"}:
            if isinstance(value, list):
                compact[key] = {"count": len(value), "first": value[:2], "last": value[-2:]}
            elif isinstance(value, dict):
                compact[key] = {"count": len(value), "sample_keys": list(value)[:12]}
            else:
                compact[key] = value
            continue
        compact[key] = value
    text = json.dumps(compact, indent=2)
    if len(text) <= CONFIGS.critic.prompt_inline_json_chars:
        return compact
    return {
        "truncated_digest": True,
        "top_level_keys": list(payload.keys()),
        "head": _clip_middle(text, CONFIGS.critic.prompt_inline_json_chars),
    }


def _execution_digest(path: Path) -> str:
    payload = load_json_object(path)
    if not isinstance(payload, dict):
        return _read_text(path, max_chars=CONFIGS.critic.prompt_inline_text_chars)
    keys = (
        "command",
        "returncode",
        "timed_out",
        "duration_sec",
        "stdout_path",
        "stderr_path",
        "artifacts_dir",
        "error_type",
        "error_message",
    )
    digest = {key: payload.get(key) for key in keys if key in payload}
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        digest["artifact_keys"] = list(artifacts)[:80]
        digest["num_artifacts"] = len(artifacts)
    return json.dumps(digest, indent=2)


def _visual_digest(path: Path) -> str:
    payload = load_json_object(path)
    if not isinstance(payload, dict):
        return _read_text(path, max_chars=CONFIGS.critic.prompt_inline_text_chars)
    sampled = payload.get("sampled_frames")
    frame_summaries = payload.get("frame_summaries")
    digest = {
        "contact_sheet_path": payload.get("contact_sheet_path"),
        "sampling": payload.get("sampling"),
        "num_sampled_frames": len(sampled) if isinstance(sampled, list) else 0,
        "sampled_frames_head": sampled[:3] if isinstance(sampled, list) else [],
        "sampled_frames_tail": sampled[-3:] if isinstance(sampled, list) else [],
        "diagnostic_frames": payload.get("diagnostic_frames"),
        "texture_summaries": payload.get("texture_summaries"),
        "texture_presence": payload.get("texture_presence"),
        "warnings": payload.get("warnings"),
        "num_frame_summaries": len(frame_summaries) if isinstance(frame_summaries, list) else 0,
    }
    return json.dumps(digest, indent=2)


def _source_paths_for_prompt(run_dir: Path) -> dict[str, str]:
    paths = {
        "scene": run_dir / "src" / "scene.py",
        "body": run_dir / "src" / "body.py",
        "action": run_dir / "src" / "action.py",
        "rendering": run_dir / "src" / "rendering.py",
        "main": run_dir / "src" / "main.py",
    }
    return {name: str(path) for name, path in paths.items() if path.exists()}


def _asset_evidence_paths_for_prompt(run_dir: Path) -> dict[str, list[str]]:
    return {
        "asset_sources": [str(path) for path in _asset_source_paths(run_dir)],
        "asset_preview_reports": [str(path) for path in _asset_preview_report_paths(run_dir)],
        "asset_preview_images": [str(path) for path in _asset_preview_image_paths(run_dir)],
        "asset_generation_reports": [
            str(path) for path in sorted((run_dir / "assets").glob("**/xml_asset_generation_report.json"))[:8]
        ],
    }


def _write_critic_evidence_index(
    *,
    run_dir: Path,
    artifact_report: dict[str, Any],
    source_report: dict[str, Any],
) -> dict[str, Any]:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    full_artifact_report_path = reports_dir / "critic_artifact_report.json"
    full_artifact_report_path.write_text(json.dumps(artifact_report, indent=2) + "\n", encoding="utf-8")

    paths = {
        "artifact_report": full_artifact_report_path,
        "execution_report": reports_dir / "execution_report.json",
        "visual_evaluation": reports_dir / "visual_evaluation.json",
        "metrics": run_dir / "artifacts" / "metrics.json",
        "event_log": run_dir / "artifacts" / "event_log.json",
        "render_stats": run_dir / "artifacts" / "render_stats.json",
        "summary": run_dir / "artifacts" / "summary.json",
        "run_result": run_dir / "artifacts" / "run_result.json",
        "planner_output": run_dir / "contracts" / "planner_output.json",
        "timing_contract": run_dir / "contracts" / "timing.json",
        "deformable_config": run_dir / "contracts" / "deformable_config.json",
        "opt_subagent_report": reports_dir / "opt_subagent_report.json",
        "opt_report": reports_dir / "opt_report.json",
        "opt_verification": reports_dir / "verification_report.json",
        "current_opt_params": run_dir / "contracts" / "current_opt_params.json",
        "best_opt_params": run_dir / "contracts" / "best_opt_params.json",
        "asset_manifest": run_dir / "assets" / "asset_manifest.json",
        "stdout": reports_dir / "stdout.txt",
        "stderr": reports_dir / "stderr.txt",
        "source_scene": run_dir / "src" / "scene.py",
        "source_body": run_dir / "src" / "body.py",
        "source_action": run_dir / "src" / "action.py",
        "source_rendering": run_dir / "src" / "rendering.py",
        "source_main": run_dir / "src" / "main.py",
        "state_cache_source_consistency": reports_dir / "state_cache_source_consistency.json",
    }
    visual_report = load_json_object(paths["visual_evaluation"])
    contact_sheet_path = None
    sampled_frames: list[str] = []
    if isinstance(visual_report, dict):
        if isinstance(visual_report.get("contact_sheet_path"), str):
            contact_sheet_path = visual_report["contact_sheet_path"]
        if isinstance(visual_report.get("sampled_frames"), list):
            sampled_frames = [str(path) for path in visual_report["sampled_frames"] if isinstance(path, str)]
    asset_preview_images = [str(path) for path in _asset_preview_image_paths(run_dir)]
    asset_preview_reports = [str(path) for path in _asset_preview_report_paths(run_dir)]
    asset_source_paths = [str(path) for path in _asset_source_paths(run_dir)]

    index: dict[str, Any] = {
        "schema_version": 1,
        "case_workspace": str(run_dir),
        "paths": {name: str(path) for name, path in paths.items()},
        "sizes_bytes": {name: _file_size(path) for name, path in paths.items()},
        "contact_sheet_path": contact_sheet_path,
        "sampled_frames": sampled_frames,
        "asset_preview_images": asset_preview_images,
        "asset_preview_reports": asset_preview_reports,
        "asset_source_paths": asset_source_paths,
        "state_cache_source_consistency": source_report,
        "notes": [
            "Generated source is referenced by path. Read only the source files needed for source-aware review.",
            "Generated asset source and preview paths are included so asset morphology can be judged directly.",
            "Large evidence files are referenced by path so the critic can inspect them without exceeding input limits.",
            "The event log is complete on disk and should be sampled or searched as needed.",
            "State-cache hash mismatches are evidence to classify by source semantics; they are not automatically a "
            "physics-cache invalidation.",
            "If Opt was used, compare the Opt report against the current root artifacts; do not treat Opt success as "
            "final acceptance unless the rerun artifacts and source evidence are physically faithful.",
        ],
    }
    index_path = reports_dir / "critic_evidence_index.json"
    index["index_path"] = str(index_path)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def _write_state_cache_source_report(run_dir: Path) -> dict[str, Any]:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _state_cache_manifest_for_critic(run_dir)
    if manifest_path is None:
        report: dict[str, Any] = {
            "schema_version": 1,
            "manifest_path": None,
            "case_root": str(run_dir.resolve()),
            "status": "not_applicable",
            "classification_required": False,
            "cached_source_count": 0,
            "current_source_count": 0,
            "mismatch_count": 0,
            "mismatches": [],
            "provenance_errors": [],
        }
    else:
        report = build_state_cache_source_consistency_report(
            manifest_path,
            case_root=run_dir,
            diff_dir=reports_dir / "state_cache_source_diffs",
        )
    report["candidate_worker_evidence_paths"] = _source_change_worker_evidence_paths(run_dir, report)
    (reports_dir / "state_cache_source_consistency.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _source_change_worker_evidence_paths(run_dir: Path, source_report: dict[str, Any]) -> list[str]:
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        return []
    roles: set[str] = set()
    for mismatch in source_report.get("mismatches", []):
        if not isinstance(mismatch, dict):
            continue
        value = mismatch.get("path")
        if isinstance(value, str):
            role = Path(value).stem
            if role in {"scene", "body", "action", "rendering"}:
                roles.add(role)
    candidates: list[Path] = []
    for role in sorted(roles):
        candidates.extend(logs_dir.glob(f"codex_{role}*.final.json"))
    existing = [path for path in candidates if path.is_file()]
    existing.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path.resolve()) for path in existing[:12]]


def _state_cache_manifest_for_critic(run_dir: Path) -> Path | None:
    render_stats = load_json_object(run_dir / "artifacts" / "render_stats.json")
    candidates: list[str] = []
    if isinstance(render_stats, dict):
        for key in ("physics_cache_manifest", "state_cache_manifest"):
            value = render_stats.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)
        replay = render_stats.get("replay")
        if isinstance(replay, dict):
            value = replay.get("physics_cache_manifest")
            if isinstance(value, str) and value:
                candidates.append(value)
    default_manifest = run_dir / "artifacts" / "state_cache" / "manifest.json"
    resolved_candidates: list[Path] = []
    for value in candidates:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        resolved_candidates.append(candidate)
        if candidate.is_file():
            return candidate
    if resolved_candidates:
        return resolved_candidates[0]
    return default_manifest if default_manifest.is_file() else None


def _default_cache_source_consistency(source_report: dict[str, Any]) -> dict[str, Any]:
    status = str(source_report.get("status") or "not_applicable")
    mismatched_files = [
        str(item.get("path"))
        for item in source_report.get("mismatches", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    ]
    if status == "not_applicable":
        classification = "not_applicable"
    elif status == "match":
        classification = "match"
    else:
        classification = "indeterminate"
    return {
        "classification": classification,
        "mismatched_files": mismatched_files,
        "physics_rerun_required": classification == "indeterminate",
        "rationale": (
            "No state cache was used."
            if classification == "not_applicable"
            else "Cached and current source hashes match."
            if classification == "match"
            else "Source provenance requires semantic review, but no valid Critic classification was available."
        ),
        "evidence": [str(source_report.get("manifest_path"))] if source_report.get("manifest_path") else [],
    }


def _normalize_cache_source_consistency(value: Any, source_report: dict[str, Any]) -> dict[str, Any]:
    default = _default_cache_source_consistency(source_report)
    status = str(source_report.get("status") or "not_applicable")
    if status in {"not_applicable", "match"}:
        return default
    if not isinstance(value, dict):
        return default
    classification = value.get("classification")
    if classification not in {"render_only", "physics_affecting", "indeterminate"}:
        return default
    mismatched_files = default["mismatched_files"]
    rationale = value.get("rationale")
    evidence = value.get("evidence")
    return {
        "classification": classification,
        "mismatched_files": mismatched_files,
        "physics_rerun_required": classification != "render_only",
        "rationale": rationale if isinstance(rationale, str) and rationale else default["rationale"],
        "evidence": [str(item) for item in evidence] if isinstance(evidence, list) else default["evidence"],
    }


def _critic_image_paths(run_dir: Path) -> tuple[Path, ...]:
    report_path = run_dir / "reports" / "visual_evaluation.json"
    report = load_json_object(report_path) if report_path.exists() else None
    image_paths: list[Path] = []
    if isinstance(report, dict):
        contact_sheet_path = report.get("contact_sheet_path")
        if isinstance(contact_sheet_path, str):
            contact_sheet = Path(contact_sheet_path)
            if contact_sheet.is_file():
                image_paths.append(contact_sheet)
    image_paths.extend(_asset_preview_image_paths(run_dir))
    if image_paths:
        return _unique_existing_paths(image_paths, limit=12)
    if not isinstance(report, dict):
        return ()
    sampled_frames = report.get("sampled_frames")
    if isinstance(sampled_frames, list):
        for item in sampled_frames:
            if isinstance(item, str):
                image_paths.append(Path(item))
    return _unique_existing_paths(image_paths, limit=12)


def _asset_evidence_bundle(run_dir: Path) -> str:
    blocks: list[str] = []
    source_paths = _asset_source_paths(run_dir)
    if source_paths:
        blocks.append("Asset source files:")
        blocks.extend(_file_block_limited(path) for path in source_paths[:6])
    preview_reports = _asset_preview_report_paths(run_dir)
    if preview_reports:
        blocks.append("Asset preview reports:")
        blocks.extend(_file_block_limited(path, max_chars=8000) for path in preview_reports[:6])
    generation_reports = sorted((run_dir / "assets").glob("**/xml_asset_generation_report.json"))
    if generation_reports:
        blocks.append("Asset generation reports:")
        blocks.extend(_file_block_limited(path, max_chars=12000) for path in generation_reports[:4])
    if not blocks:
        return "<no generated asset source or preview evidence found>"
    return "\n\n".join(blocks)


def _asset_source_paths(run_dir: Path) -> list[Path]:
    manifest = load_json_object(run_dir / "assets" / "asset_manifest.json")
    if not isinstance(manifest, dict):
        return []
    paths: list[Path] = []
    for entry in manifest.get("assets", []):
        if not isinstance(entry, dict):
            continue
        source_type = str(entry.get("source_type") or "")
        if source_type not in {"mjcf", "urdf", "generated_xml"}:
            continue
        value = entry.get("runtime_path")
        if isinstance(value, str):
            path = Path(value)
            if path.is_file() and path.suffix.lower() in {".xml", ".urdf", ".mjcf"}:
                paths.append(path)
    return list(_unique_existing_paths(paths, limit=12))


def _asset_preview_report_paths(run_dir: Path) -> list[Path]:
    reports = list((run_dir / "assets").glob("**/preview_report.json"))
    reports.extend((run_dir / "reports" / "asset_inspection").glob("**/preview_report.json"))
    return list(_unique_existing_paths(sorted(reports), limit=12))


def _asset_preview_image_paths(run_dir: Path) -> list[Path]:
    roots = [run_dir / "assets", run_dir / "reports" / "asset_inspection"]
    preferred = ("top", "iso", "front", "side")
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for stem in preferred:
            paths.extend(sorted(root.glob(f"**/{stem}.png")))
            paths.extend(sorted(root.glob(f"**/{stem}.jpg")))
        paths.extend(sorted(root.glob("**/contact_sheet.png")))
        paths.extend(sorted(root.glob("**/contact_sheet.jpg")))
    return list(_unique_existing_paths(paths, limit=10))


def _unique_existing_paths(paths: list[Path], *, limit: int) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        unique.append(resolved)
        if len(unique) >= limit:
            break
    return tuple(unique)


def _file_block_limited(path: Path, *, max_chars: int = 30000) -> str:
    text = _read_text(path, max_chars=max_chars)
    if len(text) > max_chars:
        text = _clip_middle(text, max_chars)
    suffix = path.suffix.lstrip(".") or "text"
    return f"### {path}\n```{suffix}\n{text}\n```"


def _generated_source_bundle(run_dir: Path) -> str:
    source_paths = [
        run_dir / "src" / "scene.py",
        run_dir / "src" / "body.py",
        run_dir / "src" / "action.py",
        run_dir / "src" / "rendering.py",
        run_dir / "src" / "main.py",
    ]
    return "\n\n".join(_file_block_limited(path, max_chars=PROMPT_TEXT_LIMITS["generated_source_file"]) for path in source_paths)


def _genesis_context_pointer(run_dir: Path) -> str:
    context_md = run_dir / "contracts" / "genesis_context.md"
    context_json = run_dir / "contracts" / "genesis_context.json"
    docs_dir = "<see context JSON>"
    catalog_path = "<see context JSON>"
    payload = load_json_object(context_json)
    if isinstance(payload, dict):
        docs_dir = str(payload.get("docs_dir") or docs_dir)
        catalog_path = str(payload.get("catalog_path") or catalog_path)
    return "\n".join(
        [
            "Genesis official-doc and local-source context is available on disk for on-demand review.",
            "Inspect only the specific docs/source needed to judge the run.",
            "The full context pack is not preloaded here.",
            f"- Context index: {context_md}",
            f"- Machine-readable context JSON: {context_json}",
            f"- Cached official docs directory: {docs_dir}",
            f"- Selected official-doc catalog: {catalog_path}",
            "- Active non-rigid scope: FEM+IPC only. IPC may also be used for rigid/articulated contact when enabled.",
            "- For this critic pass, use rigid/mesh/rendering docs as needed.",
            "- Prefer local Genesis source and examples over online docs if they disagree.",
        ]
    )


def _file_block(path: Path) -> str:
    suffix = path.suffix.lstrip(".") or "text"
    return f"### {path}\n```{suffix}\n{_read_text(path, max_chars=PROMPT_TEXT_LIMITS['source_file'])}\n```"


def _read_text(path: Path, *, max_chars: int | None = None) -> str:
    if not path.exists():
        return f"<missing: {path}>"
    text = path.read_text(encoding="utf-8", errors="replace")
    return _clip_middle(text, max_chars) if max_chars is not None else text


def _clip_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = f"\n<truncated {len(text) - max_chars} chars from middle>\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:]


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None
