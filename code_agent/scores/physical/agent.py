from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from code_agent.configs import CONFIGS
from code_agent.io_utils import dump_json, load_json_object
from code_agent.scores.physical.evidence import prepare_evidence
from code_agent.scores.physical.report import failed_report, normalize_report
from code_agent.utils.codex import DEFAULT_REPO_ROOT, CodexExecRequest, run_codex_exec

__all__ = ["PhysicalScoreRequest", "run_physical_score"]

SCHEMA_PATH = Path("code_agent/scores/physical/sbar_report.schema.json")
DEFAULT_REPORT_NAME = "physical_score_report.json"


@dataclass(slots=True, frozen=True)
class PhysicalScoreRequest:
    """Request for one SBAR-v1 physical prompt-alignment score."""

    run_dir: Path
    prompt: str | None = None
    prompt_file: Path | None = None
    code_root: Path | None = None
    case_id: str | None = None
    output_path: Path | None = None
    model: str | None = None
    timeout_sec: float | None = None
    force: bool = False


def run_physical_score(request: PhysicalScoreRequest) -> dict[str, Any]:
    """Run the unified SBAR-v1 Codex scorer on one generated simulation folder.

    The scorer does not execute the simulation. It inspects the supplied prompt,
    source tree, metrics/logs, and rendered visual evidence, then writes a
    structured report under ``reports/physical_score_report.json`` by default.
    """

    run_dir = request.run_dir.resolve()
    code_root = (request.code_root or run_dir).resolve()
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    output_path = (request.output_path or reports_dir / DEFAULT_REPORT_NAME).resolve()
    if output_path.exists() and not request.force:
        cached = load_json_object(output_path)
        if isinstance(cached, dict):
            return _normalize_cached_report(
                cached,
                request=request,
                run_dir=run_dir,
                code_root=code_root,
                output_path=output_path,
            )

    prompt = _resolve_prompt(run_dir=run_dir, prompt=request.prompt, prompt_file=request.prompt_file)
    case_id = request.case_id or _infer_case_id(run_dir)
    evidence = prepare_evidence(run_dir=run_dir, code_root=code_root, case_id=case_id)
    prompt_text = build_sbar_prompt(
        prompt=prompt,
        run_dir=run_dir,
        code_root=code_root,
        case_id=case_id,
        evidence=evidence.index,
    )

    result = run_codex_exec(
        CodexExecRequest(
            role="physical_score",
            prompt=prompt_text,
            cwd=DEFAULT_REPO_ROOT,
            sandbox=CONFIGS.codex.critic_sandbox,
            model=request.model or CONFIGS.codex.critic_model,
            output_schema_path=SCHEMA_PATH,
            image_paths=tuple(evidence.image_paths),
            output_jsonl_path=logs_dir / "codex_physical_score.jsonl",
            final_message_path=logs_dir / "codex_physical_score.final.json",
            timeout_sec=request.timeout_sec or CONFIGS.codex.critic_timeout_sec,
            writable_roots=(run_dir,),
            hide_builtin_assets=False,
        )
    )
    raw_report = load_json_object(Path(result.final_message_path))
    if not isinstance(raw_report, dict):
        report = failed_report(
            prompt=prompt,
            run_dir=run_dir,
            code_root=code_root,
            case_id=case_id,
            evidence=evidence.index,
            result=result.to_dict(),
            reason=result.error_message or result.error_type or "physical_score_parse_failed",
        )
    else:
        report = normalize_report(
            raw_report,
            prompt=prompt,
            run_dir=run_dir,
            code_root=code_root,
            case_id=case_id,
            evidence=evidence.index,
            codex_result=result.to_dict(),
        )
    dump_json(report, output_path)
    return report


def _resolve_prompt(*, run_dir: Path, prompt: str | None, prompt_file: Path | None) -> str:
    if prompt is not None:
        return prompt.strip()
    if prompt_file is not None:
        return prompt_file.resolve().read_text(encoding="utf-8").strip()
    for path in (run_dir / "inputs" / "user_prompt.md", run_dir / "prompt.txt", run_dir / "prompt.md"):
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace").strip()
    for path in (run_dir / "summary.json", run_dir / "reports" / "critic_report.json"):
        payload = load_json_object(path)
        if isinstance(payload, dict):
            task = payload.get("task") or payload.get("prompt")
            if isinstance(task, str) and task.strip():
                return task.strip()
    raise ValueError(
        f"Could not infer prompt for {run_dir}. Pass --prompt or --prompt-file, or provide inputs/user_prompt.md."
    )


def _infer_case_id(run_dir: Path) -> str:
    payload = load_json_object(run_dir / "summary.json")
    if isinstance(payload, dict) and isinstance(payload.get("case_id"), str):
        return payload["case_id"]
    return run_dir.name


def _normalize_cached_report(
    cached: dict[str, Any],
    *,
    request: PhysicalScoreRequest,
    run_dir: Path,
    code_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not isinstance(cached.get("rubric"), dict):
        cached.setdefault("scorer_status", "cached")
        return cached

    prompt = cached.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = _resolve_prompt(run_dir=run_dir, prompt=request.prompt, prompt_file=request.prompt_file)
    case_id = request.case_id or cached.get("case_id") or _infer_case_id(run_dir)
    evidence = _cached_evidence(cached)
    codex_result = cached.get("codex_result")
    if not isinstance(codex_result, dict):
        codex_result = {"success": True, "reused_from_cache": True}
    report = normalize_report(
        cached,
        prompt=prompt,
        run_dir=run_dir,
        code_root=code_root,
        case_id=str(case_id),
        evidence=evidence,
        codex_result=codex_result,
    )
    report["scorer_status"] = "cached"
    dump_json(report, output_path)
    return report


def _cached_evidence(report: dict[str, Any]) -> dict[str, Any]:
    index_path = report.get("evidence_index_path")
    if isinstance(index_path, str):
        payload = load_json_object(Path(index_path))
        if isinstance(payload, dict):
            return payload
    if "invalid_no_run_or_no_video" in report.get("caps_applied", []):
        return {}
    return {"video_paths": ["cached"]}


def build_sbar_prompt(
    *,
    prompt: str,
    run_dir: Path,
    code_root: Path,
    case_id: str | None,
    evidence: dict[str, Any],
) -> str:
    return f"""
You are the single unified SBAR-v1 physical prompt-alignment evaluator.

Evaluate one generated physics simulation output against the original user prompt.
Do not modify files. Do not run expensive simulations, training, rendering, or network calls. You may read source files,
JSON reports, logs, metrics, and visual artifacts from disk. You may run lightweight read-only shell commands such as
ls, sed, rg, file, and json inspection commands if they help.

Original prompt:
{prompt}

Case id: {case_id or "<unknown>"}
Run/output folder: {run_dir}
Code root: {code_root}

Evidence index:
{json.dumps(evidence, indent=2, ensure_ascii=False)}

Metric definition:
- Metric name: SBAR-v1.
- Categories: scene, body, action, render.
- Final score before caps is:
  overall = 0.20 * scene_score + 0.275 * body_score + 0.275 * action_score + 0.25 * render_score - violation_penalty
- render_score must be:
  render_score = 0.70 * render_faithfulness + 0.30 * render_aesthetic_quality
- Each category score is 0 to 100.
- Positive rubric item scores must be exactly one of 1, 0.5, or 0:
  1 means clearly satisfied by code, artifacts, metrics, or visual evidence.
  0.5 means partially satisfied or plausible but weakly evidenced.
  0 means absent or contradicted.
- Forbidden items use violation=true or false.
- Violation penalty must be:
  40 * weighted_forbidden_violation_fraction, or 0 when no forbidden items exist.
- Apply these exact cap labels when relevant:
  invalid_no_run_or_no_video: final overall cannot exceed 20 when there is no usable execution/render evidence.
  not_real_simulation: final overall cannot exceed 40 when the result is mostly a pre-render, pure keyframe animation,
    shader fake, or hard-coded visual effect instead of a physics simulation.
  severe_forbidden_violation: final overall cannot exceed 60 when a prompt's explicit "do not" constraint is seriously
    violated.

Rubric construction rules:
- First split the prompt into atomic requirements and assign each atom to scene, body, action, render, or forbidden.
- Use weights from 1 to 5. Weight 5 is critical, 3 is important, 1 is minor.
- Scene covers environment, solver/backend, global physics parameters, gravity, world layout, floor/walls/containers,
  global contact/friction/collision settings, and background scene setup.
- Body covers objects, assets, geometry, counts, material/color/texture, physical body type, object-level physical
  parameters, and asset morphology/fidelity.
- Action covers time-varying behavior, control, release/drag/rotation/launch, collisions, contact, deformation,
  sliding/rolling/settling, event ordering, final dynamic state, and whether motion comes from real simulation rather
  than teleport/keyframes.
- Render covers target duration, video/frame availability, camera framing, visibility of important objects/actions,
  lighting, clarity, material readability, visual style requested by the prompt, and aesthetic quality/composition.
- Render faithfulness is about whether the requested content and behavior are visible.
- Render aesthetic quality is about composition, lighting, clarity, style coherence, polish, and visual appeal.

Evidence rules:
- Prefer direct code evidence and runtime artifacts over comments or self-reported metadata.
- Do not require our internal scene.py/body.py/action.py/rendering.py layout. This metric must work for other baselines.
  Map arbitrary code organization into the conceptual SBAR categories.
- Use the visual contact sheet and frame paths for render/style/visibility judgments.
- Use metrics, traces, event logs, execution reports, and source code for physical and action judgments.
- If the code references generated assets, inspect manifests/previews/source files when available.
- Penalize missing evidence, but do not invent failures when the prompt does not ask for a feature.

Return JSON matching sbar_report.schema.json. Keep evidence strings concise and cite concrete paths, metrics, code
patterns, or visual observations when possible. Ensure all numeric scores obey the formula above; the wrapper will
recompute and normalize them from your rubric.
""".strip()
