from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from code_agent.configs import CONFIGS
from code_agent.io_utils import load_json_object
from code_agent.utils.codex import DEFAULT_REPO_ROOT, CodexExecRequest, run_codex_exec
from code_agent.prompts.common import SOURCE_AWARE_REPAIR_GUIDE
from code_agent.prompts.critic import (
    CRITIC_ASSET_EVALUATION_GUIDE,
    CRITIC_DECISION_GUIDE,
    CRITIC_EVIDENCE_READING_GUIDE,
    CRITIC_GENERAL_RULES,
    CRITIC_VISUAL_EVIDENCE_GUIDE,
)

PROMPT_TEXT_LIMITS = {
    "execution_report": 120_000,
    "generated_source_file": 60_000,
    "source_file": 50_000,
    "json_report": 50_000,
    "stdout_stderr": 40_000,
}


def run_codex_critic(*, run_dir: Path, task: str, artifact_report: dict[str, Any]) -> dict[str, Any]:
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"
    prompt = _critic_prompt(run_dir=run_dir, task=task, artifact_report=artifact_report)
    image_paths = _critic_image_paths(run_dir)
    result = run_codex_exec(
        CodexExecRequest(
            role="critic",
            prompt=prompt,
            cwd=DEFAULT_REPO_ROOT,
            sandbox=CONFIGS.codex.critic_sandbox,
            model=CONFIGS.codex.critic_model,
            output_schema_path=Path("code_agent/specs/critic_report.schema.json"),
            image_paths=image_paths,
            output_jsonl_path=logs_dir / "codex_critic.jsonl",
            final_message_path=logs_dir / "codex_critic.final.json",
            timeout_sec=CONFIGS.codex.critic_timeout_sec,
        )
    )
    report = load_json_object(Path(result.final_message_path))
    if report is None:
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
            "evidence": {"metrics": [], "frames": [], "video": None, "event_logs": []},
        }
    report["codex_result"] = {
        "returncode": result.exit_code,
        "ok": result.success,
        "duration_sec": result.duration_sec,
        "final_message_path": result.final_message_path,
        "stderr_path": result.stderr_path,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }
    (reports_dir / "codex_critic_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _critic_prompt(*, run_dir: Path, task: str, artifact_report: dict[str, Any]) -> str:
    evidence_index = _write_critic_evidence_index(run_dir=run_dir, artifact_report=artifact_report)
    metrics = _read_text(run_dir / "artifacts" / "metrics.json")
    render_stats = _read_text(run_dir / "artifacts" / "render_stats.json")
    visual_evaluation = _read_text(run_dir / "reports" / "visual_evaluation.json")
    execution_report = _read_text(run_dir / "reports" / "execution_report.json", max_chars=PROMPT_TEXT_LIMITS["execution_report"])
    generated_source = _generated_source_bundle(run_dir)
    planner_output = _read_text(run_dir / "contracts" / "planner_output.json")
    timing_contract = _read_text(run_dir / "contracts" / "timing.json")
    deformable_config = _read_text(run_dir / "contracts" / "deformable_config.json")
    asset_manifest = _read_text(run_dir / "assets" / "asset_manifest.json")
    asset_evidence = _asset_evidence_bundle(run_dir)
    genesis_context = _genesis_context_pointer(run_dir)
    summary = _read_text(run_dir / "artifacts" / "summary.json")
    run_result = _read_text(run_dir / "artifacts" / "run_result.json")
    stdout = _read_text(run_dir / "reports" / "stdout.txt", max_chars=PROMPT_TEXT_LIMITS["stdout_stderr"])
    stderr = _read_text(run_dir / "reports" / "stderr.txt", max_chars=PROMPT_TEXT_LIMITS["stdout_stderr"])
    return textwrap.dedent(
        f"""
        {CRITIC_GENERAL_RULES}

        Original task prompt:
        {task}

        Case workspace:
        {run_dir}

        Evidence index:
        {json.dumps(evidence_index, indent=2)}

        {CRITIC_EVIDENCE_READING_GUIDE}

        Execution report:
        {execution_report}

        Metrics:
        {metrics}

        Event log:
        Full event log is available at {run_dir / "artifacts" / "event_log.json"}.

        Render stats:
        {render_stats}

        Visual evidence:
        {visual_evaluation}

        Planner output:
        {planner_output}

        Timing contract:
        {timing_contract}

        FEM/IPC capability/config contract:
        {deformable_config}

        Asset manifest:
        {asset_manifest}

        Generated asset source and preview evidence:
        {asset_evidence}

        Genesis documentation and local-code context:
        {genesis_context}

        Generated source:
        {generated_source}

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


def _write_critic_evidence_index(*, run_dir: Path, artifact_report: dict[str, Any]) -> dict[str, Any]:
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
        "asset_manifest": run_dir / "assets" / "asset_manifest.json",
        "stdout": reports_dir / "stdout.txt",
        "stderr": reports_dir / "stderr.txt",
        "source_scene": run_dir / "src" / "scene.py",
        "source_body": run_dir / "src" / "body.py",
        "source_action": run_dir / "src" / "action.py",
        "source_rendering": run_dir / "src" / "rendering.py",
        "source_main": run_dir / "src" / "main.py",
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
        "notes": [
            "Generated source is also inlined in the critic prompt for source-aware review.",
            "Generated asset source and preview paths are included so asset morphology can be judged directly.",
            "Large evidence files are referenced by path so the critic can inspect them without exceeding input limits.",
            "The event log is complete on disk and should be sampled or searched as needed.",
        ],
    }
    index_path = reports_dir / "critic_evidence_index.json"
    index["index_path"] = str(index_path)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


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
