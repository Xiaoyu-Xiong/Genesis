from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from code_agent.utils.codex import CodexExecRequest, run_codex_exec
from code_agent.configs import CONFIGS


def run_codex_critic(*, run_dir: Path, task: str, artifact_report: dict[str, Any]) -> dict[str, Any]:
    reports_dir = run_dir / "reports"
    logs_dir = run_dir / "logs"
    prompt = _critic_prompt(run_dir=run_dir, task=task, artifact_report=artifact_report)
    image_paths = _critic_image_paths(run_dir)
    result = run_codex_exec(
        CodexExecRequest(
            role="critic",
            prompt=prompt,
            cwd=Path.cwd(),
            sandbox=CONFIGS.codex.critic_sandbox,
            model=CONFIGS.codex.critic_model,
            output_schema_path=Path("code_agent/specs/critic_report.schema.json"),
            image_paths=image_paths,
            output_jsonl_path=logs_dir / "codex_critic.jsonl",
            final_message_path=logs_dir / "codex_critic.final.json",
            timeout_sec=300.0,
        )
    )
    report = _load_json(Path(result.final_message_path))
    if report is None:
        report = {
            "verdict": "inconclusive",
            "score": 0.0,
            "observations": ["Codex critic did not return parseable JSON."],
            "failure_modes": ["critic.parse_failed"],
            "recommended_owner": "none",
            "repair_summary": None,
            "evidence": {"metrics": [], "frames": [], "video": None, "event_logs": []},
        }
    report["codex_result"] = {
        "returncode": result.exit_code,
        "ok": result.success,
        "duration_sec": result.duration_sec,
        "final_message_path": result.final_message_path,
        "stderr_path": result.stderr_path,
    }
    (reports_dir / "codex_critic_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _critic_prompt(*, run_dir: Path, task: str, artifact_report: dict[str, Any]) -> str:
    metrics = _read_text(run_dir / "artifacts" / "metrics.json")
    event_log = _read_text(run_dir / "artifacts" / "event_log.json")
    render_stats = _read_text(run_dir / "artifacts" / "render_stats.json")
    visual_evaluation = _read_text(run_dir / "reports" / "visual_evaluation.json")
    execution_report = _read_text(run_dir / "reports" / "execution_report.json")
    generated_source = _generated_source_bundle(run_dir)
    planner_output = _read_text(run_dir / "contracts" / "planner_output.json")
    timing_contract = _read_text(run_dir / "contracts" / "timing.json")
    asset_manifest = _read_text(run_dir / "assets" / "asset_manifest.json")
    genesis_context = _genesis_context_pointer(run_dir)
    summary = _read_text(run_dir / "artifacts" / "summary.json")
    run_result = _read_text(run_dir / "artifacts" / "run_result.json")
    stdout = _read_text(run_dir / "reports" / "stdout.txt")
    stderr = _read_text(run_dir / "reports" / "stderr.txt")
    return textwrap.dedent(
        f"""
        You are the single-pass Codex Critic for a generated Genesis rigid or rigid-mesh simulation.
        The full repository and current case workspace are available for read-only context. You may inspect additional
        source, contracts, reports, logs, assets, and artifacts with read-only commands if needed. Do not edit files.
        Read the supplied evidence, inspect the attached render/contact-sheet image when present, and return JSON only.

        Original task prompt:
        {task}

        Case workspace:
        {run_dir}

        Artifact evaluation report:
        {json.dumps(artifact_report, indent=2)}

        Execution report:
        {execution_report}

        Metrics:
        {metrics}

        Event log:
        {event_log}

        Render stats:
        {render_stats}

        Visual evidence:
        {visual_evaluation}

        Planner output:
        {planner_output}

        Timing contract:
        {timing_contract}

        Asset manifest:
        {asset_manifest}

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

        Decide whether the run passes as a generated rigid simulation result. Compare the original task prompt,
        generated source, execution artifacts, metrics, event logs, render stats, and visual evidence. Prioritize
        execution correctness, required artifacts, plausible movement, physically coherent staging, and whether the
        visual evidence matches the task. The output should not merely satisfy numeric proxies; while staying faithful
        to the text prompt, it should look reasonable, logical, and visually coherent.

        When sampled frame paths, contact sheets, texture summaries, or texture-presence warnings are available, use
        them as review evidence alongside numeric metrics instead of relying only on event logs. If meshes or textures
        are involved, check whether orientation, scale, material binding, and rendered texture appearance are consistent
        with the source and manifest.

        If repair is needed, make `repair_summary` detailed, source-aware, and directly actionable. Name the owner
        module, the concrete behavior that is wrong, the evidence proving it, the likely source-level cause, and the
        changes that should be made. Avoid vague advice; give enough detail that the next Planner request can instruct
        the responsible worker precisely. Do not compress important source-level feedback just to keep the answer short.

        Return JSON matching critic_report.schema.json. `recommended_owner` must be one of:
        planner, scene, body, action, rendering, integrator, execution, none.
        Use `needs_repair` when there is a clear owner-routed fix.
        """
    ).strip()


def _critic_image_paths(run_dir: Path) -> tuple[Path, ...]:
    report_path = run_dir / "reports" / "visual_evaluation.json"
    if not report_path.exists():
        return ()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ()
    image_paths: list[Path] = []
    contact_sheet_path = report.get("contact_sheet_path")
    if isinstance(contact_sheet_path, str):
        contact_sheet = Path(contact_sheet_path)
        if contact_sheet.is_file():
            image_paths.append(contact_sheet)
    if image_paths:
        return tuple(image_paths)
    sampled_frames = report.get("sampled_frames")
    if isinstance(sampled_frames, list):
        for item in sampled_frames:
            if isinstance(item, str):
                frame_path = Path(item)
                if frame_path.is_file():
                    image_paths.append(frame_path)
    return tuple(image_paths)


def _generated_source_bundle(run_dir: Path) -> str:
    source_paths = [
        run_dir / "src" / "scene.py",
        run_dir / "src" / "body.py",
        run_dir / "src" / "action.py",
        run_dir / "src" / "rendering.py",
        run_dir / "src" / "main.py",
    ]
    return "\n\n".join(_file_block(path) for path in source_paths)


def _genesis_context_pointer(run_dir: Path) -> str:
    context_md = run_dir / "contracts" / "genesis_context.md"
    context_json = run_dir / "contracts" / "genesis_context.json"
    docs_dir = "<see context JSON>"
    catalog_path = "<see context JSON>"
    payload = _load_json(context_json)
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
            "- Active non-rigid scope: FEM+IPC only. For this critic pass, use rigid/mesh/rendering docs as needed.",
            "- Prefer local Genesis source and examples over online docs if they disagree.",
        ]
    )


def _file_block(path: Path) -> str:
    suffix = path.suffix.lstrip(".") or "text"
    return f"### {path}\n```{suffix}\n{_read_text(path)}\n```"


def _read_text(path: Path) -> str:
    if not path.exists():
        return f"<missing: {path}>"
    text = path.read_text(encoding="utf-8", errors="replace")
    return text


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
