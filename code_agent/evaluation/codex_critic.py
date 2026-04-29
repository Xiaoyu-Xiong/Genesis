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
    result = run_codex_exec(
        CodexExecRequest(
            role="critic",
            prompt=prompt,
            cwd=Path.cwd(),
            sandbox=CONFIGS.codex.critic_sandbox,
            model=CONFIGS.codex.critic_model,
            output_schema_path=Path("code_agent/specs/critic_report.schema.json"),
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
    event_log = _read_text(run_dir / "artifacts" / "event_log.json", limit=8000)
    render_stats = _read_text(run_dir / "artifacts" / "render_stats.json", limit=8000)
    execution_report = _read_text(run_dir / "reports" / "execution_report.json", limit=8000)
    return textwrap.dedent(
        f"""
        You are the single-pass Codex Critic for a generated Genesis rigid primitive simulation.
        Do not edit files. Do not run commands. Read the supplied evidence and return JSON only.

        Task:
        {task}

        Artifact evaluation report:
        {json.dumps(artifact_report, indent=2)}

        Execution report:
        {execution_report}

        Metrics:
        {metrics}

        Event log excerpt:
        {event_log}

        Render stats:
        {render_stats}

        Decide whether the run passes as a generated rigid primitive result. Prioritize execution correctness, required
        artifacts, plausible movement, and whether render evidence exists when available.

        Return JSON matching critic_report.schema.json. `recommended_owner` must be one of:
        planner, scene, body, action, rendering, integrator, execution, none.
        Use `needs_repair` when there is a clear owner-routed fix.
        """
    ).strip()


def _read_text(path: Path, limit: int = 12000) -> str:
    if not path.exists():
        return f"<missing: {path}>"
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:limit]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
