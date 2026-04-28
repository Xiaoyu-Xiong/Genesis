from __future__ import annotations

import json
import textwrap
from pathlib import Path

from code_agent.codex.runner import run_codex_exec
from code_agent.configs import CONFIGS

from . import action, body, rendering, scene
from .common import COMMON_RULES, RIGID_API_GUIDE, WorkerDispatchResult, WorkerRole, WorkerSpec

WORKERS: dict[WorkerRole, WorkerSpec] = {
    "scene": scene.SPEC,
    "body": body.SPEC,
    "action": action.SPEC,
    "rendering": rendering.SPEC,
}

PLACEHOLDER_MODULES: dict[WorkerRole, str] = {
    "scene": '''from __future__ import annotations


def create_scene(backend: str):
    raise NotImplementedError("scene worker did not replace this placeholder")
''',
    "body": '''from __future__ import annotations


def create_bodies(scene, task: str):
    raise NotImplementedError("body worker did not replace this placeholder")
''',
    "action": '''from __future__ import annotations

from pathlib import Path


def run_actions(scene, actors, *, out_dir: Path, steps: int = 40):
    raise NotImplementedError("action worker did not replace this placeholder")
''',
    "rendering": '''from __future__ import annotations

from pathlib import Path


def render_outputs(*, out_dir: Path, event_log_path: Path | None = None, metrics_path: Path | None = None):
    raise NotImplementedError("rendering worker did not replace this placeholder")
''',
}


def dispatch_workers(*, case_dir: Path, task: str, repair_context: str | None = None) -> list[WorkerDispatchResult]:
    src_dir = case_dir / "src"
    logs_dir = case_dir / "logs"
    src_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    _seed_worker_files(case_dir)

    results: list[WorkerDispatchResult] = []
    for role in ("scene", "body", "action", "rendering"):
        results.append(_run_worker(case_dir=case_dir, task=task, spec=WORKERS[role], repair_context=repair_context))
    return results


def repair_worker(*, case_dir: Path, task: str, owner: str, failure_context: str) -> WorkerDispatchResult | None:
    role = _normalize_owner(owner)
    if role is None:
        return None
    return _run_worker(case_dir=case_dir, task=task, spec=WORKERS[role], repair_context=failure_context)


def write_worker_dispatch_report(case_dir: Path, results: list[WorkerDispatchResult]) -> None:
    report = {
        "workers": [
            {
                "role": item.role,
                "ok": item.ok,
                "target_path": str(item.target_path),
                "returncode": item.codex_result.returncode,
                "duration_sec": item.codex_result.duration_sec,
                "final_message_path": str(item.codex_result.final_message_path),
                "worker_status": item.worker_report.get("status") if item.worker_report else None,
                "exports": item.worker_report.get("exports") if item.worker_report else [],
                "error_message": item.error_message,
            }
            for item in results
        ]
    }
    path = case_dir / "reports" / "worker_dispatch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _seed_worker_files(case_dir: Path) -> None:
    for role, spec in WORKERS.items():
        path = case_dir / spec.target_file
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(PLACEHOLDER_MODULES[role], encoding="utf-8")


def _run_worker(
    *,
    case_dir: Path,
    task: str,
    spec: WorkerSpec,
    repair_context: str | None,
) -> WorkerDispatchResult:
    target_path = case_dir / spec.target_file
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = case_dir / "logs"
    schema_path = Path("code_agent/specs/worker_report.schema.json")
    prompt = _worker_prompt(case_dir=case_dir, task=task, spec=spec, repair_context=repair_context)
    result = run_codex_exec(
        role=spec.role if repair_context is None else f"{spec.role}_repair",
        prompt=prompt,
        workdir=Path.cwd(),
        logs_dir=logs_dir,
        sandbox=CONFIGS.codex.worker_sandbox,
        model=CONFIGS.codex.worker_model,
        output_schema=schema_path,
        timeout_sec=600.0,
    )
    worker_report, error_message = _parse_worker_report(result.final_message_path)
    source_code = worker_report.get("source_code") if worker_report else None
    if isinstance(source_code, str) and source_code.strip():
        target_path.write_text(source_code.rstrip() + "\n", encoding="utf-8")
    ok = (
        result.ok
        and worker_report is not None
        and worker_report.get("status") == "completed"
        and isinstance(source_code, str)
        and spec.required_export in source_code
        and target_path.exists()
        and target_path.stat().st_size > 0
        and "NotImplementedError" not in target_path.read_text(encoding="utf-8", errors="replace")
    )
    return WorkerDispatchResult(
        role=spec.role,
        ok=ok,
        target_path=target_path,
        codex_result=result,
        worker_report=worker_report,
        error_message=error_message,
    )


def _worker_prompt(*, case_dir: Path, task: str, spec: WorkerSpec, repair_context: str | None) -> str:
    mode = "repair the existing module source" if repair_context else "author the module source"
    return textwrap.dedent(
        f"""
        {COMMON_RULES}

        Task prompt:
        {task}

        Role: {spec.role} worker
        Responsibility: {spec.responsibility}
        Mode: {mode}
        Exact target file: {case_dir / spec.target_file}
        Allowed write paths: none. Do not edit files. Return source code in JSON only.
        Required export: `{spec.required_export}`
        Required `source_code`: complete Python content that the coordinator can write to `{spec.target_file}`.

        {RIGID_API_GUIDE}

        Role-specific instructions:
        {textwrap.dedent(spec.prompt_body).strip()}

        {"Failure context to fix:" if repair_context else ""}
        {repair_context or ""}

        Final response must be JSON matching worker_report.schema.json.
        Set status to `completed` only when `source_code` contains the full replacement module.
        Set `commands_run` to an empty list if you did not run commands.
        Include changed_files with exactly `{spec.target_file}` when completed, even though the coordinator writes it.
        """
    ).strip()


def _parse_worker_report(path: Path) -> tuple[dict[str, object] | None, str | None]:
    if not path.exists():
        return None, f"missing final message: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid worker JSON: {exc}"
    if not isinstance(data, dict):
        return None, "worker final message is not a JSON object"
    return data, None


def _normalize_owner(owner: str) -> WorkerRole | None:
    lowered = owner.strip().lower()
    if lowered in WORKERS:
        return lowered  # type: ignore[return-value]
    if lowered == "render":
        return "rendering"
    return None
