from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from code_agent.codex.runner import run_codex_exec
from code_agent.configs import CONFIGS
from code_agent.evaluation.simple import evaluate_run
from code_agent.execution.runner import run_generated_simulation
from code_agent.orchestration.generator import write_project
from code_agent.orchestration.integrator import write_main
from code_agent.orchestration.workers import dispatch_workers, repair_worker, write_worker_dispatch_report


@dataclass(slots=True)
class Case:
    case_id: str
    task: str


def load_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise ValueError(f"Invalid case line, expected case_id|prompt: {line}")
        case_id, task = line.split("|", 1)
        cases.append(Case(case_id.strip(), task.strip()))
    return cases


def _write_case_inputs(case_dir: Path, case: Case) -> None:
    inputs = case_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "user_prompt.md").write_text(case.task + "\n", encoding="utf-8")
    (inputs / "capabilities.md").write_text("Rigid CPU smoke implementation. Mesh and XML are represented by primitive stand-ins in this MVP.\n", encoding="utf-8")


def _generate_project(case_dir: Path, case: Case, generation_mode: str):
    if generation_mode == "fallback":
        return write_project(run_dir=case_dir, case_id=case.case_id, task=case.task)

    results = dispatch_workers(case_dir=case_dir, task=case.task)
    write_worker_dispatch_report(case_dir, results)
    return write_main(run_dir=case_dir, task=case.task)


def _failure_context(case_dir: Path, critic: dict[str, object]) -> str:
    reports_dir = case_dir / "reports"
    parts = [
        "Critic report:",
        json.dumps(critic, indent=2)[:8000],
        "stderr:",
        _read_text(reports_dir / "stderr.txt", limit=4000),
        "stdout:",
        _read_text(reports_dir / "stdout.txt", limit=4000),
    ]
    return "\n\n".join(parts)


def _read_text(path: Path, limit: int) -> str:
    if not path.exists():
        return f"<missing {path}>"
    return path.read_text(encoding="utf-8", errors="replace")[:limit]


def _maybe_run_planner(case_dir: Path, case: Case, codex_mode: str) -> dict[str, object]:
    contracts = case_dir / "contracts"
    logs = case_dir / "logs"
    contracts.mkdir(parents=True, exist_ok=True)
    planner_output = {
        "scene_brief": {"case_id": case.case_id, "task": case.task},
        "scene_plan": {"backend": "cpu", "strategy": "rigid primitive smoke"},
        "module_contracts": ["scene", "body", "action"],
    }
    if codex_mode == "off":
        (contracts / "planner_output.json").write_text(json.dumps(planner_output, indent=2) + "\n", encoding="utf-8")
        return planner_output

    prompt = (
        "Return a concise JSON planning summary for this Genesis rigid CPU smoke task. "
        "Do not edit files. Include scene_brief, scene_plan, module_contracts, risks. "
        f"Task: {case.task}"
    )
    try:
        result = run_codex_exec(
            role="planner",
            prompt=prompt,
            workdir=Path.cwd(),
            logs_dir=logs,
            sandbox=CONFIGS.codex.planner_sandbox,
            model=CONFIGS.codex.planner_model,
            timeout_sec=180.0,
        )
        planner_output["codex_planner"] = {"returncode": result.returncode, "ok": result.ok}
    except Exception as exc:
        if codex_mode == "required":
            raise
        planner_output["codex_planner"] = {"returncode": -1, "ok": False, "error": str(exc)}
    (contracts / "planner_output.json").write_text(json.dumps(planner_output, indent=2) + "\n", encoding="utf-8")
    return planner_output


def run_suite(
    *,
    tasks_file: Path,
    out_dir: Path,
    backend: str,
    max_cases: int | None,
    codex_mode: str,
    timeout_sec: float,
    generation_mode: str,
    render: bool,
    repair_rounds: int,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(tasks_file)
    if max_cases is not None:
        cases = cases[:max_cases]
    results = []
    for case in cases:
        case_dir = out_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_case_inputs(case_dir, case)
        _maybe_run_planner(case_dir, case, codex_mode)
        project = _generate_project(case_dir, case, generation_mode)
        execution = run_generated_simulation(
            main_py=project.main_py,
            run_dir=case_dir,
            backend=backend,
            timeout_sec=timeout_sec,
            render=render,
        )
        critic = evaluate_run(
            run_dir=case_dir,
            task=case.task,
            execution_ok=execution.ok,
            require_render=render,
            use_codex_critic=True,
        )

        repair_attempts = 0
        while generation_mode == "codex" and critic["verdict"] != "pass" and repair_attempts < repair_rounds:
            owner = str(critic.get("recommended_owner", "none"))
            repaired = repair_worker(
                case_dir=case_dir,
                task=case.task,
                owner=owner,
                failure_context=_failure_context(case_dir, critic),
            )
            if repaired is None:
                break
            write_worker_dispatch_report(case_dir, [repaired])
            project = write_main(run_dir=case_dir, task=case.task)
            execution = run_generated_simulation(
                main_py=project.main_py,
                run_dir=case_dir,
                backend=backend,
                timeout_sec=timeout_sec,
                render=render,
            )
            critic = evaluate_run(
                run_dir=case_dir,
                task=case.task,
                execution_ok=execution.ok,
                require_render=render,
                use_codex_critic=True,
            )
            repair_attempts += 1

        results.append(
            {
                "case_id": case.case_id,
                "execution_ok": execution.ok,
                "verdict": critic["verdict"],
                "recommended_owner": critic.get("recommended_owner", "none"),
                "repair_attempts": repair_attempts,
                "case_dir": str(case_dir),
            }
        )
    summary = {
        "tasks_file": str(tasks_file),
        "out_dir": str(out_dir),
        "backend": backend,
        "codex_mode": codex_mode,
        "generation_mode": generation_mode,
        "render": render,
        "num_cases": len(results),
        "num_passed": sum(1 for item in results if item["verdict"] == "pass"),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
