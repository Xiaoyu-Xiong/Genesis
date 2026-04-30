from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from code_agent.context.genesis import build_genesis_context_pack, install_genesis_context_pack
from code_agent.io_utils import dump_json
from code_agent.planner.session import PlannerSession, PlannerSessionConfig


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
    (inputs / "capabilities.md").write_text(
        "Genesis local GPU code generation for rigid, articulated, FEM+IPC deformable, mesh, texture, rendering, "
        "critic, and repair workflows. Other non-rigid families are out of scope. Planner controls worker dispatch, "
        "execution, critic, and repair actions.\n",
        encoding="utf-8",
    )


def run_suite(
    *,
    tasks_file: Path,
    out_dir: Path,
    backend: str,
    max_cases: int | None,
    timeout_sec: float,
    render: bool,
    repair_rounds: int,
    steps: int | None = None,
    duration_sec: float | None = None,
    render_fps: int | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(tasks_file)
    if max_cases is not None:
        cases = cases[:max_cases]

    genesis_context = build_genesis_context_pack(out_dir)
    results: list[dict[str, object]] = []
    for case in cases:
        case_dir = out_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_case_inputs(case_dir, case)
        install_genesis_context_pack(case_dir, genesis_context)
        session = PlannerSession(
            PlannerSessionConfig(
                case_id=case.case_id,
                task=case.task,
                case_dir=case_dir,
                backend=backend,
                timeout_sec=timeout_sec,
                render=render,
                repair_rounds=repair_rounds,
                steps=steps,
                duration_sec=duration_sec,
                render_fps=render_fps,
            )
        )
        results.append(session.run())

    summary = {
        "tasks_file": str(tasks_file),
        "out_dir": str(out_dir),
        "backend": backend,
        "render": render,
        "steps": steps,
        "duration_sec": duration_sec,
        "render_fps": render_fps,
        "genesis_context_path": str(genesis_context.markdown_path),
        "genesis_context_json_path": str(genesis_context.json_path),
        "genesis_docs_dir": str(genesis_context.docs_dir),
        "num_cases": len(results),
        "num_passed": sum(1 for item in results if item["verdict"] == "pass"),
        "results": results,
    }
    dump_json(summary, out_dir / "summary.json")
    return summary
