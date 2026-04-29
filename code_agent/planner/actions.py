from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from code_agent.configs import CONFIGS
from code_agent.evaluation.runner import evaluate_generated_run
from code_agent.io_utils import dump_json
from code_agent.utils.execution import run_generated_simulation
from code_agent.utils.integrator import write_main
from code_agent.utils.timing import resolve_timing
from code_agent.writer.dispatcher import dispatch_worker_roles, repair_worker, write_worker_dispatch_report
from code_agent.writer.common import WorkerRole

WORKER_ROLES: tuple[WorkerRole, ...] = ("scene", "body", "action", "rendering")


class EpisodeActionExecutor:
    """Execute planner-selected actions against one episode session."""

    def __init__(self, session: Any):
        self.session = session

    def execute(self, action: dict[str, Any], turn: int) -> dict[str, Any]:
        name = action.get("action")
        try:
            if name == "write_plan":
                return self._handle_write_plan(action)
            if name == "spawn_workers":
                return self._handle_spawn_workers(action)
            if name == "run_integrator":
                return self._handle_run_integrator()
            if name == "run_execution":
                return self._handle_run_execution(action)
            if name == "run_critic":
                return self._handle_run_critic()
            if name == "request_repair":
                return self._handle_request_repair(action)
            if name == "run_python":
                return self._handle_run_command(action, turn, label="python", executable=("uv", "run", "python"))
            if name == "run_pytest":
                return self._handle_run_command(action, turn, label="pytest", executable=("uv", "run", "pytest"))
            if name == "finish":
                return self._handle_finish(action)
        except Exception as exc:  # noqa: BLE001 - convert tool exceptions into planner-visible observations.
            return {"ok": False, "status": "error", "message": f"{type(exc).__name__}: {exc}"}
        return {"ok": False, "status": "invalid_action", "message": f"Unsupported action: {name!r}"}

    def _handle_write_plan(self, action: dict[str, Any]) -> dict[str, Any]:
        planner_output = action.get("planner_output")
        if not isinstance(planner_output, dict):
            return {"ok": False, "status": "invalid_action", "message": "write_plan requires planner_output object."}
        errors = self.session.validate_json_schema(planner_output, Path("code_agent/specs/planner_output.schema.json"))
        if errors:
            return {
                "ok": False,
                "status": "invalid_planner_output",
                "message": "planner_output failed schema validation.",
                "errors": errors[:8],
            }
        planner_output_path = self.session.contracts_dir / "planner_output.json"
        dump_json(planner_output, planner_output_path)
        timing = resolve_timing(
            planner_output=planner_output,
            steps=self.session.config.steps,
            duration_sec=self.session.config.duration_sec,
            render_fps=self.session.config.render_fps,
        )
        self.session.timing = timing
        dump_json(timing.to_dict(), self.session.contracts_dir / "timing.json")
        self.session.state["planner_output_path"] = str(planner_output_path)
        self.session.state["timing"] = timing.to_dict()
        episode_plan_path = self.session.contracts_dir / "episode_plan.json"
        dump_json(
            {
                "planner_output_path": str(planner_output_path),
                "planner_output": planner_output,
                "timing": timing.to_dict(),
                "rationale": action.get("rationale"),
                "created_at_unix": time.time(),
            },
            episode_plan_path,
        )
        self.session.state["episode_plan_path"] = str(episode_plan_path)
        return {
            "ok": True,
            "status": "planned",
            "message": "Planner output and timing were accepted.",
            "planner_output_path": str(planner_output_path),
            "episode_plan_path": str(episode_plan_path),
            "timing": timing.to_dict(),
        }

    def _handle_spawn_workers(self, action: dict[str, Any]) -> dict[str, Any]:
        planner_output = self.session.current_planner_output()
        if planner_output is None:
            return {"ok": False, "status": "precondition_failed", "message": "planner_output is missing."}
        roles = self._roles_from_action(action)
        if not roles:
            return {"ok": False, "status": "invalid_action", "message": "spawn_workers requires at least one role."}
        results = dispatch_worker_roles(
            case_dir=self.session.case_dir,
            task=self.session.config.task,
            planner_output=planner_output,
            roles=roles,
            repair_context=action.get("repair_brief") if isinstance(action.get("repair_brief"), str) else None,
        )
        self.session.record_worker_results(results)
        write_worker_dispatch_report(self.session.case_dir, results)
        all_ok = self.session.all_workers_ok()
        if all_ok:
            self.session.state["control"]["needs_integration"] = True
            self.session.state["control"]["needs_execution"] = False
            self.session.state["control"]["needs_critic"] = False
        active_parallelism = min(len(roles), CONFIGS.harness.max_parallel_workers)
        return {
            "ok": all(item.ok for item in results),
            "status": "workers_dispatched",
            "roles": list(roles),
            "parallel": active_parallelism > 1,
            "max_parallel_workers": active_parallelism,
            "all_workers_ok": all_ok,
        }

    def _handle_run_integrator(self) -> dict[str, Any]:
        if not self.session.all_workers_ok():
            return {"ok": False, "status": "precondition_failed", "message": "All workers must be ok before integration."}
        timing = self.session.current_timing()
        main_py = write_main(
            run_dir=self.session.case_dir,
            task=self.session.config.task,
            default_steps=timing.steps,
            default_render_fps=timing.render_fps,
            default_duration_sec=timing.duration_sec,
            default_target_video_frames=timing.target_video_frames,
        )
        self.session.state["integration"] = {
            "ok": True,
            "main_py": str(main_py),
            "updated_at_unix": time.time(),
        }
        self.session.state["control"]["needs_integration"] = False
        self.session.state["control"]["needs_execution"] = True
        self.session.state["control"]["needs_critic"] = False
        return {"ok": True, "status": "integrated", "main_py": str(main_py)}

    def _handle_run_execution(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.session.state.get("integration") is None:
            return {"ok": False, "status": "precondition_failed", "message": "integration is missing."}
        timing = self.session.current_timing()
        main_py = Path(str(self.session.state["integration"]["main_py"]))
        backend = str(action.get("backend") or self.session.config.backend)
        render = self.session.config.render if action.get("render") is None else bool(action.get("render"))
        timeout_sec = float(action.get("timeout_sec") or self.session.config.timeout_sec)
        execution = run_generated_simulation(
            main_py=main_py,
            run_dir=self.session.case_dir,
            backend=backend,
            timeout_sec=timeout_sec,
            render=render,
            steps=timing.steps,
            render_fps=timing.render_fps,
            duration_sec=timing.duration_sec,
            target_video_frames=timing.target_video_frames,
        )
        execution_report = execution.to_dict()
        self.session.state["execution"] = self.session.execution_excerpt(execution_report)
        self.session.state["control"]["needs_execution"] = False
        self.session.state["control"]["needs_critic"] = True
        return {"ok": execution.ok, "status": "executed", "execution": self.session.state["execution"]}

    def _handle_run_critic(self) -> dict[str, Any]:
        execution = self.session.state.get("execution")
        if not isinstance(execution, dict):
            return {"ok": False, "status": "precondition_failed", "message": "execution report is missing."}
        critic = evaluate_generated_run(
            run_dir=self.session.case_dir,
            task=self.session.config.task,
            execution_ok=bool(execution.get("ok")),
            require_render=self.session.config.render,
            use_codex_critic=True,
        )
        self.session.state["critic"] = critic
        self.session.state["control"]["needs_critic"] = False
        return {"ok": critic.get("verdict") == "pass", "status": "critic_evaluated", "critic": self.session.critic_excerpt(critic)}

    def _handle_request_repair(self, action: dict[str, Any]) -> dict[str, Any]:
        planner_output = self.session.current_planner_output()
        if planner_output is None:
            return {"ok": False, "status": "precondition_failed", "message": "planner_output is missing."}
        budgets = self.session.state["budgets"]
        if int(budgets["repair_attempts"]) >= int(budgets["max_repair_rounds"]):
            return {"ok": False, "status": "budget_exhausted", "message": "repair budget exhausted."}
        owner = str(action.get("owner") or self.session.recommended_owner())
        repair_brief = action.get("repair_brief")
        if not isinstance(repair_brief, str) or not repair_brief.strip():
            repair_brief = self.session.failure_context()
        repaired = repair_worker(
            case_dir=self.session.case_dir,
            task=self.session.config.task,
            owner=owner,
            failure_context=repair_brief,
        )
        budgets["repair_attempts"] = int(budgets["repair_attempts"]) + 1
        if repaired is None:
            return {"ok": False, "status": "invalid_owner", "message": f"Cannot repair owner {owner!r}."}
        self.session.record_worker_results([repaired])
        write_worker_dispatch_report(self.session.case_dir, [repaired])
        self.session.state["control"]["needs_integration"] = self.session.all_workers_ok()
        self.session.state["control"]["needs_execution"] = False
        self.session.state["control"]["needs_critic"] = False
        return {
            "ok": repaired.ok,
            "status": "repair_dispatched",
            "owner": owner,
            "all_workers_ok": self.session.all_workers_ok(),
        }

    def _handle_run_command(
        self,
        action: dict[str, Any],
        turn: int,
        *,
        label: str,
        executable: tuple[str, ...],
    ) -> dict[str, Any]:
        arg_key = "python_args" if label == "python" else "pytest_args"
        raw_args = action.get(arg_key)
        if not isinstance(raw_args, list) or not all(isinstance(item, str) for item in raw_args):
            return {"ok": False, "status": "invalid_action", "message": f"{label} requires {arg_key} string array."}
        cwd_choice = action.get("cwd") or "repo"
        cwd = Path.cwd() if cwd_choice == "repo" else self.session.case_dir
        timeout_sec = float(action.get("timeout_sec") or min(self.session.config.timeout_sec, 300.0))
        command = [*executable, *raw_args]
        stdout_path = self.session.command_dir / f"turn_{turn:03d}_{label}.stdout.txt"
        stderr_path = self.session.command_dir / f"turn_{turn:03d}_{label}.stderr.txt"
        started = time.time()
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_sec,
                check=False,
            )
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            stdout = _decode_timeout_stream(exc.stdout)
            stderr = _decode_timeout_stream(exc.stderr)
        duration = time.time() - started
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        command_report = {
            "label": label,
            "command": command,
            "cwd": str(cwd),
            "returncode": returncode,
            "duration_sec": duration,
            "timed_out": timed_out,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        self.session.state.setdefault("commands", []).append(command_report)
        return {"ok": returncode == 0, "status": "command_finished", "command": command_report}

    def _handle_finish(self, action: dict[str, Any]) -> dict[str, Any]:
        verdict = action.get("verdict") or "inconclusive"
        if verdict == "pass":
            critic = self.session.state.get("critic")
            if not isinstance(critic, dict) or critic.get("verdict") != "pass":
                return {"ok": False, "status": "precondition_failed", "message": "finish pass requires critic pass."}
        if verdict not in {"pass", "fail", "inconclusive"}:
            verdict = "inconclusive"
        self.session.state["status"] = verdict
        self.session.state["stop_reason"] = action.get("summary") or action.get("rationale")
        return {"ok": verdict == "pass", "status": "finished", "verdict": verdict}

    def _roles_from_action(self, action: dict[str, Any]) -> tuple[WorkerRole, ...]:
        raw_roles = action.get("roles")
        if not isinstance(raw_roles, list):
            return ()
        roles: list[WorkerRole] = []
        for role in raw_roles:
            if role in WORKER_ROLES and role not in roles:
                roles.append(role)
        return tuple(roles)


def _decode_timeout_stream(stream: bytes | str | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream
