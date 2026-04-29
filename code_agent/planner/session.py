from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - the uv environment normally has jsonschema.
    Draft202012Validator = None  # type: ignore[assignment]

from code_agent.io_utils import dump_json
from code_agent.utils.timing import TimingPlan, resolve_timing
from code_agent.planner.actions import EpisodeActionExecutor, WORKER_ROLES
from code_agent.planner.agent import EpisodePlanner
from code_agent.writer.common import WorkerDispatchResult


@dataclass(slots=True, frozen=True)
class PlannerSessionConfig:
    case_id: str
    task: str
    case_dir: Path
    backend: str
    timeout_sec: float
    render: bool
    repair_rounds: int
    steps: int | None = None
    duration_sec: float | None = None
    render_fps: int | None = None
    max_planner_turns: int | None = None


class PlannerSession:
    """Planner-led controller for one generated simulation case."""

    def __init__(self, config: PlannerSessionConfig):
        self.config = config
        self.case_dir = config.case_dir
        self.contracts_dir = self.case_dir / "contracts"
        self.reports_dir = self.case_dir / "reports"
        self.logs_dir = self.case_dir / "logs"
        self.command_dir = self.reports_dir / "planner_commands"
        self.action_history_path = self.reports_dir / "planner_actions.jsonl"
        self.dispatch_history_path = self.reports_dir / "dispatch_history.jsonl"
        self.state_path = self.reports_dir / "episode_state.json"
        self.summary_path = self.case_dir / "summary.json"
        self.timing: TimingPlan | None = None
        max_turns = config.max_planner_turns or max(12, 7 + config.repair_rounds * 5)
        self.state: dict[str, Any] = {
            "schema_version": 1,
            "case_id": config.case_id,
            "task": config.task,
            "status": "running",
            "turn_index": 0,
            "planner_output_path": None,
            "timing": None,
            "control": {
                "needs_integration": False,
                "needs_execution": False,
                "needs_critic": False,
            },
            "budgets": {
                "max_planner_turns": max_turns,
                "max_repair_rounds": max(0, config.repair_rounds),
                "repair_attempts": 0,
            },
            "workers": {
                role: {
                    "status": "pending",
                    "ok": False,
                    "target_file": f"src/{role}.py",
                    "latest_report": None,
                    "latest_error": None,
                }
                for role in WORKER_ROLES
            },
            "integration": None,
            "execution": None,
            "critic": None,
            "commands": [],
            "observations": [],
            "stop_reason": None,
        }
        self.planner = EpisodePlanner(self)
        self.actions = EpisodeActionExecutor(self)

    def run(self) -> dict[str, Any]:
        self._ensure_dirs()
        self.persist_state()
        max_turns = int(self.state["budgets"]["max_planner_turns"])
        for turn in range(max_turns):
            if self.state["status"] != "running":
                break
            self.state["turn_index"] = turn
            action = self.planner.request_action(turn)
            result = self.actions.execute(action, turn)
            self.record_action(turn=turn, action=action, result=result)
            self.observe(turn=turn, action=str(action.get("action")), result=result)
            self.persist_state()

        if self.state["status"] == "running":
            self.state["status"] = "fail"
            self.state["stop_reason"] = "max_planner_turns_exhausted"
            self.persist_state()

        summary = self.build_summary()
        dump_json(summary, self.summary_path)
        return summary

    def _ensure_dirs(self) -> None:
        for path in (self.contracts_dir, self.reports_dir, self.logs_dir, self.command_dir):
            path.mkdir(parents=True, exist_ok=True)
        for path in (self.action_history_path, self.dispatch_history_path):
            if path.exists():
                path.unlink()

    def record_worker_results(self, results: list[WorkerDispatchResult]) -> None:
        for item in results:
            self.state["workers"][item.role] = {
                "status": "ok" if item.ok else "failed",
                "ok": item.ok,
                "target_file": str(item.target_path),
                "latest_report": item.worker_report,
                "latest_error": item.error_message,
                "codex": {
                    "returncode": item.codex_result.exit_code,
                    "duration_sec": item.codex_result.duration_sec,
                    "final_message_path": item.codex_result.final_message_path,
                    "stderr_path": item.codex_result.stderr_path,
                },
            }
            self.append_jsonl(
                self.dispatch_history_path,
                {
                    "turn": self.state["turn_index"],
                    "role": item.role,
                    "ok": item.ok,
                    "target_path": str(item.target_path),
                    "worker_report": item.worker_report,
                    "error_message": item.error_message,
                },
            )

    def record_action(self, *, turn: int, action: dict[str, Any], result: dict[str, Any]) -> None:
        self.append_jsonl(
            self.action_history_path,
            {
                "turn": turn,
                "action": action,
                "result": self.json_safe(result),
                "created_at_unix": time.time(),
            },
        )

    def observe(self, *, turn: int, action: str, result: dict[str, Any]) -> None:
        self.state.setdefault("observations", []).append(
            {
                "turn": turn,
                "action": action,
                "ok": bool(result.get("ok")),
                "status": result.get("status"),
                "message": result.get("message") or self.short_result_message(result),
            }
        )

    def short_result_message(self, result: dict[str, Any]) -> str:
        if "roles" in result:
            return f"roles={result['roles']}, all_workers_ok={result.get('all_workers_ok')}"
        if "owner" in result:
            return f"owner={result['owner']}, all_workers_ok={result.get('all_workers_ok')}"
        if "verdict" in result:
            return f"verdict={result['verdict']}"
        return str(result.get("status"))

    def build_summary(self) -> dict[str, Any]:
        critic = self.state.get("critic") if isinstance(self.state.get("critic"), dict) else {}
        execution = self.state.get("execution") if isinstance(self.state.get("execution"), dict) else {}
        verdict = "pass" if self.state["status"] == "pass" else "fail"
        if critic and critic.get("verdict") != "pass":
            verdict = "fail"
        return {
            "case_id": self.config.case_id,
            "verdict": verdict,
            "execution_ok": bool(execution.get("ok")),
            "recommended_owner": critic.get("recommended_owner", "none") if isinstance(critic, dict) else "none",
            "repair_attempts": self.state["budgets"]["repair_attempts"],
            "case_dir": str(self.case_dir),
            "timing": self.state.get("timing"),
            "episode_state_path": str(self.state_path),
            "planner_actions_path": str(self.action_history_path),
            "dispatch_history_path": str(self.dispatch_history_path),
            "stop_reason": self.state.get("stop_reason"),
        }

    def current_planner_output(self) -> dict[str, Any] | None:
        path_text = self.state.get("planner_output_path")
        if not isinstance(path_text, str):
            return None
        return self.load_json(Path(path_text))

    def current_timing(self) -> TimingPlan:
        if self.timing is not None:
            return self.timing
        planner_output = self.current_planner_output()
        timing = resolve_timing(
            planner_output=planner_output,
            steps=self.config.steps,
            duration_sec=self.config.duration_sec,
            render_fps=self.config.render_fps,
        )
        self.timing = timing
        self.state["timing"] = timing.to_dict()
        return timing

    def all_workers_ok(self) -> bool:
        return all(bool(self.state["workers"][role].get("ok")) for role in WORKER_ROLES)

    def recommended_owner(self) -> str:
        critic = self.state.get("critic")
        if isinstance(critic, dict):
            return str(critic.get("recommended_owner", "none"))
        return "none"

    def failure_context(self) -> str:
        parts = [
            "Planner repair brief was empty; using current failure context.",
            "Critic report:",
            json.dumps(self.state.get("critic"), indent=2)[:8000],
            "Execution report:",
            json.dumps(self.state.get("execution"), indent=2)[:4000],
            "stderr:",
            self.read_text(self.reports_dir / "stderr.txt", limit=4000),
            "stdout:",
            self.read_text(self.reports_dir / "stdout.txt", limit=4000),
        ]
        return "\n\n".join(parts)

    def critic_excerpt(self, critic: dict[str, Any]) -> dict[str, Any]:
        return {
            "verdict": critic.get("verdict"),
            "recommended_owner": critic.get("recommended_owner"),
            "summary": critic.get("summary"),
            "missing_artifacts": critic.get("missing_artifacts"),
            "codex_critic_verdict": (
                critic.get("codex_critic_report", {}).get("verdict")
                if isinstance(critic.get("codex_critic_report"), dict)
                else None
            ),
        }

    def execution_excerpt(self, execution: dict[str, Any]) -> dict[str, Any]:
        artifacts = execution.get("artifacts")
        artifact_map = artifacts if isinstance(artifacts, dict) else {}
        frame_count = sum(1 for key in artifact_map if str(key).startswith("frame_"))
        important_artifacts = {
            key: artifact_map.get(key)
            for key in ("event_log", "metrics", "render", "render_stats", "run_result", "summary")
            if artifact_map.get(key)
        }
        return {
            "command": execution.get("command"),
            "returncode": execution.get("returncode"),
            "duration_sec": execution.get("duration_sec"),
            "stdout_path": execution.get("stdout_path"),
            "stderr_path": execution.get("stderr_path"),
            "ok": execution.get("ok"),
            "frame_count": frame_count,
            "artifacts": important_artifacts,
        }

    def persist_state(self) -> None:
        dump_json(self.json_safe(self.state), self.state_path)

    def load_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def validate_json_schema(self, payload: dict[str, Any], schema_path: Path) -> list[str]:
        if Draft202012Validator is None:
            return []
        schema = self.load_json(schema_path)
        if schema is None:
            return [f"schema missing or invalid: {schema_path}"]
        validator = Draft202012Validator(schema)
        return [error.message for error in sorted(validator.iter_errors(payload), key=lambda item: item.path)]

    def read_text(self, path: Path, *, limit: int) -> str:
        if not path.exists():
            return f"<missing {path}>"
        return path.read_text(encoding="utf-8", errors="replace")[:limit]

    def append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(self.json_safe(payload), ensure_ascii=False) + "\n")

    def json_safe(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self.json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [self.json_safe(item) for item in value]
        return value
