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

from code_agent.configs import deformable_config_dict
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
    deformable_enabled: bool = False
    ipc_enabled: bool = False
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
        self.deformable_config_path = self.contracts_dir / "deformable_config.json"
        self.action_history_path = self.reports_dir / "planner_actions.jsonl"
        self.dispatch_history_path = self.reports_dir / "dispatch_history.jsonl"
        self.state_path = self.reports_dir / "episode_state.json"
        self.summary_path = self.case_dir / "summary.json"
        self.timing: TimingPlan | None = None
        self.deformable_config = deformable_config_dict(
            deformable_enabled=config.deformable_enabled,
            ipc_enabled=config.ipc_enabled,
        )
        max_turns = config.max_planner_turns or max(12, 7 + config.repair_rounds * 5)
        self.state: dict[str, Any] = {
            "schema_version": 1,
            "case_id": config.case_id,
            "task": config.task,
            "status": "running",
            "turn_index": 0,
            "planner_output_path": None,
            "timing": None,
            "capabilities": {
                "deformable_enabled": config.deformable_enabled,
                "ipc_enabled": config.ipc_enabled,
                "deformable_scope": "FEM only when enabled; MPM/PBD/SPH remain out of scope.",
                "ipc_scope": "IPC may be enabled for rigid/articulated contact; deformable forces IPC on.",
                "deformable_config_path": str(self.deformable_config_path),
            },
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
            "assets": {
                "status": "not_requested",
                "ok": False,
                "asset_manifest_path": None,
                "asset_generation_report_path": None,
                "selected_asset_names": [],
                "skipped_asset_names": [],
                "num_assets": 0,
                "schema_errors": [],
                "jobs": {},
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
        self.write_deformable_config_contract()
        self.actions.assets.adopt_layout_asset_manifest()
        for path in (self.action_history_path, self.dispatch_history_path):
            if path.exists():
                path.unlink()

    def write_deformable_config_contract(self) -> None:
        dump_json(self.deformable_config, self.deformable_config_path)

    def accept_planner_output(self, planner_output: dict[str, Any], *, rationale: str | None = None) -> dict[str, Any]:
        errors = self.validate_json_schema(planner_output, Path("code_agent/specs/planner_output.schema.json"))
        if errors:
            return {
                "ok": False,
                "status": "invalid_planner_output",
                "message": "planner_output failed schema validation.",
                "errors": errors,
            }

        planner_output_path = self.contracts_dir / "planner_output.json"
        dump_json(planner_output, planner_output_path)
        self.write_deformable_config_contract()

        timing = resolve_timing(
            planner_output=planner_output,
            steps=self.config.steps,
            duration_sec=self.config.duration_sec,
            render_fps=self.config.render_fps,
        )
        self.timing = timing
        dump_json(timing.to_dict(), self.contracts_dir / "timing.json")

        self.state["planner_output_path"] = str(planner_output_path)
        self.state["timing"] = timing.to_dict()

        episode_plan_path = self.contracts_dir / "episode_plan.json"
        dump_json(
            {
                "planner_output_path": str(planner_output_path),
                "planner_output": planner_output,
                "timing": timing.to_dict(),
                "rationale": rationale,
                "created_at_unix": time.time(),
            },
            episode_plan_path,
        )
        self.state["episode_plan_path"] = str(episode_plan_path)
        return {
            "ok": True,
            "status": "planner_output_accepted",
            "planner_output_path": str(planner_output_path),
            "episode_plan_path": str(episode_plan_path),
            "timing": timing.to_dict(),
        }

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
                    "error_type": item.codex_result.error_type,
                    "error_message": item.codex_result.error_message,
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
        if "asset_manifest_path" in result:
            return f"assets={result.get('num_assets')}, manifest={result.get('asset_manifest_path')}"
        if "asset_inspection_report_path" in result:
            return (
                f"asset_errors={result.get('asset_error_count')}, "
                f"asset_warnings={result.get('asset_warning_count')}, "
                f"report={result.get('asset_inspection_report_path')}"
            )
        if "verdict" in result:
            return f"verdict={result['verdict']}"
        return str(result.get("status"))

    def build_summary(self) -> dict[str, Any]:
        critic = self.state.get("critic") if isinstance(self.state.get("critic"), dict) else {}
        execution = self.state.get("execution") if isinstance(self.state.get("execution"), dict) else {}
        status = str(self.state.get("status") or "fail")
        verdict = status if status in {"pass", "fail", "inconclusive"} else "fail"
        if verdict == "pass" and critic and critic.get("verdict") != "pass":
            verdict = "fail"
        return {
            "case_id": self.config.case_id,
            "verdict": verdict,
            "status": status,
            "execution_ok": bool(execution.get("ok")),
            "recommended_owner": critic.get("recommended_owner", "none") if isinstance(critic, dict) else "none",
            "repair_attempts": self.state["budgets"]["repair_attempts"],
            "case_dir": str(self.case_dir),
            "timing": self.state.get("timing"),
            "asset_manifest_path": self.state.get("assets", {}).get("asset_manifest_path"),
            "episode_state_path": str(self.state_path),
            "planner_actions_path": str(self.action_history_path),
            "dispatch_history_path": str(self.dispatch_history_path),
            "stop_reason": self.state.get("stop_reason"),
            "blocked_reason": self.state.get("blocked_reason"),
            "deformable_enabled": self.config.deformable_enabled,
            "ipc_enabled": self.config.ipc_enabled,
            "deformable_config_path": str(self.deformable_config_path),
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

    def asset_manifest_ready(self) -> bool:
        assets = self.state.get("assets")
        if not isinstance(assets, dict) or not assets.get("ok"):
            return False
        manifest_path = assets.get("asset_manifest_path")
        return isinstance(manifest_path, str) and Path(manifest_path).exists()

    def recommended_owner(self) -> str:
        critic = self.state.get("critic")
        if isinstance(critic, dict):
            return str(critic.get("recommended_owner", "none"))
        return "none"

    def failure_context(self) -> str:
        parts = [
            "Planner repair brief was empty; using current failure context.",
            "Critic report:",
            json.dumps(self.state.get("critic"), indent=2),
            "Execution report:",
            json.dumps(self.state.get("execution"), indent=2),
            "stderr:",
            self.read_text(self.reports_dir / "stderr.txt"),
            "stdout:",
            self.read_text(self.reports_dir / "stdout.txt"),
        ]
        return "\n\n".join(parts)

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

    def read_text(self, path: Path) -> str:
        if not path.exists():
            return f"<missing {path}>"
        return path.read_text(encoding="utf-8", errors="replace")

    def genesis_context_prompt(self) -> str:
        context_md = self.contracts_dir / "genesis_context.md"
        context_json = self.contracts_dir / "genesis_context.json"
        context = self.load_json(context_json) or {}
        docs_dir = context.get("docs_dir")
        catalog_path = context.get("catalog_path")
        return "\n".join(
            [
                "Genesis official-doc and local-source context is available on disk for on-demand reading.",
                "Do not assume the context pack is preloaded; inspect only the pieces relevant to this turn.",
                f"- Context index: {context_md}",
                f"- Machine-readable context JSON: {context_json}",
                f"- Cached official docs directory: {docs_dir or '<see context JSON>'}",
                f"- Selected official-doc catalog: {catalog_path or '<see context JSON>'}",
                f"- FEM deformable generation enabled: {self.config.deformable_enabled}.",
                f"- IPC contact/coupling enabled: {self.config.ipc_enabled}.",
                f"- Effective FEM/IPC config: {self.deformable_config_path}",
                "- Active non-rigid scope when FEM is enabled: FEM+IPC only.",
                "- If FEM is disabled but IPC is enabled, rigid bodies and articulated MJCF/URDF assets may still use "
                "IPC for contact/coupling.",
                "- Rigid bodies, articulated MJCF/URDF robots, generated meshes, textures, and rendering are in scope "
                "as explicitly requested rigid/mesh scenes or as FEM+IPC support.",
                "- Prefer local Genesis source and examples over online docs if they disagree.",
            ]
        )

    def layout_context_prompt(self) -> str:
        layout_context = self.case_dir / "inputs" / "layout_context.md"
        if not layout_context.exists():
            return ""
        return layout_context.read_text(encoding="utf-8", errors="replace")

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
