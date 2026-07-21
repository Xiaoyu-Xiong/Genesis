from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - the uv environment normally has jsonschema.
    Draft202012Validator = None  # type: ignore[assignment]

from code_agent.assets.builtin_guard import builtin_asset_violations
from code_agent.configs import CONFIGS, deformable_config_dict, rigid_config_dict
from code_agent.io_utils import dump_json
from code_agent.utils.codex import CODEX_INFRA_ERROR_TYPES, DEFAULT_REPO_ROOT
from code_agent.utils.timing import TimingPlan, resolve_timing
from code_agent.planner.actions import EpisodeActionExecutor
from code_agent.planner.action_handlers.worker_actions import WORKER_ROLES
from code_agent.planner.agent import EpisodePlanner
from code_agent.writer.common import WorkerDispatchResult

OPT_RESULT_STATUSES = {"success", "partial_success", "needs_more_optimization", "needs_rewrite", "failed"}
OPT_RERUN_STATUSES = {"success", "partial_success", "needs_more_optimization"}


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
    opt_enabled: bool = CONFIGS.opt.enabled
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
        self.rigid_config_path = self.contracts_dir / "rigid_config.json"
        self.action_history_path = self.reports_dir / "planner_actions.jsonl"
        self.dispatch_history_path = self.reports_dir / "dispatch_history.jsonl"
        self.state_path = self.reports_dir / "episode_state.json"
        self.summary_path = self.case_dir / "summary.json"
        self.timing: TimingPlan | None = None
        self.deformable_config = deformable_config_dict()
        self.rigid_config = rigid_config_dict()
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
                "physics_selection": "planner_decides",
                "deformable_enabled": bool(self.deformable_config["enabled"]),
                "ipc_enabled": bool(self.deformable_config["ipc_enabled"]),
                "deformable_scope": "FEM volumetric and FEM.Cloth thin-shell when enabled; MPM/PBD/SPH remain out of scope.",
                "ipc_scope": "IPC may be enabled for rigid/articulated contact; deformable forces IPC on.",
                "deformable_config_path": str(self.deformable_config_path),
                "rigid_config_path": str(self.rigid_config_path),
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
            "opt": {
                "enabled": config.opt_enabled,
                "status": "not_requested" if config.opt_enabled else "disabled",
                "attempts": 0,
                "latest_result": None,
                "latest_request": None,
                "history": [],
            },
            "physics_validation": {
                "status": "pending",
                "accepted_state_cache_manifest": None,
                "accepted_at_unix": None,
            },
            "final_render": {
                "required": bool(config.render),
                "status": "pending" if config.render else "not_required",
                "attempts": 0,
                "latest_render_profile": None,
                "latest_render_stats_path": None,
                "latest_issue": None,
                "passed_at_unix": None,
            },
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
            self.refresh_opt_state_from_report()
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
        self.write_rigid_config_contract()
        self.actions.assets.adopt_layout_asset_manifest()
        for path in (self.action_history_path, self.dispatch_history_path):
            if path.exists():
                path.unlink()

    def write_deformable_config_contract(self) -> None:
        dump_json(self.deformable_config, self.deformable_config_path)

    def write_rigid_config_contract(self) -> None:
        dump_json(self.rigid_config, self.rigid_config_path)

    def resolve_planner_physics_plan(self, planner_output: dict[str, Any]) -> dict[str, Any]:
        raw_plan = planner_output.get("physics_plan")
        if not isinstance(raw_plan, dict):
            return {"ok": False, "errors": ["physics_plan must be an object."]}

        mode = str(raw_plan.get("mode") or "")
        if mode == "fem_ipc":
            deformable_enabled = True
            ipc_enabled = True
            default_kind = "soft_body"
        elif mode == "rigid_ipc":
            deformable_enabled = False
            ipc_enabled = True
            default_kind = "none"
        elif mode == "rigid":
            deformable_enabled = False
            ipc_enabled = False
            default_kind = "none"
        else:
            return {"ok": False, "errors": [f"Unsupported physics_plan.mode: {mode!r}."]}

        errors: list[str] = []
        if bool(raw_plan.get("deformable_enabled")) != deformable_enabled:
            errors.append(f"physics_plan.deformable_enabled must match mode {mode!r}: expected {deformable_enabled}.")
        if not deformable_enabled and bool(raw_plan.get("ipc_enabled")) != ipc_enabled:
            errors.append(f"physics_plan.ipc_enabled must match mode {mode!r}: expected {ipc_enabled}.")

        if deformable_enabled:
            ipc_enabled = True

        deformable_kind = str(raw_plan.get("deformable_kind") or default_kind)
        if not deformable_enabled:
            deformable_kind = "none"
        elif deformable_kind == "none":
            errors.append("physics_plan.deformable_kind must not be none when deformable_enabled is true.")

        if errors:
            return {"ok": False, "errors": errors}

        physics_plan = {
            "mode": mode,
            "deformable_enabled": deformable_enabled,
            "deformable_kind": deformable_kind,
            "ipc_enabled": ipc_enabled,
            "rationale": str(raw_plan.get("rationale") or "Planner selected physics mode for this case."),
        }
        return {"ok": True, "physics_plan": physics_plan}

    def _sync_capability_state(self, physics_plan: dict[str, Any]) -> None:
        capabilities = self.state.setdefault("capabilities", {})
        if not isinstance(capabilities, dict):
            capabilities = {}
            self.state["capabilities"] = capabilities
        capabilities.update(
            {
                "physics_selection": "planner_selected",
                "physics_mode": physics_plan.get("mode"),
                "deformable_kind": physics_plan.get("deformable_kind"),
                "deformable_enabled": bool(physics_plan.get("deformable_enabled")),
                "ipc_enabled": bool(physics_plan.get("ipc_enabled")),
                "deformable_config_path": str(self.deformable_config_path),
                "rigid_config_path": str(self.rigid_config_path),
            }
        )

    def accept_planner_output(self, planner_output: dict[str, Any], *, rationale: str | None = None) -> dict[str, Any]:
        errors = self.validate_json_schema(planner_output, Path("code_agent/specs/planner_output.schema.json"))
        if errors:
            return {
                "ok": False,
                "status": "invalid_planner_output",
                "message": "planner_output failed schema validation.",
                "errors": errors,
            }
        asset_violations = builtin_asset_violations(planner_output, label="planner_output")
        if asset_violations:
            return {
                "ok": False,
                "status": "forbidden_builtin_asset_reference",
                "message": "planner_output references forbidden Genesis built-in assets.",
                "errors": asset_violations,
            }
        resolved_physics = self.resolve_planner_physics_plan(planner_output)
        if not resolved_physics.get("ok"):
            return {
                "ok": False,
                "status": "invalid_physics_plan",
                "message": "planner_output physics_plan is inconsistent with physics mode rules.",
                "errors": resolved_physics.get("errors", []),
            }
        planner_output = dict(planner_output)
        planner_output["physics_plan"] = resolved_physics["physics_plan"]
        self.deformable_config = deformable_config_dict(
            physics_mode=str(resolved_physics["physics_plan"]["mode"]),
        )
        self._sync_capability_state(resolved_physics["physics_plan"])

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
        self.refresh_opt_state_from_report()
        critic = self.state.get("critic") if isinstance(self.state.get("critic"), dict) else {}
        execution = self.state.get("execution") if isinstance(self.state.get("execution"), dict) else {}
        status = str(self.state.get("status") or "fail")
        verdict = status if status in {"pass", "fail", "inconclusive"} else "fail"
        if verdict == "pass" and critic and critic.get("verdict") != "pass":
            verdict = "fail"
        critic_infra_status = self._critic_infra_status(critic)
        infra_blocked_reason = self._infra_blocked_reason(critic_infra_status=critic_infra_status)
        outcome_class = self._outcome_class(verdict=verdict, infra_blocked_reason=infra_blocked_reason)
        return {
            "case_id": self.config.case_id,
            "verdict": verdict,
            "status": status,
            "outcome_class": outcome_class,
            "execution_ok": bool(execution.get("ok")),
            "critic_infra_status": critic_infra_status,
            "infra_blocked_reason": infra_blocked_reason,
            "retry_recommended": outcome_class == "infra_blocked",
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
            "physics_mode": self.state.get("capabilities", {}).get("physics_mode"),
            "deformable_enabled": bool(self.deformable_config.get("enabled")),
            "ipc_enabled": bool(self.deformable_config.get("ipc_enabled")),
            "opt_enabled": self.config.opt_enabled,
            "deformable_config_path": str(self.deformable_config_path),
            "opt": self.state.get("opt"),
            "physics_validation": self.state.get("physics_validation"),
            "final_render": self.state.get("final_render"),
        }

    def _critic_infra_status(self, critic: dict[str, Any]) -> str:
        status = critic.get("critic_infra_status") if isinstance(critic, dict) else None
        return status if isinstance(status, str) and status else "ok"

    def _infra_blocked_reason(self, *, critic_infra_status: str) -> str | None:
        if critic_infra_status not in {"ok", "not_used"}:
            return critic_infra_status
        worker_reason = self._worker_infra_blocked_reason()
        if worker_reason is not None:
            return worker_reason
        blocked_reason = self.state.get("blocked_reason")
        if isinstance(blocked_reason, dict):
            blocked_type = str(blocked_reason.get("type") or "")
            if blocked_type == "codex_usage_limit":
                return "quota_blocked"
            if blocked_type == "codex_auth_failed":
                return "auth_failed"
            if blocked_type == "codex_input_too_large":
                return "planner_prompt_too_large"
            if blocked_type == "timeout":
                return "planner_timeout"
            if blocked_type in CODEX_INFRA_ERROR_TYPES:
                return blocked_type
        stop_reason = str(self.state.get("stop_reason") or "").lower()
        if "usage limit" in stop_reason or "purchase more credits" in stop_reason:
            return "quota_blocked"
        if "401 unauthorized" in stop_reason:
            return "auth_failed"
        if "input exceeds the maximum length" in stop_reason:
            return "critic_prompt_too_large"
        return None

    def _worker_infra_blocked_reason(self) -> str | None:
        workers = self.state.get("workers")
        if not isinstance(workers, dict):
            return None
        for role, data in workers.items():
            if not isinstance(data, dict) or data.get("ok"):
                continue
            codex = data.get("codex")
            if not isinstance(codex, dict):
                continue
            error_type = codex.get("error_type")
            if isinstance(error_type, str) and error_type in CODEX_INFRA_ERROR_TYPES:
                return f"{role}_worker:{error_type}"
        return None

    def _outcome_class(self, *, verdict: str, infra_blocked_reason: str | None) -> str:
        if verdict == "pass":
            return "pass"
        if infra_blocked_reason is not None:
            return "infra_blocked"
        if verdict == "fail":
            return "fail"
        return "semantic_inconclusive"

    def refresh_opt_state_from_report(self) -> None:
        """Reconcile Opt state with a structured report that landed after timeout.

        Codex can write its final JSON just as the wrapper is timing out. The
        planner state should reflect the parseable Opt payload on disk rather
        than a stale wrapper-level timeout status.
        """

        opt = self.state.get("opt")
        if not self.config.opt_enabled or not isinstance(opt, dict):
            return
        payload = self._load_latest_opt_payload()
        if payload is None:
            return
        status = payload.get("status")
        if status not in OPT_RESULT_STATUSES:
            return

        result_payload = self.json_safe(payload)
        latest_result = opt.get("latest_result")
        if isinstance(latest_result, dict) and latest_result == result_payload and opt.get("status") == status:
            return

        opt["status"] = status
        opt["latest_result"] = result_payload
        opt["updated_at_unix"] = time.time()
        report = self.load_json(self.reports_dir / "opt_subagent_report.json")
        request = report.get("request") if isinstance(report, dict) else None
        if isinstance(request, dict):
            opt["latest_request"] = self.json_safe(request)
        history = opt.setdefault("history", [])
        if isinstance(history, list):
            attempt = int(opt.get("attempts") or len(history) or 1)
            if not history or history[-1].get("result") != result_payload:
                history.append({"attempt": attempt, "result": result_payload, "source": "disk_reconcile"})

        if self.state.get("status") == "running" and status in OPT_RERUN_STATUSES:
            self.sync_best_opt_params_to_current(selected_by="planner.opt_disk_reconcile")
            critic = self.state.get("critic")
            if not isinstance(critic, dict) or critic.get("verdict") != "pass":
                self.state["control"]["needs_execution"] = True
                self.state["control"]["needs_critic"] = False

    def _load_latest_opt_payload(self) -> dict[str, Any] | None:
        final_payload = self.load_json(self.logs_dir / "codex_opt_subagent.final.json")
        if isinstance(final_payload, dict) and final_payload.get("status") in OPT_RESULT_STATUSES:
            return final_payload
        report = self.load_json(self.reports_dir / "opt_subagent_report.json")
        result = report.get("result") if isinstance(report, dict) else None
        if isinstance(result, dict) and result.get("status") in OPT_RESULT_STATUSES:
            return result
        return None

    def sync_best_opt_params_to_current(self, *, selected_by: str) -> str | None:
        best_path = self.contracts_dir / "best_opt_params.json"
        current_path = self.contracts_dir / "current_opt_params.json"
        payload = self.load_json(best_path)
        if payload is None:
            return None
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = dict(metadata)
        metadata["selected_by"] = selected_by
        metadata["selected_at_unix"] = time.time()
        payload = dict(payload)
        payload["metadata"] = metadata
        dump_json(payload, current_path)
        return str(current_path)

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
        return isinstance(manifest_path, str) and self._stable_path(Path(manifest_path)).exists()

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

    def simdebug_card_context_for_role(
        self,
        target_role: str,
        *,
        turn: int | None = None,
        dispatch_reason: str = "",
        requested_card_ids: tuple[str, ...] | list[str] | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> str:
        if not self.simdebug_cards_enabled():
            return ""
        from code_agent.context.simdebug import format_simdebug_cards_for_prompt, select_simdebug_cards

        role = str(target_role or "planner")
        case_state: dict[str, Any] = {
            "task": self.config.task,
            "case_id": self.config.case_id,
            "turn": self.state.get("turn_index") if turn is None else turn,
            "target_role": role,
            "dispatch_reason": dispatch_reason,
            "physics_modes": list(self.simdebug_physics_modes()),
            "deformable_enabled": bool(self.deformable_config.get("enabled")),
            "ipc_enabled": bool(self.deformable_config.get("ipc_enabled")),
            "deformable_config": self.deformable_config,
            "planner_state": self.state,
        }
        if extra_state:
            case_state["dispatch_state"] = extra_state
        selection = select_simdebug_cards(case_state, target_role=role, requested_card_ids=requested_card_ids)
        dispatch = {
            "schema_version": 1,
            "turn": case_state["turn"],
            "case_id": self.config.case_id,
            "target_role": role,
            "dispatch_reason": dispatch_reason,
            "requested_card_ids": list(requested_card_ids or ()),
            "selection": selection,
        }
        simdebug_state = self.state.setdefault("simdebug", {})
        if not isinstance(simdebug_state, dict):
            simdebug_state = {}
            self.state["simdebug"] = simdebug_state
        selected_ids = [
            item.get("id") for item in selection.get("selected_cards", []) if isinstance(item, dict) and item.get("id")
        ]
        latest_by_role = simdebug_state.setdefault("latest_by_role", {})
        if isinstance(latest_by_role, dict):
            latest_by_role[role] = {
                "turn": case_state["turn"],
                "dispatch_reason": dispatch_reason,
                "selected_card_ids": selected_ids,
                "dispatch_path": str(self.reports_dir / f"simdebug_card_dispatch_{role}.json"),
            }
        simdebug_state["latest_turn"] = case_state["turn"]
        simdebug_state["latest_target_role"] = role
        simdebug_state["latest_selected_card_ids"] = selected_ids
        simdebug_state["latest_dispatch_path"] = str(self.reports_dir / f"simdebug_card_dispatch_{role}.json")
        dump_json(self.json_safe(dispatch), self.reports_dir / f"simdebug_card_dispatch_{role}.json")
        if role == "planner":
            dump_json(self.json_safe(dispatch), self.reports_dir / "simdebug_card_dispatch.json")
        self.append_jsonl(self.reports_dir / "simdebug_card_dispatch.jsonl", dispatch)
        return format_simdebug_cards_for_prompt(selection)

    def simdebug_physics_modes(self) -> tuple[str, ...]:
        capabilities = self.state.get("capabilities")
        if isinstance(capabilities, dict) and capabilities.get("physics_selection") == "planner_selected":
            if bool(self.deformable_config.get("enabled")):
                return ("fem_ipc",)
            if bool(self.deformable_config.get("ipc_enabled")):
                return ("rigid_ipc",)
            return ("rigid",)

        return ("rigid", "rigid_ipc", "fem_ipc")

    def simdebug_card_ids_from_action(self, action: dict[str, Any], target_role: str) -> tuple[str, ...] | None:
        if not self.simdebug_cards_enabled():
            return None
        raw = action.get("simdebug_cards")
        if not isinstance(raw, dict):
            return None

        role = str(target_role or "").strip().lower().replace("-", "_").replace(" ", "_")
        worker_roles = {"scene", "body", "action", "rendering"}
        ordered_ids: list[str] = []
        saw_explicit_bucket = False

        def add_bucket(key: str) -> None:
            nonlocal saw_explicit_bucket
            if key not in raw:
                return
            saw_explicit_bucket = True
            value = raw.get(key)
            if value is None:
                return
            if isinstance(value, str):
                values = [value]
            elif isinstance(value, list):
                values = value
            else:
                return
            for item in values:
                if not isinstance(item, str):
                    continue
                card_id = item.strip()
                if card_id and card_id not in ordered_ids:
                    ordered_ids.append(card_id)

        add_bucket("all")
        if role in worker_roles:
            add_bucket("workers")
            add_bucket("all_workers")
        add_bucket(role)
        return tuple(ordered_ids) if saw_explicit_bucket else None

    def simdebug_cards_enabled(self) -> bool:
        from code_agent.prompts import prompt_mode

        if prompt_mode() == "legacy":
            return False
        flag = os.environ.get("CODE_AGENT_SIMDEBUG_CARDS")
        if flag is None:
            return True
        return flag.strip().lower() not in {"0", "false", "off", "no"}

    def persist_state(self) -> None:
        dump_json(self.json_safe(self.state), self.state_path)

    def load_json(self, path: Path) -> dict[str, Any] | None:
        path = self._stable_path(path)
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
        schema = self.load_json(self._stable_path(schema_path))
        if schema is None:
            return [f"schema missing or invalid: {schema_path}"]
        validator = Draft202012Validator(schema)
        return [error.message for error in sorted(validator.iter_errors(payload), key=lambda item: item.path)]

    def read_text(self, path: Path) -> str:
        path = self._stable_path(path)
        if not path.exists():
            return f"<missing {path}>"
        return path.read_text(encoding="utf-8", errors="replace")

    def genesis_context_prompt(self) -> str:
        context_md = self.contracts_dir / "genesis_context.md"
        context_json = self.contracts_dir / "genesis_context.json"
        context = self.load_json(context_json) or {}
        docs_dir = context.get("docs_dir")
        catalog_path = context.get("catalog_path")
        capabilities = self.state.get("capabilities") if isinstance(self.state.get("capabilities"), dict) else {}
        physics_selection = str(capabilities.get("physics_selection") or "planner_decides")
        return "\n".join(
            [
                "Genesis official-doc and local-source context is available on disk for on-demand reading.",
                "Do not assume the context pack is preloaded; inspect only the pieces relevant to this turn.",
                f"- Context index: {context_md}",
                f"- Machine-readable context JSON: {context_json}",
                f"- Cached official docs directory: {docs_dir or '<see context JSON>'}",
                f"- Selected official-doc catalog: {catalog_path or '<see context JSON>'}",
                f"- Physics selection status: {physics_selection}.",
                f"- Current FEM deformable generation enabled: {bool(self.deformable_config.get('enabled'))}.",
                f"- Current IPC contact/coupling enabled: {bool(self.deformable_config.get('ipc_enabled'))}.",
                f"- Allowed SimDebug physics modes for this turn: {', '.join(self.simdebug_physics_modes())}.",
                f"- Effective FEM/IPC config: {self.deformable_config_path}",
                "- Active non-rigid scope when FEM is enabled: FEM+IPC only, including FEM.Cloth thin-shell cloth.",
                "- PBD cloth remains out of scope; use ready cloth_mesh assets plus gs.materials.FEM.Cloth for cloth.",
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
        path = self._stable_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(self.json_safe(payload), ensure_ascii=False) + "\n")

    def _stable_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return DEFAULT_REPO_ROOT / path

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
