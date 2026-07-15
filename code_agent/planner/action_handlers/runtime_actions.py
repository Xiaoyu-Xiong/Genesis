from __future__ import annotations

import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from code_agent.assets.builtin_guard import builtin_asset_violations, case_source_builtin_asset_violations
from code_agent.configs import CONFIGS
from code_agent.evaluation.runner import evaluate_generated_run
from code_agent.io_utils import decode_process_stream
from code_agent.opt.agent import run_opt_agent
from code_agent.opt.types import OptAgentRequest
from code_agent.utils.codex import DEFAULT_REPO_ROOT
from code_agent.utils.execution import run_generated_simulation
from code_agent.utils.local_execution import build_local_execution_env
from code_agent.utils.integrator import write_main


class RuntimeActionHandler:
    """Planner action handlers for planning artifacts, execution, critic, and controlled commands."""

    def __init__(self, session: Any):
        self.session = session

    def write_plan(self, action: dict[str, Any]) -> dict[str, Any]:
        planner_output = action.get("planner_output")
        if not isinstance(planner_output, dict):
            return {"ok": False, "status": "invalid_action", "message": "write_plan requires planner_output object."}
        accepted = self.session.accept_planner_output(
            planner_output,
            rationale=str(action.get("rationale") or ""),
        )
        if not accepted.get("ok"):
            return accepted
        return {
            "ok": True,
            "status": "planned",
            "message": "Planner output and timing were accepted.",
            "planner_output_path": accepted.get("planner_output_path"),
            "episode_plan_path": accepted.get("episode_plan_path"),
            "timing": accepted.get("timing"),
        }

    def run_integrator(self) -> dict[str, Any]:
        planner_output = self.session.current_planner_output()
        if planner_requires_asset_manifest(planner_output) and not self.session.asset_manifest_ready():
            assets = self.session.state.get("assets", {})
            status = assets.get("status") if isinstance(assets, dict) else None
            return {
                "ok": False,
                "status": "precondition_failed",
                "message": f"Asset manifest is required before integration. Current asset status: {status}.",
            }
        if not self.session.all_workers_ok():
            return {
                "ok": False,
                "status": "precondition_failed",
                "message": "All workers must be ok before integration.",
            }
        asset_violations = case_source_builtin_asset_violations(self.session.case_dir)
        if asset_violations:
            return {
                "ok": False,
                "status": "forbidden_builtin_asset_reference",
                "message": "Generated source references forbidden Genesis built-in assets.",
                "errors": asset_violations,
            }
        timing = self.session.current_timing()
        main_py = write_main(
            run_dir=self.session.case_dir,
            task=self.session.config.task,
            default_steps=timing.steps,
            default_render_fps=timing.render_fps,
            default_sim_dt=timing.sim_dt,
            default_sim_substeps=timing.sim_substeps,
            default_render_every_n_steps=timing.render_every_n_steps,
            default_render_res=timing.render_res,
            default_duration_sec=timing.duration_sec,
            default_target_video_frames=timing.target_video_frames,
            deformable_cfg=self.session.deformable_config,
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

    def run_execution(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.session.state.get("integration") is None:
            return {"ok": False, "status": "precondition_failed", "message": "integration is missing."}
        asset_violations = case_source_builtin_asset_violations(self.session.case_dir)
        if asset_violations:
            return {
                "ok": False,
                "status": "forbidden_builtin_asset_reference",
                "message": "Generated source references forbidden Genesis built-in assets.",
                "errors": asset_violations,
            }
        timing = self.session.current_timing()
        main_py = Path(str(self.session.state["integration"]["main_py"]))
        backend = str(action.get("backend") or self.session.config.backend)
        render = self.session.config.render if action.get("render") is None else bool(action.get("render"))
        timeout_sec = float(self.session.config.timeout_sec)
        render_profile = self._render_profile_from_action(action)
        replay_cache = self._replay_cache_from_action(action, render_profile=render_profile)
        render_only = self._bool_from_action(
            action.get("render_only"),
            default=render_profile == "final_path_traced" and replay_cache is not None,
        )
        # Physics executions must always leave a replayable cache. Do not let a
        # planner action disable this; only render-only replay skips writing a
        # new physics cache because it does not step physics.
        save_state_cache = not render_only
        require_state_cache = not render_only
        execution = run_generated_simulation(
            main_py=main_py,
            run_dir=self.session.case_dir,
            backend=backend,
            timeout_sec=timeout_sec,
            render=render,
            steps=timing.steps,
            render_fps=timing.render_fps,
            sim_dt=timing.sim_dt,
            sim_substeps=timing.sim_substeps,
            render_every_n_steps=timing.render_every_n_steps,
            render_res=timing.render_res,
            duration_sec=timing.duration_sec,
            target_video_frames=timing.target_video_frames,
            render_profile=render_profile,
            save_state_cache=save_state_cache,
            require_state_cache=require_state_cache,
            replay_cache=replay_cache,
            render_only=render_only,
        )
        execution_report = execution.to_dict()
        execution_report["render_profile"] = render_profile
        execution_report["save_state_cache"] = save_state_cache
        execution_report["require_state_cache"] = require_state_cache
        execution_report["replay_cache"] = None if replay_cache is None else str(replay_cache)
        execution_report["render_only"] = render_only
        self.session.state["execution"] = execution_report
        self._record_render_execution(
            render_profile=render_profile,
            replay_cache=replay_cache,
            render_only=render_only,
        )
        self.session.state["control"]["needs_execution"] = False
        self.session.state["control"]["needs_critic"] = True
        return {"ok": execution.ok, "status": "executed", "execution": self.session.state["execution"]}

    def run_critic(self, action: dict[str, Any]) -> dict[str, Any]:
        execution = self.session.state.get("execution")
        if not isinstance(execution, dict):
            return {"ok": False, "status": "precondition_failed", "message": "execution report is missing."}
        critic = evaluate_generated_run(
            run_dir=self.session.case_dir,
            task=self.session.config.task,
            execution_ok=bool(execution.get("ok")),
            require_render=self.session.config.render,
            use_codex_critic=True,
            simdebug_card_context=self.session.simdebug_card_context_for_role(
                "critic",
                turn=self.session.state.get("turn_index"),
                dispatch_reason="run_critic",
                requested_card_ids=self.session.simdebug_card_ids_from_action(action, "critic"),
                extra_state={"execution": execution},
            ),
        )
        self.session.state["critic"] = critic
        self._record_render_critic_result(critic=critic, execution=execution)
        self.session.state["control"]["needs_critic"] = False
        return {"ok": critic.get("verdict") == "pass", "status": "critic_evaluated", "critic": critic}

    def run_opt(self, action: dict[str, Any]) -> dict[str, Any]:
        opt_state = self.session.state.setdefault(
            "opt",
            {
                "enabled": self.session.config.opt_enabled,
                "status": "not_requested" if self.session.config.opt_enabled else "disabled",
                "attempts": 0,
                "latest_result": None,
                "latest_request": None,
                "history": [],
            },
        )
        opt_state["enabled"] = self.session.config.opt_enabled
        if not self.session.config.opt_enabled:
            opt_state["status"] = "disabled"
            return {
                "ok": False,
                "status": "opt_disabled",
                "message": "Opt is disabled for this suite run; Planner must use the normal repair/critic path.",
            }
        if self.session.state.get("integration") is None:
            return {"ok": False, "status": "precondition_failed", "message": "integration is missing."}

        timing = self.session.current_timing()
        render_flag = action.get("render")
        simdebug_card_context = self.session.simdebug_card_context_for_role(
            "opt",
            turn=self.session.state.get("turn_index"),
            dispatch_reason="run_opt",
            requested_card_ids=self.session.simdebug_card_ids_from_action(action, "opt"),
            extra_state={"planner_action": action, "critic": self.session.state.get("critic")},
        )
        request = OptAgentRequest(
            case_dir=self.session.case_dir,
            original_prompt=self.session.config.task,
            planner_intent=self._opt_planner_intent(action),
            allowed_edits=(
                "src/action.py for control schedules, target poses, controller gains, force limits, and action hooks",
                (
                    "src/body.py for material, contact, density, friction, initial setting, layout, and "
                    "body-parameter hooks only"
                ),
                "src/scene.py for solver/contact/timestep hooks only",
                (
                    "assets/xml/**/*.xml for validated scalar actuator/joint/geom parameter patches only; no "
                    "topology edits"
                ),
                "contracts/*.json",
                "reports/*.json",
                "artifacts/opt_*",
            ),
            forbidden_changes=(
                "Do not change task semantics or required entities.",
                "Do not directly write dynamic object state after initialization.",
                "Do not add hidden constraints, attachments, suction, fake joints, or task-object teleportation.",
                "Do not add/remove XML bodies, joints, geoms, actuators, meshes, or change XML topology during Opt.",
                "Do not edit src/rendering.py or optimize rendering/camera/visual-only variables.",
                "Do not edit repository-level pipeline code during the Opt pass.",
            ),
            max_rollouts=None,
            backend=str(action.get("backend") or CONFIGS.opt.agent_backend or self.session.config.backend),
            timeout_sec=float(action.get("timeout_sec") or CONFIGS.opt.agent_timeout_sec),
            render_baseline=CONFIGS.opt.agent_render_baseline if render_flag is None else bool(render_flag),
            render_best=CONFIGS.opt.agent_render_best if render_flag is None else bool(render_flag),
            steps=timing.steps,
            duration_sec=timing.duration_sec,
            render_fps=timing.render_fps,
            sim_dt=timing.sim_dt,
            sim_substeps=timing.sim_substeps,
            render_every_n_steps=timing.render_every_n_steps,
            render_res=timing.render_res,
            target_video_frames=timing.target_video_frames,
            success_criteria=tuple(self._planner_success_criteria()),
            simdebug_card_context=simdebug_card_context,
        )
        result = run_opt_agent(request)
        result_payload = asdict(result)
        request_payload = asdict(request)
        request_payload["simdebug_card_context"] = "<omitted; see reports/simdebug_card_dispatch_opt.json>"
        attempts = int(opt_state.get("attempts") or 0) + 1
        opt_state.update(
            {
                "status": result.status,
                "attempts": attempts,
                "latest_result": result_payload,
                "latest_request": self.session.json_safe(request_payload),
                "updated_at_unix": time.time(),
            }
        )
        history = opt_state.setdefault("history", [])
        if isinstance(history, list):
            history.append(
                {"attempt": attempts, "result": result_payload, "request": self.session.json_safe(request_payload)}
            )

        synced_current = None
        if result.status in {"success", "partial_success", "needs_more_optimization"}:
            synced_current = self._sync_best_opt_params_to_current()
            self.session.state["control"]["needs_execution"] = True
            self.session.state["control"]["needs_critic"] = False
            message = (
                "Opt completed and selected parameters for rerun. "
                "Planner should run_execution next so root artifacts reflect the optimized case."
            )
        elif result.status == "needs_rewrite":
            message = "Opt diagnosed a structural issue; Planner should route repair or regeneration using the recommendation."
        else:
            message = "Opt failed or returned inconclusive evidence; Planner should inspect the Opt report."

        return {
            "ok": result.status in {"success", "partial_success", "needs_more_optimization"},
            "status": f"opt_{result.status}",
            "message": message,
            "opt": result_payload,
            "synced_current_opt_params": synced_current,
        }

    def run_command(
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
        cwd = DEFAULT_REPO_ROOT if cwd_choice == "repo" else self.session.case_dir
        timeout_sec = float(
            action.get("timeout_sec") or min(self.session.config.timeout_sec, CONFIGS.harness.command_timeout_sec)
        )
        command = [*executable, *raw_args]
        asset_violations = builtin_asset_violations(command, label=f"{label}_command")
        if asset_violations:
            return {
                "ok": False,
                "status": "forbidden_builtin_asset_reference",
                "message": "Planner command references forbidden Genesis built-in assets.",
                "errors": asset_violations,
            }
        stdout_path = self.session.command_dir / f"turn_{turn:03d}_{label}.stdout.txt"
        stderr_path = self.session.command_dir / f"turn_{turn:03d}_{label}.stderr.txt"
        started = time.time()
        timed_out = False
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=build_local_execution_env({"GENESIS_BACKEND": self.session.config.backend}),
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
            stdout = decode_process_stream(exc.stdout)
            stderr = decode_process_stream(exc.stderr)
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

    def finish(self, action: dict[str, Any]) -> dict[str, Any]:
        assets = self.session.state.get("assets")
        if isinstance(assets, dict) and assets.get("status") == "running":
            return {
                "ok": False,
                "status": "precondition_failed",
                "message": (
                    "Cannot finish while asset generation is still running; choose wait_mesh_assets or wait_xml_assets."
                ),
            }
        verdict = action.get("verdict") or "inconclusive"
        if verdict == "pass":
            critic = self.session.state.get("critic")
            if not isinstance(critic, dict) or critic.get("verdict") != "pass":
                return {"ok": False, "status": "precondition_failed", "message": "finish pass requires critic pass."}
            if not self._final_render_passed():
                return {
                    "ok": False,
                    "status": "precondition_failed",
                    "message": "finish pass requires an accepted final_path_traced render, not only debug physics evidence.",
                }
        if verdict not in {"pass", "fail", "inconclusive"}:
            verdict = "inconclusive"
        self.session.state["status"] = verdict
        self.session.state["stop_reason"] = action.get("summary") or action.get("rationale")
        return {"ok": verdict == "pass", "status": "finished", "verdict": verdict}

    def _opt_planner_intent(self, action: dict[str, Any]) -> str:
        parts = [
            (
                "Optimize the generated case only if the evidence suggests a compact continuous-parameter search can "
                "improve it. Candidate variables may include action controls, initial settings/layout, "
                "material/contact properties, actuator gains/limits, XML scalar parameters, or solver/contact settings."
            ),
            f"Planner rationale: {action.get('rationale') or '<none>'}",
        ]
        notes = action.get("notes")
        if isinstance(notes, list) and notes:
            parts.append("Planner notes: " + "; ".join(str(item) for item in notes))
        critic = self.session.state.get("critic")
        if isinstance(critic, dict):
            parts.append("Latest critic summary: " + str(critic.get("summary") or critic.get("repair_summary") or ""))
            parts.append("Latest critic recommended_owner: " + str(critic.get("recommended_owner") or "none"))
        return "\n".join(parts)

    def _planner_success_criteria(self) -> list[str]:
        planner_output = self.session.current_planner_output()
        scene_brief = planner_output.get("scene_brief") if isinstance(planner_output, dict) else None
        criteria = scene_brief.get("success_criteria") if isinstance(scene_brief, dict) else None
        if isinstance(criteria, list):
            return [str(item) for item in criteria if isinstance(item, str) and item]
        return []

    def _sync_best_opt_params_to_current(self) -> str | None:
        return self.session.sync_best_opt_params_to_current(selected_by="planner.run_opt")

    def _render_profile_from_action(self, action: dict[str, Any]) -> str:
        value = action.get("render_profile")
        if value in {"debug_raster", "final_path_traced"}:
            return str(value)
        return "debug_raster"

    def _bool_from_action(self, value: object, *, default: bool) -> bool:
        if value is None:
            return bool(default)
        return bool(value)

    def _replay_cache_from_action(self, action: dict[str, Any], *, render_profile: str) -> Path | None:
        raw = action.get("replay_cache")
        if isinstance(raw, str) and raw.strip():
            return self._case_relative_path(raw.strip())
        if render_profile == "final_path_traced":
            return self._accepted_state_cache_manifest()
        return None

    def _case_relative_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.session.case_dir / path

    def _accepted_state_cache_manifest(self) -> Path | None:
        physics = self.session.state.get("physics_validation")
        if not isinstance(physics, dict):
            return None
        manifest = physics.get("accepted_state_cache_manifest")
        if not isinstance(manifest, str) or not manifest:
            return None
        path = Path(manifest)
        return path if path.is_absolute() else self.session.case_dir / path

    def _record_render_execution(
        self,
        *,
        render_profile: str,
        replay_cache: Path | None,
        render_only: bool,
    ) -> None:
        if render_profile != "final_path_traced":
            return
        final = self.session.state.setdefault("final_render", {})
        if not isinstance(final, dict):
            final = {}
            self.session.state["final_render"] = final
        final["required"] = bool(self.session.config.render)
        final["status"] = "executed"
        final["attempts"] = int(final.get("attempts") or 0) + 1
        final["latest_render_profile"] = render_profile
        final["latest_replay_cache"] = None if replay_cache is None else str(replay_cache)
        final["latest_render_only"] = render_only
        final["latest_issue"] = None

    def _record_render_critic_result(self, *, critic: dict[str, Any], execution: dict[str, Any]) -> None:
        render_profile = str(execution.get("render_profile") or "debug_raster")
        render_stats_path = self.session.case_dir / "artifacts" / "render_stats.json"
        render_stats = self.session.load_json(render_stats_path) or {}
        if render_profile == "final_path_traced":
            self._record_final_render_critic_result(
                critic=critic,
                render_stats=render_stats,
                render_stats_path=render_stats_path,
            )
            return

        if critic.get("verdict") == "pass":
            physics = self.session.state.setdefault("physics_validation", {})
            if not isinstance(physics, dict):
                physics = {}
                self.session.state["physics_validation"] = physics
            physics["status"] = "passed"
            physics["accepted_at_unix"] = time.time()
            cache_manifest = render_stats.get("state_cache_manifest")
            physics["accepted_state_cache_manifest"] = cache_manifest if isinstance(cache_manifest, str) else None
            physics["accepted_render_stats_path"] = str(render_stats_path)
            final = self.session.state.setdefault("final_render", {})
            if isinstance(final, dict) and final.get("required") and final.get("status") in {None, "pending"}:
                final["status"] = "needed"

    def _record_final_render_critic_result(
        self,
        *,
        critic: dict[str, Any],
        render_stats: dict[str, Any],
        render_stats_path: Path,
    ) -> None:
        final = self.session.state.setdefault("final_render", {})
        if not isinstance(final, dict):
            final = {}
            self.session.state["final_render"] = final
        final["required"] = bool(self.session.config.render)
        final["latest_render_stats_path"] = str(render_stats_path)
        final["latest_render_profile"] = "final_path_traced"
        path_tracing = render_stats.get("path_tracing") if isinstance(render_stats, dict) else None
        path_tracing_enabled = isinstance(path_tracing, dict) and path_tracing.get("enabled") is True
        rendered = bool(render_stats.get("rendered")) if isinstance(render_stats, dict) else False
        if not path_tracing_enabled:
            critic["verdict"] = "fail"
            critic["recommended_owner"] = "rendering"
            critic["summary"] = (
                "Final render was requested with render_profile=final_path_traced, but render_stats.json does not "
                "confirm path_tracing.enabled=true. Repair scene/rendering so Genesis RayTracer is actually used."
            )
            final["status"] = "needs_repair"
            final["latest_issue"] = "path_tracing_not_enabled"
            return
        if not rendered:
            critic["verdict"] = "fail"
            critic["recommended_owner"] = "rendering"
            critic["summary"] = "Final path-traced render did not produce a non-empty video."
            final["status"] = "needs_repair"
            final["latest_issue"] = "final_video_missing_or_empty"
            return
        if critic.get("verdict") == "pass":
            final["status"] = "passed"
            final["passed_at_unix"] = time.time()
            final["latest_issue"] = None
        else:
            final["status"] = "needs_repair"
            final["latest_issue"] = "critic_rejected_final_render_quality"

    def _final_render_passed(self) -> bool:
        final = self.session.state.get("final_render")
        if not isinstance(final, dict) or not final.get("required"):
            return True
        return final.get("status") == "passed"


def planner_requires_asset_manifest(planner_output: dict[str, Any] | None) -> bool:
    if not isinstance(planner_output, dict):
        return False
    dispatch_graph = planner_output.get("dispatch_graph")
    if not isinstance(dispatch_graph, dict):
        return False
    return bool(dispatch_graph.get("wait_for_asset_manifest"))
