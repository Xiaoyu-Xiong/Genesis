from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_agent.configs import CONFIGS, runtime_defaults_dict
from code_agent.io_utils import load_json_object
from code_agent.utils.codex import DEFAULT_REPO_ROOT, CodexExecRequest, run_codex_exec
from code_agent.prompts.planner import (
    PLANNER_ACTION_POLICY_GUIDE,
    PLANNER_GENERAL_RULES,
    planner_available_actions_section,
    planner_fem_ipc_capability_section,
)


class EpisodePlanner:
    """Build Planner prompts and request the next structured episode action."""

    def __init__(self, session: Any):
        self.session = session

    def request_action(self, turn: int) -> dict[str, Any]:
        result = run_codex_exec(
            CodexExecRequest(
                role=f"planner_turn_{turn:03d}",
                prompt=self._planner_prompt(turn=turn),
                cwd=DEFAULT_REPO_ROOT,
                sandbox=CONFIGS.codex.planner_sandbox,
                model=CONFIGS.codex.planner_model,
                output_schema_path=Path("code_agent/specs/planner_action.schema.json"),
                output_jsonl_path=self.session.logs_dir / f"codex_planner_turn_{turn:03d}.jsonl",
                final_message_path=self.session.logs_dir / f"codex_planner_turn_{turn:03d}.final.json",
                timeout_sec=CONFIGS.codex.planner_timeout_sec,
                writable_roots=(self.session.case_dir,),
            )
        )
        planner_invocations = self.session.state.setdefault("planner_invocations", [])
        planner_invocations.append(
            {
                "turn": turn,
                "returncode": result.exit_code,
                "ok": result.success,
                "duration_sec": result.duration_sec,
                "final_message_path": result.final_message_path,
                "stderr_path": result.stderr_path,
                "error_type": result.error_type,
                "error_message": result.error_message,
            }
        )
        if not result.success:
            if result.error_type in {"codex_usage_limit", "codex_auth_failed", "codex_input_too_large", "timeout"}:
                control = self.session.state.get("control")
                pending = [
                    name
                    for name, active in (control.items() if isinstance(control, dict) else [])
                    if bool(active)
                ]
                blocked = {
                    "type": result.error_type,
                    "message": result.error_message,
                    "turn": turn,
                    "output_jsonl_path": result.output_jsonl_path,
                    "stderr_path": result.stderr_path,
                    "pending_work": pending,
                    "latest_artifacts_may_be_stale": bool(pending),
                }
                self.session.state["blocked_reason"] = blocked
                stale_note = (
                    f" Pending work: {', '.join(pending)}. Latest artifacts may be stale relative to source."
                    if pending
                    else ""
                )
                return {
                    "action": "finish",
                    "rationale": "Planner is blocked by Codex infrastructure, not by generated-code execution.",
                    "verdict": "inconclusive",
                    "summary": (
                        f"Planner blocked by Codex infrastructure ({result.error_type}); retry this case later."
                        f"{stale_note} See {result.output_jsonl_path}."
                    ),
                }
            return {
                "action": "finish",
                "rationale": f"Planner invocation failed with exit code {result.exit_code}.",
                "verdict": "fail",
                "summary": f"Planner failed; see {result.stderr_path}.",
            }
        payload = load_json_object(Path(result.final_message_path))
        if payload is None:
            return {
                "action": "finish",
                "rationale": "Planner did not return parseable JSON.",
                "verdict": "fail",
                "summary": f"Invalid planner final message: {result.final_message_path}",
            }
        return payload

    def _planner_prompt(self, *, turn: int | None = None) -> str:
        non_ipc_defaults = runtime_defaults_dict(ipc_enabled=False)
        ipc_defaults = runtime_defaults_dict(ipc_enabled=True)
        deformable_enabled = bool(self.session.deformable_config.get("enabled"))
        ipc_enabled = bool(self.session.deformable_config.get("ipc_enabled"))
        deformable_config_text = json.dumps(self.session.deformable_config, indent=2)
        prompt_state = self._prompt_state()
        state_text = json.dumps(prompt_state, indent=2)
        genesis_context = self.session.genesis_context_prompt()
        layout_context = self.session.layout_context_prompt()
        simdebug_context = self._simdebug_context_prompt(prompt_state, turn=turn)
        capability_section = planner_fem_ipc_capability_section(
            deformable_enabled=deformable_enabled,
            ipc_enabled=ipc_enabled,
            physics_modes=self.session.simdebug_physics_modes(),
            deformable_config_path=self.session.deformable_config_path,
            deformable_config_text=deformable_config_text,
        )
        actions_section = planner_available_actions_section(
            non_ipc_defaults=non_ipc_defaults,
            ipc_defaults=ipc_defaults,
            opt_enabled=self.session.config.opt_enabled,
        )
        return "\n\n".join(
            [
                PLANNER_GENERAL_RULES,
                f"Task:\n{self.session.config.task}",
                (
                    "User-provided layout context:\n"
                    f"{layout_context}\n\n"
                    "The planner_output schema has no dedicated top-level layout field, so summarize the relevant "
                    "layout constraints inside scene_brief, scene_plan, module_contracts.input_dependencies, "
                    "validation_expectation, execution_plan.notes, or risk_register as appropriate. If the layout "
                    "prepared ready repo_asset entries in the asset manifest, reuse those logical_names rather than "
                    "requesting generated_mesh replacements."
                    if layout_context
                    else ""
                ),
                f"Genesis documentation and local-code context:\n{genesis_context}",
                simdebug_context if simdebug_context else "",
                capability_section,
                actions_section,
                PLANNER_ACTION_POLICY_GUIDE,
                f"Current episode state:\n{state_text}",
            ]
        ).strip()

    def _simdebug_context_prompt(self, prompt_state: dict[str, Any], *, turn: int | None = None) -> str:
        if not self.session.simdebug_cards_enabled():
            return ""
        return self.session.simdebug_card_context_for_role(
            "planner",
            turn=turn,
            dispatch_reason="planner_prompt",
            extra_state={"prompt_state": prompt_state},
        )

    def _prompt_state(self) -> dict[str, Any]:
        state = dict(self.session.state)
        state["observations"] = self.session.state.get("observations", [])
        state["commands"] = self.session.state.get("commands", [])
        state["asset_manifest"] = self._asset_manifest()
        state["harness_guide"] = self._harness_guide()
        return state

    def _harness_guide(self) -> list[str]:
        state = self.session.state
        guide: list[str] = []
        if state.get("planner_output_path") is None:
            guide.append("planner_output missing: next valid action is write_plan.")
            return guide
        infra_blockers = self._codex_infra_blockers()
        if infra_blockers:
            guide.append(
                "Codex or critic infrastructure blocked one or more required agent calls: "
                f"{', '.join(infra_blockers)}. Choose finish with verdict inconclusive; do not route this as a "
                "generated-code repair failure."
            )
            return guide
        planner_output = self.session.current_planner_output()
        if self._planner_waits_for_asset_manifest(planner_output) and not self.session.asset_manifest_ready():
            assets = state.get("assets") if isinstance(state.get("assets"), dict) else {}
            asset_status = assets.get("status")
            if asset_status == "not_requested":
                guide.append(
                    "planner_output requests generated assets: start the required mesh and/or XML asset jobs, then use "
                    "later turns for non-asset-dependent writers while background asset jobs run."
                )
                return guide
            if asset_status == "running":
                guide.append(
                    "asset jobs are running in the background: spawn non-asset-dependent missing writers now, or "
                    "choose wait_mesh_assets / wait_xml_assets before a manifest-dependent role/integration."
                )
            elif asset_status == "failed":
                failure_classes = assets.get("failure_classes")
                if isinstance(failure_classes, list) and "mesh.prompt_length_exceeded" in failure_classes:
                    guide.append(
                        "mesh asset generation failed because the Meshy prompt exceeded the 800-character limit. "
                        "Choose start_mesh_assets for the affected asset_names and include a complete revised "
                        "planner_output whose affected generated_mesh request is rewritten shorter. Do not append "
                        "feedback text to the old prompt."
                    )
                    return guide
                guide.append(
                    "asset generation failed: retry the failed asset family with a revised planner_output that "
                    "rewrites the affected asset request, or finish fail if the request is infeasible."
                )
                return guide
            else:
                guide.append(
                    "asset manifest is not ready: choose start/wait actions for the required mesh or XML asset family."
                )
                return guide
        missing = [role for role, data in state["workers"].items() if not data.get("ok")]
        if missing:
            guide.append(
                "workers not ok: "
                f"{', '.join(missing)}. Prefer one spawn_workers action containing all of them unless a concrete "
                "source/report dependency requires serial scheduling."
            )
            return guide
        control = state["control"]
        if control.get("needs_integration"):
            guide.append("worker source changed: run_integrator is needed.")
        elif control.get("needs_execution"):
            guide.append("integrated project is ready: run_execution is needed.")
        elif control.get("needs_critic"):
            guide.append("execution is ready: run_critic is needed.")
        else:
            critic = state.get("critic")
            if isinstance(critic, dict) and critic.get("verdict") == "pass":
                guide.append("critic passed: finish pass is valid.")
            elif isinstance(critic, dict):
                opt_state = state.get("opt")
                opt_attempts = int(opt_state.get("attempts") or 0) if isinstance(opt_state, dict) else 0
                opt_status = str(opt_state.get("status") or "") if isinstance(opt_state, dict) else ""
                if self.session.config.opt_enabled and opt_attempts == 0:
                    guide.append(
                        "critic did not pass: choose run_opt if the case is runnable and the remaining miss looks like "
                        "a continuous parameter/control residual with real handles in action/body/scene; choose repair "
                        "for structural defects such as invalid assets, missing entities, impossible geometry, missing "
                        "metrics, or invisible behavior."
                    )
                elif self.session.config.opt_enabled and opt_status == "needs_rewrite":
                    guide.append(
                        "Opt reported needs_rewrite: use its recommendation to route source repair or asset "
                        "regeneration if budget remains."
                    )
                else:
                    guide.append("critic did not pass: repair if budget remains, otherwise finish fail.")
        return guide

    def _codex_infra_blockers(self) -> list[str]:
        blockers: list[str] = []
        critic = self.session.state.get("critic")
        if isinstance(critic, dict):
            infra_status = critic.get("critic_infra_status")
            if critic.get("verdict") == "inconclusive" and isinstance(infra_status, str) and infra_status != "ok":
                blockers.append(f"critic:{infra_status}")
        workers = self.session.state.get("workers")
        if isinstance(workers, dict):
            for role, data in workers.items():
                if not isinstance(data, dict) or data.get("ok"):
                    continue
                codex = data.get("codex")
                if isinstance(codex, dict):
                    error_type = codex.get("error_type")
                    if error_type in {"codex_usage_limit", "codex_auth_failed", "codex_input_too_large", "timeout"}:
                        blockers.append(f"{role}_worker:{error_type}")
        return blockers

    def _asset_manifest(self) -> dict[str, Any] | None:
        assets = self.session.state.get("assets")
        if not isinstance(assets, dict):
            return None
        manifest_path = assets.get("asset_manifest_path")
        if not isinstance(manifest_path, str):
            return None
        manifest = load_json_object(Path(manifest_path))
        if manifest is None:
            return None
        raw_assets = manifest.get("assets")
        asset_entries = raw_assets if isinstance(raw_assets, list) else []
        return {
            "asset_manifest_path": manifest_path,
            "assets": asset_entries,
            "unresolved_risks": manifest.get("unresolved_risks", []),
        }

    def _planner_waits_for_asset_manifest(self, planner_output: dict[str, Any] | None) -> bool:
        if not isinstance(planner_output, dict):
            return False
        dispatch_graph = planner_output.get("dispatch_graph")
        if not isinstance(dispatch_graph, dict):
            return False
        return bool(dispatch_graph.get("wait_for_asset_manifest"))
