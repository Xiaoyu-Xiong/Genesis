from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from code_agent.utils.codex import CodexExecRequest, run_codex_exec
from code_agent.configs import CONFIGS


class EpisodePlanner:
    """Build Planner prompts and request the next structured episode action."""

    def __init__(self, session: Any):
        self.session = session

    def request_action(self, turn: int) -> dict[str, Any]:
        result = run_codex_exec(
            CodexExecRequest(
                role=f"planner_turn_{turn:03d}",
                prompt=self._planner_prompt(),
                cwd=Path.cwd(),
                sandbox=CONFIGS.codex.planner_sandbox,
                model=CONFIGS.codex.planner_model,
                output_schema_path=Path("code_agent/specs/planner_action.schema.json"),
                output_jsonl_path=self.session.logs_dir / f"codex_planner_turn_{turn:03d}.jsonl",
                final_message_path=self.session.logs_dir / f"codex_planner_turn_{turn:03d}.final.json",
                timeout_sec=300.0,
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
            }
        )
        if not result.success:
            return {
                "action": "finish",
                "rationale": f"Planner invocation failed with exit code {result.exit_code}.",
                "verdict": "fail",
                "summary": f"Planner failed; see {result.stderr_path}.",
            }
        payload = self._load_json(Path(result.final_message_path))
        if payload is None:
            return {
                "action": "finish",
                "rationale": "Planner did not return parseable JSON.",
                "verdict": "fail",
                "summary": f"Invalid planner final message: {result.final_message_path}",
            }
        return payload

    def _planner_prompt(self) -> str:
        sim_dt = CONFIGS.runtime.sim_dt
        render_fps = CONFIGS.runtime.render_fps
        state_excerpt = json.dumps(self._prompt_state(), indent=2)
        return textwrap.dedent(
            f"""
            You are the Planner Agent for one Genesis code-generation episode.
            You do not edit files and do not run shell commands directly. Return one JSON action only, matching
            planner_action.schema.json. The Python harness will execute the action and call you again with updated state.
            Include every schema field in every response. Use null for irrelevant scalar/object fields and [] for
            irrelevant array fields.

            Task:
            {self.session.config.task}

            Available actions:
            - write_plan: create planner_output for this case. Include a complete `planner_output` object matching
              planner_output.schema.json. Infer duration from the task yourself. Use sim_dt={sim_dt}; use render_fps={render_fps}
              unless the task explicitly asks for another fps. Use mode local_gpu and backend gpu by default.
              Module contract required exports must match the current implementation interfaces exactly:
              scene=`create_scene`; body=`create_bodies`; action=`run_actions`; rendering=`setup_rendering`,
              `capture_frame`, and `finalize_rendering`.
            - spawn_workers: start one or more generation workers. Use `roles` from scene, body, action, rendering.
            - run_integrator: wire generated modules into src/main.py.
            - run_execution: run generated code through the harness on the local GPU.
            - run_critic: ask the read-only critic to evaluate execution artifacts.
            - request_repair: send `repair_brief` to the owning worker when critic/execution evidence shows a fix.
            - run_python: optional controlled `uv run python ...` command. Use `python_args` and cwd repo/case.
            - run_pytest: optional controlled `uv run pytest ...` command. Use `pytest_args` and cwd repo/case.
            - finish: end the episode with verdict pass, fail, or inconclusive.

            Action policy:
            - If `planner_output_path` is null, choose write_plan.
            - If any generation worker is missing or failed, choose spawn_workers or request_repair for the relevant owner.
            - Only choose run_integrator after scene/body/action/rendering are all ok.
            - Only choose run_execution after integration is current.
            - Only choose run_critic after execution is current.
            - Only choose finish pass after the latest critic verdict is pass.
            - If critic fails and repair budget remains, choose request_repair for the most relevant generation owner.
            - Prefer run_execution over generic run_python for generated simulations.

            Current episode state:
            {state_excerpt}
            """
        ).strip()

    def _prompt_state(self) -> dict[str, Any]:
        state = dict(self.session.state)
        state["observations"] = self.session.state.get("observations", [])[-8:]
        state["commands"] = self.session.state.get("commands", [])[-5:]
        state["harness_guide"] = self._harness_guide()
        return state

    def _harness_guide(self) -> list[str]:
        state = self.session.state
        guide: list[str] = []
        if state.get("planner_output_path") is None:
            guide.append("planner_output missing: next valid action is write_plan.")
            return guide
        missing = [role for role, data in state["workers"].items() if not data.get("ok")]
        if missing:
            guide.append(f"workers not ok: {', '.join(missing)}.")
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
                guide.append("critic did not pass: repair if budget remains, otherwise finish fail.")
        return guide

    def _load_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
