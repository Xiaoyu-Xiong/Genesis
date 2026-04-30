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
                timeout_sec=900.0,
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
        state_text = json.dumps(self._prompt_state(), indent=2)
        genesis_context = self.session.genesis_context_prompt()
        return textwrap.dedent(
            f"""
            You are the Planner Agent for one Genesis code-generation episode.
            The full repository and current case workspace are available for context. You may inspect files, source,
            reports, logs, assets, and generated artifacts when deciding the next action. You may run lightweight
            read-only inspection commands when useful, but do not mutate files or run expensive simulations yourself.
            Return one JSON action only, matching planner_action.schema.json. The Python harness will execute the action
            and call you again with updated state.
            Do not overly compress planning details to save tokens; detailed instructions are preferred when they help
            downstream workers make correct source-level decisions.
            Include every schema field in every response. Use null for irrelevant scalar/object fields and [] for
            irrelevant array fields.

            Task:
            {self.session.config.task}

            Genesis documentation and local-code context:
            {genesis_context}

            Available actions:
            - write_plan: create planner_output for this case. Include a complete `planner_output` object matching
              planner_output.schema.json. Infer duration from the task yourself. Use sim_dt={sim_dt};
              use render_fps={render_fps} unless the task explicitly asks for another fps. Use mode local_gpu and
              backend gpu by default.
              Make the plan detailed enough that each writer can implement its part without guessing: describe desired
              layout, entity identities, physical roles, timing, camera/render expectations, asset orientation/texture
              needs, success criteria, likely failure modes, and per-module validation expectations.
              For objects that genuinely require generated geometry, add `asset_requests` with
              asset_type=`generated_mesh` and set dispatch_graph.wait_for_asset_manifest=true when writers should wait
              for generated mesh paths before writing code. In each asset request, `scale` and `bbox` are positive XYZ
              dimensions in meters only; do not use them for position, lower bounds, centers, or signed extents.
              Any object whose requested appearance depends on texture, patterned surface detail, decorative material
              variation, image-like surface content, or nontrivial visual ornamentation must be represented by a
              Meshy-generated asset request, even when the task does not explicitly say "mesh". Do not ask writers to
              fake those textured objects with plain primitive colors or simple Genesis surfaces.
              Prefer a dispatch_graph that enables the code-writing workers to run together after required assets are
              ready. Treat scene, body, action, and rendering as parallel-capable by default when their contracts
              contain enough shared layout/entity/timing detail; add serial edges only for concrete dependencies that
              truly require seeing another worker's generated source or report.
              Module contract required exports must match the current implementation interfaces exactly:
              scene=`create_scene`; body=`create_bodies`; action=`run_actions`; rendering=`setup_rendering`,
              `capture_frame`, and `finalize_rendering`.
            - start_mesh_assets: start Planner-requested generated mesh assets in the background and return
              immediately. Use `asset_names` to restrict generation to specific asset request names, or null/[] to
              generate all generated_mesh requests. Prefer this over blocking generation when any writer can make
              progress without the final manifest.
            - generate_mesh_assets: compatibility action that starts mesh assets and waits for completion in the same
              turn. Use it only when you deliberately want to serialize asset generation before all writers.
            - wait_mesh_assets: wait for a previously started background mesh asset job to finish and validate
              assets/asset_manifest.json.
            - spawn_workers: start one or more generation workers. Use `roles` from scene, body, action, rendering.
              Roles in a single spawn_workers action are dispatched concurrently by the harness with no default cap
              beyond the number of requested roles. Prefer maximal safe parallelism: after required assets are ready,
              usually spawn every missing writer role together, because each worker can read planner_output,
              asset_manifest, repository code, and the case workspace. Split dependent work across multiple Planner
              turns only when you can identify a concrete dependency that would make parallel writing likely incorrect.
              If mesh assets are still running, you may still spawn writer roles whose module_contracts do not list
              asset_dependencies or asset_manifest input dependencies. Wait for mesh assets only before spawning roles
              that need canonical generated mesh paths.
            - run_integrator: wire generated modules into src/main.py.
            - run_execution: run generated code through the harness on the local GPU.
            - run_critic: ask the read-only critic to evaluate execution artifacts.
            - request_repair: send `repair_brief` to the owning worker when critic/execution evidence shows a fix.
              Repair briefs must be detailed, source-aware, and actionable: compare the original text prompt, the latest
              execution/visual output, metrics/event logs, and relevant generated source. Tell the target worker exactly
              what behavior is wrong, what evidence proves it, which module boundary owns it, and what a convincing fix
              should accomplish. Avoid vague instructions like "improve the trajectory" when concrete evidence exists.
            - run_python: optional controlled `uv run python ...` command. Use `python_args` and cwd repo/case.
            - run_pytest: optional controlled `uv run pytest ...` command. Use `pytest_args` and cwd repo/case.
            - finish: end the episode with verdict pass, fail, or inconclusive.

            Action policy:
            - If `planner_output_path` is null, choose write_plan.
            - If planner_output dispatch_graph.wait_for_asset_manifest is true and assets.status is not_requested,
              prefer start_mesh_assets first. On the following turn, spawn any missing writer roles that do not require
              the asset manifest while the asset job continues in the background.
            - If assets.status is running, choose spawn_workers for non-asset-dependent missing roles, or
              wait_mesh_assets when the next useful writer/integration step requires the manifest.
            - If any generation worker is missing or failed, choose spawn_workers or request_repair for the relevant
              owner.
            - To improve speed, prefer grouping all currently missing writer roles into one spawn_workers action.
              Keep dependencies serial only when a specific worker must inspect another worker's completed source/report
              before it can write a correct module.
            - Only choose run_integrator after scene/body/action/rendering are all ok.
            - Only choose run_execution after integration is current.
            - Only choose run_critic after execution is current.
            - Only choose finish pass after the latest critic verdict is pass.
            - If critic fails and repair budget remains, choose request_repair for the most relevant generation owner.
            - Prefer run_execution over generic run_python for generated simulations.
            - The final simulation should not merely satisfy numeric proxies. It should match the input text prompt and
              look physically and visually reasonable, coherent, and logically staged.

            Current episode state:
            {state_text}
            """
        ).strip()

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
        planner_output = self.session.current_planner_output()
        if self._planner_waits_for_asset_manifest(planner_output) and not self.session.asset_manifest_ready():
            assets = state.get("assets") if isinstance(state.get("assets"), dict) else {}
            asset_status = assets.get("status")
            if asset_status == "not_requested":
                guide.append(
                    "planner_output requests generated assets: prefer start_mesh_assets, then use later turns for "
                    "non-asset-dependent writers while the background asset job runs."
                )
                return guide
            if asset_status == "running":
                guide.append(
                    "mesh assets are running in the background: spawn non-asset-dependent missing writers now, or "
                    "choose wait_mesh_assets before a manifest-dependent role/integration."
                )
            elif asset_status == "failed":
                guide.append("mesh asset generation failed: restart assets, repair the plan, or finish fail.")
                return guide
            else:
                guide.append(
                    "asset manifest is not ready: choose start_mesh_assets or wait_mesh_assets as appropriate."
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

    def _asset_manifest(self) -> dict[str, Any] | None:
        assets = self.session.state.get("assets")
        if not isinstance(assets, dict):
            return None
        manifest_path = assets.get("asset_manifest_path")
        if not isinstance(manifest_path, str):
            return None
        manifest = self._load_json(Path(manifest_path))
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
