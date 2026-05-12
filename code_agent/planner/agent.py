from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from code_agent.configs import CONFIGS
from code_agent.utils.codex import CodexExecRequest, run_codex_exec
from code_agent.utils.general_prompts import (
    FEM_MATERIAL_SELECTION_GUIDE,
    GENERATED_RESULT_QUALITY_GUIDE,
    IPC_FAILURE_DIAGNOSTIC_GUIDE,
    PHYSICAL_CAUSALITY_CONTRACT,
    PLANNER_GENERAL_RULES,
    RIGID_IPC_COUPLING_GUIDE,
    SOURCE_AWARE_REPAIR_GUIDE,
)


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
                timeout_sec=CONFIGS.codex.planner_timeout_sec,
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
            if result.error_type == "codex_usage_limit":
                control = self.session.state.get("control")
                pending = [
                    name
                    for name, active in (control.items() if isinstance(control, dict) else [])
                    if bool(active)
                ]
                blocked = {
                    "type": "codex_usage_limit",
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
                    "rationale": "Planner is blocked by Codex usage limits, not by generated-code execution.",
                    "verdict": "inconclusive",
                    "summary": (
                        "Planner blocked by Codex usage limit; retry after quota resets or credits are available."
                        f"{stale_note} See {result.output_jsonl_path}."
                    ),
                }
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
        sim_substeps = CONFIGS.runtime.sim_substeps
        render_every_n_steps = CONFIGS.runtime.render_every_n_steps
        render_fps = CONFIGS.runtime.render_fps
        render_res = CONFIGS.runtime.render_res
        deformable_enabled = self.session.config.deformable_enabled
        ipc_enabled = self.session.config.ipc_enabled
        deformable_config_text = json.dumps(self.session.deformable_config, indent=2)
        state_text = json.dumps(self._prompt_state(), indent=2)
        genesis_context = self.session.genesis_context_prompt()
        return textwrap.dedent(
            f"""
            {PLANNER_GENERAL_RULES}

            Task:
            {self.session.config.task}

            Genesis documentation and local-code context:
            {genesis_context}

            FEM/IPC capability:
            - FEM deformable enabled: {deformable_enabled}
            - IPC contact/coupling enabled: {ipc_enabled} (forced on whenever FEM deformable is enabled)
            - Effective config contract: {self.session.deformable_config_path}
            - Effective config values:
            {deformable_config_text}
            - If FEM deformable is false and the task fundamentally requires soft-body, jelly, elastic, FEM, or visible
              deformation behavior, choose finish with verdict inconclusive. Do not write a rigid-body substitute.
            - If FEM deformable is true and the task requires soft-body behavior, use FEM+IPC only. Do not use MPM,
              PBD, SPH, cloth-only shortcuts, or rigid-only substitutes.
            - If FEM deformable is false but IPC is true, rigid/articulated contact scenes may use IPC. Tell workers to
              keep bodies rigid/articulated, configure `gs.options.IPCCouplerOptions`, and use rigid IPC coupling
              materials for contact-heavy rigid behavior.
              Use this coupling guide when choosing object roles and body/action contracts:
              {RIGID_IPC_COUPLING_GUIDE}
              Use this IPC failure diagnostic guide when interpreting execution logs:
              {IPC_FAILURE_DIAGNOSTIC_GUIDE}
            - If IPC is false, workers must not instantiate `gs.options.IPCCouplerOptions`.
            - All FEM, IPC, tet, precision, and FEM material-range defaults must come from `deformable_cfg` /
              contracts/deformable_config.json in generated code.
            - FEM elastic material choices must include explicit `E`, `nu`, and `rho` values selected from the
              `deformable_cfg` ranges and defaults. Use this material guide when instructing body/action/critic work:
              {FEM_MATERIAL_SELECTION_GUIDE}

            Available actions:
            - write_plan: create planner_output for this case. Include a complete `planner_output` object matching
              planner_output.schema.json. Infer duration from the task yourself. Use sim_dt={sim_dt},
              sim_substeps={sim_substeps}, render_fps={render_fps}, render_every_n_steps={render_every_n_steps}, and
              render_res={render_res} unless the task explicitly requires different values. Use mode local_gpu and
              backend gpu by default.
              Make the plan detailed enough that each writer can implement its part without guessing: describe desired
              layout, entity identities, physical roles, timing, camera/render expectations, asset orientation/texture
              needs, success criteria, likely failure modes, and per-module validation expectations.
              For objects that genuinely require generated geometry, add `asset_requests` with
              asset_type=`generated_mesh` and set dispatch_graph.wait_for_asset_manifest=true when writers should wait
              for generated mesh paths before writing code. In each asset request, `scale` and `bbox` are positive XYZ
              dimensions in meters only; do not use them for position, lower bounds, centers, or signed extents.
              Meshy mesh generation accepts at most 800 characters in the final mesh-agent prompt. Keep each
              generated_mesh request concise enough that the assembled prompt from name, purpose, simulation_role,
              dimensions, texture_needs, and the automatic simulation-ready geometry suffix stays within that limit.
              Mesh asset generation owns mesh validity. If a generated mesh's manifold, texture transfer, or Genesis
              FEM import validation fails, regenerate that asset through the mesh asset action; do not ask body/scene
              workers to rewrite or procedurally repair the mesh geometry.
              For articulated robots, grippers, gates, latches, actuated mechanisms, or any task object that is best
              represented as one self-contained primitive MJCF body tree with joints and actuators, add an XML/MJCF
              asset request with asset_type=`generated_xml` or `mjcf`. Describe the required joints, actuator semantics,
              base behavior, approximate dimensions, and control affordances in purpose/simulation_role. XML/MJCF asset
              requests are primitive-geom only; do not use them for textured decorative mesh objects.
              When the plan uses XML/MJCF actuators, make the body/action contracts explicit: body must expose stable
              actuator names, joint names, DOF groups, or control handles in `actors`, and action must drive those
              handles with Genesis actuator/DOF/force control APIs after initialization. Do not ask action to create
              motion for an XML articulated asset by overwriting root pose, qpos, or velocities during the simulation.
              If the actuator contract is missing, the generated code should fail clearly so critic can assign a
              source-aware body/action repair.
              Across all task types, direct state writes such as setting entity pose, root qpos, DOF position, or DOF
              velocity are initialization-only. After stepping begins, motion should be expressed through physically
              meaningful controls: actuator commands, DOF controllers, motors, external forces/torques, or initial
              velocities set before the first step.
              {PHYSICAL_CAUSALITY_CONTRACT}
              Any object whose requested appearance depends on texture, patterned surface detail, decorative material
              variation, image-like surface content, or nontrivial visual ornamentation must be represented by a
              Meshy-generated asset request, even when the task does not explicitly say "mesh". Do not ask writers to
              fake those textured objects with plain primitive colors or simple Genesis surfaces.
              Prefer a dispatch_graph that enables the code-writing workers to run together after required assets are
              ready. Treat scene, body, action, and rendering as parallel-capable by default when their contracts
              contain enough shared layout/entity/timing detail; add serial edges only for concrete dependencies that
              truly require seeing another worker's generated source or report.
              Module contract required exports must match the current implementation interfaces exactly:
              scene=`create_scene(backend, *, sim_dt, sim_substeps, deformable_cfg)`;
              body=`create_bodies(scene, task, *, deformable_cfg)`;
              action=`run_actions(scene, actors, *, out_dir, steps, render_state=None)`;
              rendering=`setup_rendering(..., render_every_n_steps, render_res)`, `capture_frame`, and
              `finalize_rendering`.
            - start_mesh_assets: start Planner-requested generated mesh assets in the background and return
              immediately. Use `asset_names` to restrict generation to specific asset request names, or null/[] to
              generate all generated_mesh requests. Prefer this over blocking generation when any writer can make
              progress without the final manifest.
            - wait_mesh_assets: wait for a previously started background mesh asset job to finish and validate
              assets/asset_manifest.json.
            - start_xml_assets: start Planner-requested XML/MJCF assets in the background and return immediately. Use
              `asset_names` to restrict generation to specific XML/MJCF asset request names, or null/[] to generate all
              XML/MJCF requests. XML asset workers are parallel-capable by default, and this asset job may overlap with
              mesh asset jobs and code-writing workers that do not need the manifest yet.
            - wait_xml_assets: wait for a previously started background XML/MJCF asset job to finish and merge its
              partial manifest into assets/asset_manifest.json.
            - spawn_workers: start one or more generation workers. Use `roles` from scene, body, action, rendering.
              Roles in a single spawn_workers action are dispatched concurrently by the harness with no default cap
              beyond the number of requested roles. Prefer maximal safe parallelism: after required assets are ready,
              usually spawn every missing writer role together, because each worker can read planner_output,
              asset_manifest, repository code, and the case workspace. Split dependent work across multiple Planner
              turns only when you can identify a concrete dependency that would make parallel writing likely incorrect.
              If mesh or XML assets are still running, you may still spawn writer roles whose module_contracts do not
              list asset_dependencies or asset_manifest input dependencies. Wait for the relevant asset jobs only before
              spawning roles that need canonical generated mesh or XML paths.
            - run_integrator: wire generated modules into src/main.py.
            - run_execution: run generated code through the harness on the local GPU.
            - run_critic: ask the read-only critic to evaluate execution artifacts.
            - request_repair: send `repair_brief` to the owning worker when critic/execution evidence shows a fix.
              {SOURCE_AWARE_REPAIR_GUIDE}
            - run_python: optional controlled `uv run python ...` command. Use `python_args` and cwd repo/case.
            - run_pytest: optional controlled `uv run pytest ...` command. Use `pytest_args` and cwd repo/case.
            - finish: end the episode with verdict pass, fail, or inconclusive.

            Action policy:
            - If `planner_output_path` is null, choose write_plan.
            - If planner_output dispatch_graph.wait_for_asset_manifest is true and assets.status is not_requested,
              start all required asset families. Use start_mesh_assets for generated_mesh requests and start_xml_assets
              for generated_xml/mjcf requests. You can start one family in one Planner turn and the other in the next
              while the first continues in the background.
            - If one asset family is already running but another required family is absent from assets.jobs, start the
              absent family before waiting, so independent mesh and XML work can overlap.
            - If assets.status is running, choose spawn_workers for non-asset-dependent missing roles, or
              wait_mesh_assets / wait_xml_assets when the next useful writer/integration step requires the manifest.
            - If any generation worker is missing or failed, choose spawn_workers or request_repair for the relevant
              owner.
            - If assets/asset_manifest.json or reports/asset_generation_report.json show a generated_mesh entry with
              status failed, failed manifold validation, failed Genesis FEM import validation, missing/corrupt texture,
              or an unsuitable generated topology, choose start_mesh_assets for the affected asset_names. Do not
              request body/scene/action/rendering repair for mesh-intrinsic defects; those workers should only fix
              placement, material use, controls, or rendering around ready assets.
            - To improve speed, prefer grouping all currently missing writer roles into one spawn_workers action.
              Keep dependencies serial only when a specific worker must inspect another worker's completed source/report
              before it can write a correct module.
            - Only choose run_integrator after scene/body/action/rendering are all ok.
            - Only choose run_execution after integration is current.
            - Only choose run_critic after execution is current.
            - Only choose finish pass after the latest critic verdict is pass.
            - If the latest critic verdict is inconclusive because `codex_result.error_type` is `codex_usage_limit`,
              choose finish with verdict inconclusive; do not request code repair from a missing/blocked critic review.
            - If critic fails and repair budget remains, choose request_repair for the most relevant generation owner.
            - If deterministic artifact checks, stderr, or stdout mention `ipc.initial_penetration`, libuipc initial
              penetration/intersection/thickness/distance/sanity-check failure, choose request_repair for `body` with
              the concrete error details; treat this as an initial placement/clearance issue, not an
              execution-environment issue, unless the logs clearly show a missing dependency or runtime setup failure.
              If this appears together with `IPC rigid state accessor feature is unavailable...`, treat the accessor
              message as secondary to the invalid IPC world unless the same accessor failure is reproduced without any
              initial-geometry or `World is not valid` diagnostics.
            - Prefer run_execution over generic run_python for generated simulations.
            - {GENERATED_RESULT_QUALITY_GUIDE}

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
        usage_blockers = self._codex_usage_limit_blockers()
        if usage_blockers:
            guide.append(
                "Codex usage limit blocked one or more agent calls: "
                f"{', '.join(usage_blockers)}. Choose finish with verdict inconclusive; do not route this as a code "
                "repair failure."
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
                        "Choose write_plan with the same plan but shorter affected generated_mesh asset_requests "
                        "(simplify purpose, simulation_role, and texture_needs once), then retry start_mesh_assets "
                        "for those asset_names."
                    )
                    return guide
                guide.append(
                    "asset generation failed: restart the failed asset family, repair the plan, or finish fail."
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
                guide.append("critic did not pass: repair if budget remains, otherwise finish fail.")
        return guide

    def _codex_usage_limit_blockers(self) -> list[str]:
        blockers: list[str] = []
        critic = self.session.state.get("critic")
        if isinstance(critic, dict):
            codex_report = critic.get("codex_critic_report")
            codex_result = codex_report.get("codex_result") if isinstance(codex_report, dict) else None
            if isinstance(codex_result, dict) and codex_result.get("error_type") == "codex_usage_limit":
                blockers.append("critic")
        workers = self.session.state.get("workers")
        if isinstance(workers, dict):
            for role, data in workers.items():
                if not isinstance(data, dict) or data.get("ok"):
                    continue
                codex = data.get("codex")
                if isinstance(codex, dict) and codex.get("error_type") == "codex_usage_limit":
                    blockers.append(f"{role}_worker")
        return blockers

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
