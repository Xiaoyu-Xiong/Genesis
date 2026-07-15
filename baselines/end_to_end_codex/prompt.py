from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from baselines.end_to_end_codex.configs import (
    DEFAULT_EXECUTION_TIMEOUT_SEC,
    deformable_config_dict,
    runtime_defaults_dict,
)
from baselines.end_to_end_codex.prompt_clauses import (
    BUILTIN_ASSET_POLICY_GUIDE,
    COLLISION_CONTACT_CONTRACT,
    GENESIS_IMPLEMENTATION_GUIDE,
    GENERATED_RESULT_QUALITY_GUIDE,
    PHYSICAL_CAUSALITY_CONTRACT,
    SCALE_POLICY_GUIDE,
)


def build_end_to_end_prompt(
    *,
    case_id: str,
    task: str,
    case_dir: Path,
    backend: str,
    render: bool,
    steps: int | None,
    duration_sec: float | None,
    render_fps: int | None,
    genesis_context: str,
    layout_context: str = "",
) -> str:
    """Build the single-agent prompt for the end-to-end baseline."""

    non_ipc_defaults = runtime_defaults_dict(ipc_enabled=False)
    ipc_defaults = runtime_defaults_dict(ipc_enabled=True)
    cli_overrides = {
        "backend": backend,
        "render": render,
        "steps": steps,
        "duration_sec": duration_sec,
        "render_fps": render_fps,
    }
    timing_args = _timing_args(steps=steps, duration_sec=duration_sec, render_fps=render_fps)
    render_arg = "--render" if render else "--no-render"
    prepare_command = (
        "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python "
        f"-m baselines.end_to_end_codex.case_tools prepare-contracts --case-dir {case_dir}{timing_args}"
    )
    mesh_command = (
        "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python "
        f"-m baselines.end_to_end_codex.case_tools generate-mesh-assets --case-dir {case_dir}"
    )
    adaptive_ipc_command = (
        "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python "
        f"-m baselines.end_to_end_codex.case_tools apply-adaptive-ipc --case-dir {case_dir}"
    )
    simulation_command = (
        "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python "
        f"-m baselines.end_to_end_codex.case_tools run-simulation --case-dir {case_dir} --backend {backend} "
        f"--timeout-sec {DEFAULT_EXECUTION_TIMEOUT_SEC:g} {render_arg}{timing_args}"
    )
    return textwrap.dedent(
        f"""
        You are the only Codex code-generation agent for one Genesis simulation baseline case.
        There is no separate Planner, worker split, repair worker, Opt worker, or critic in this baseline. You are the
        merged agent: plan the scene, write all code, generate required Meshy assets through the provided harness tool,
        run simulations through the provided locked harness tool, inspect evidence, repair your own code, and deliver
        the final runnable case.

        Task prompt:
        {task}

        Case:
        - case_id: {case_id}
        - case workspace: {case_dir}
        - generated source root: {case_dir / "src"}
        - contracts root: {case_dir / "contracts"}
        - reports root: {case_dir / "reports"}
        - assets root: {case_dir / "assets"}
        - artifacts root: {case_dir / "artifacts"}
        - requested runner overrides: {json.dumps(cli_overrides, indent=2)}

        {"User-provided layout context:" if layout_context else ""}
        {layout_context}

        Genesis documentation and local-code context:
        {genesis_context}

        Required writes:
        1. Write `{case_dir / "src" / "main.py"}` as a standalone generated entrypoint.
        2. Write `{case_dir / "contracts" / "planner_output.json"}` matching `code_agent/specs/planner_output.schema.json`.
        3. If useful, write small helper modules under `{case_dir / "src"}` only. Do not edit repository source.

        Final response:
        Return JSON matching `code_agent/specs/worker_report.schema.json`.
        Use role `end_to_end_codex`. Set status `completed` only after all required files are written and the latest
        locked `run-simulation` report has `ok=true` / `status="execution_passed"`.
        Use status `partial` if code is written but the latest simulation still fails, times out, was not run, or lacks
        required artifacts. Use status `blocked` only for genuine external blockers such as repeated Codex/tool
        infrastructure failure, unavailable assets, or repeated identical environment failure that source edits cannot
        address. In either non-completed case, explain the blocker or failing evidence in `unresolved_risks`.
        Include changed_files with paths relative to the case workspace, for example `src/main.py` and
        `contracts/planner_output.json`.

        Execution contract for `src/main.py`:
        - It must parse these arguments: `--backend`, `--out-dir`, `--steps`, `--fps`/`--render-fps`,
          `--duration-sec`, `--target-video-frames`, `--sim-dt`, `--sim-substeps`,
          `--render-every-n-steps`, `--render-res WIDTH HEIGHT`, `--deformable-config`, `--render`, `--no-render`.
        - It will be run from the case workspace by the repository harness through
          `uv run --no-sync python src/main.py ...`.
        - It must initialize Genesis from those arguments and write outputs under `args.out_dir`.
        - Write useful evidence files when possible: `metrics.json`, `event_log.json`, `render_stats.json`, and a
          video or frames when rendering is enabled.
        - You should run the simulation yourself while debugging, but only through the locked baseline command below.
          The suite harness will also run one final official execution after your Codex call finishes.

        Planner-output contract:
        - Choose `physics_plan.mode` yourself: `rigid`, `rigid_ipc`, or `fem_ipc`.
        - Choose `physics_plan.deformable_kind` yourself: `none`, `soft_body`, `cloth`, or `soft_body_and_cloth`.
          There is no separate baseline cloth enable flag. Generated code should branch from this planner choice and
          from `contracts/deformable_config.json` when needed.
        - Use ordinary rigid mode for normal rigid scenes, rigid IPC only for unusually demanding rigid/articulated
          contact, and FEM+IPC for any soft-body, deformable, or FEM.Cloth behavior.
        - Put explicit runtime values in `execution_plan`. Use these defaults unless the task needs different values:
          non-IPC={json.dumps(non_ipc_defaults)}, IPC/FEM={json.dumps(ipc_defaults)}.
        - If the runner supplied `steps`, `duration_sec`, or `render_fps`, make the execution plan compatible with
          those overrides.
        - Include any generated 3D asset needs in `asset_requests`. You may call the existing Meshy mesh asset pipeline
          yourself through the provided command below, and the baseline harness may also call it after your Codex call
          to ensure the final manifest is ready.
        - For Meshy assets, use `name`, `asset_type: "generated_mesh"`, and concise prompt fields. Use `bbox` for
          approximate positive XYZ dimensions in meters and `scale: null` unless a uniform scalar is truly needed.
        - Generated code that uses Meshy assets must read `assets/asset_manifest.json` at runtime and select ready
          entries by `logical_name`; do not hard-code provider output paths.

        Effective deformable config examples:
        - rigid: {json.dumps(deformable_config_dict(physics_mode="rigid"), sort_keys=True)}
        - fem_ipc enabled fields follow the same contract and are available in `contracts/deformable_config.json`
          before simulation.

        Iterative self-debugging commands:
        - Prepare/update contracts after writing `contracts/planner_output.json`:
          `{prepare_command}`
        - Generate current Meshy asset requests after writing `asset_requests`:
          `{mesh_command}`
        - Apply adaptive IPC contact d-hat after preparing contracts, if IPC is enabled:
          `{adaptive_ipc_command}`
        - Run the generated simulation through the shared Genesis execution lock:
          `{simulation_command}`
        - If simulation needs Meshy assets and you have not generated them yet, add `--ensure-mesh-assets` to the
          run-simulation command.
        - Read `reports/baseline_agent_simulation_report.json`, `reports/execution_report.json`, `reports/stdout.txt`,
          `reports/stderr.txt`, `artifacts/metrics.json`, `artifacts/event_log.json`, and render artifacts after each
          run.
        - You must run `run-simulation` at least once after writing `src/main.py` and `planner_output.json`, and rerun
          it after every source, contract, asset, or timing change that could affect execution.
        - A failed, timed-out, skipped, or artifact-missing simulation is not an acceptable final state. Inspect the
          newest stdout/stderr/execution report, repair the source or contracts, and rerun.
        - Continue this repair loop until the latest locked simulation passes and writes the requested evidence, or
          until a hard external blocker remains after concrete repair attempts. Do not mark `completed` based only on
          code inspection, an unverified short smoke test, or a previous failing run.
        - All tool calls append to `reports/baseline_agent_tool_history.jsonl`, which is useful final evidence.

        Allowed commands while writing:
        - You may inspect files with read-only shell commands such as `pwd`, `ls`, `find`, `rg`, `sed`, and `cat`.
        - You may run the exact baseline tool commands above. Do not run direct `python src/main.py`, direct
          `uv run ... src/main.py`, pytest, or any other Genesis/rendering/simulation command that bypasses the shared
          execution lock.
        - Do not mutate `.venv` or repository files outside the case workspace.

        Core physical and asset rules:
        {PHYSICAL_CAUSALITY_CONTRACT}

        {COLLISION_CONTACT_CONTRACT}

        {SCALE_POLICY_GUIDE}

        {BUILTIN_ASSET_POLICY_GUIDE}

        {GENESIS_IMPLEMENTATION_GUIDE}

        {GENERATED_RESULT_QUALITY_GUIDE}
        """
    ).strip()


def _timing_args(*, steps: int | None, duration_sec: float | None, render_fps: int | None) -> str:
    args: list[str] = []
    if steps is not None:
        args.extend(["--steps", str(int(steps))])
    if duration_sec is not None:
        args.extend(["--duration-sec", str(float(duration_sec))])
    if render_fps is not None:
        args.extend(["--render-fps", str(int(render_fps))])
    return "" if not args else " " + " ".join(args)


def load_layout_context(case_dir: Path) -> str:
    path = case_dir / "inputs" / "layout_context.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def load_genesis_context_summary(case_dir: Path) -> str:
    context_md = case_dir / "contracts" / "genesis_context.md"
    context_json = case_dir / "contracts" / "genesis_context.json"
    payload: dict[str, Any] = {}
    if context_json.exists():
        try:
            raw = json.loads(context_json.read_text(encoding="utf-8"))
            payload = raw if isinstance(raw, dict) else {}
        except json.JSONDecodeError:
            payload = {}
    return "\n".join(
        [
            "Genesis official-doc and local-source context is available on disk for on-demand reading.",
            f"- Context index: {context_md}",
            f"- Machine-readable context JSON: {context_json}",
            f"- Cached official docs directory: {payload.get('docs_dir', '<see context JSON>')}",
            f"- Selected official-doc catalog: {payload.get('catalog_path', '<see context JSON>')}",
            "- Prefer local Genesis source and examples over online docs if they disagree.",
        ]
    )
