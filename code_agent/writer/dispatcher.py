from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import textwrap
from pathlib import Path

from code_agent.io_utils import load_json_object
from code_agent.utils.codex import DEFAULT_REPO_ROOT, CodexExecRequest, run_codex_exec
from code_agent.configs import CONFIGS

from . import action, body, rendering, scene
from code_agent.utils.general_prompts import FEM_IPC_API_GUIDE, RIGID_API_GUIDE, WORKER_COMMON_RULES

from .common import WorkerDispatchResult, WorkerRole, WorkerSpec

WORKERS: dict[WorkerRole, WorkerSpec] = {
    "scene": scene.SPEC,
    "body": body.SPEC,
    "action": action.SPEC,
    "rendering": rendering.SPEC,
}

PLACEHOLDER_MODULES: dict[WorkerRole, str] = {
    "scene": '''from __future__ import annotations


def create_scene(backend: str, *, sim_dt: float, sim_substeps: int, deformable_cfg: dict):
    raise NotImplementedError("scene worker did not replace this placeholder")
''',
    "body": '''from __future__ import annotations


def create_bodies(scene, task: str, *, deformable_cfg: dict):
    raise NotImplementedError("body worker did not replace this placeholder")
''',
    "action": '''from __future__ import annotations

from pathlib import Path


def run_actions(scene, actors, *, out_dir: Path, steps: int, render_state=None):
    raise NotImplementedError("action worker did not replace this placeholder")
''',
    "rendering": '''from __future__ import annotations

from pathlib import Path


def setup_rendering(
    scene,
    actors,
    *,
    out_dir: Path,
    steps: int,
    fps: int,
    duration_sec: float | None = None,
    target_video_frames: int | None = None,
    render_every_n_steps: int = 1,
    render_res: tuple[int, int] = (640, 480),
):
    raise NotImplementedError("rendering worker did not replace this placeholder")


def capture_frame(render_state: dict, step: int) -> None:
    raise NotImplementedError("rendering worker did not replace this placeholder")


def finalize_rendering(render_state: dict, *, event_log_path: Path | None = None, metrics_path: Path | None = None):
    raise NotImplementedError("rendering worker did not replace this placeholder")
''',
}


def dispatch_worker_roles(
    *,
    case_dir: Path,
    task: str,
    planner_output: dict[str, object],
    roles: tuple[WorkerRole, ...] | list[WorkerRole],
    repair_context: str | None = None,
) -> list[WorkerDispatchResult]:
    src_dir = case_dir / "src"
    logs_dir = case_dir / "logs"
    src_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    _seed_worker_files(case_dir)

    ordered_roles = tuple(roles)
    max_workers = resolve_writer_parallelism(len(ordered_roles))
    if max_workers == 1:
        return [
            _run_worker(
                case_dir=case_dir,
                task=task,
                planner_output=planner_output,
                spec=WORKERS[role],
                repair_context=repair_context,
            )
            for role in ordered_roles
        ]

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="writer") as executor:
        futures = [
            executor.submit(
                _run_worker,
                case_dir=case_dir,
                task=task,
                planner_output=planner_output,
                spec=WORKERS[role],
                repair_context=repair_context,
            )
            for role in ordered_roles
        ]
        return [future.result() for future in futures]


def repair_worker(*, case_dir: Path, task: str, owner: str, failure_context: str) -> WorkerDispatchResult | None:
    role = _normalize_owner(owner)
    if role is None:
        return None
    return _run_worker(
        case_dir=case_dir,
        task=task,
        planner_output=_load_planner_output(case_dir),
        spec=WORKERS[role],
        repair_context=failure_context,
    )


def write_worker_dispatch_report(case_dir: Path, results: list[WorkerDispatchResult]) -> None:
    active_parallelism = resolve_writer_parallelism(len(results))
    report = {
        "dispatch": {
            "num_workers": len(results),
            "configured_max_parallel_workers": CONFIGS.harness.max_parallel_workers,
            "max_parallel_workers": active_parallelism,
            "parallel": active_parallelism > 1,
        },
        "workers": [
            {
                "role": item.role,
                "ok": item.ok,
                "target_path": str(item.target_path),
                "returncode": item.codex_result.exit_code,
                "duration_sec": item.codex_result.duration_sec,
                "final_message_path": item.codex_result.final_message_path,
                "worker_status": item.worker_report.get("status") if item.worker_report else None,
                "exports": item.worker_report.get("exports") if item.worker_report else [],
                "error_message": item.error_message,
            }
            for item in results
        ]
    }
    path = case_dir / "reports" / "worker_dispatch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def resolve_writer_parallelism(num_roles: int) -> int:
    if num_roles <= 0:
        return 0
    configured = CONFIGS.harness.max_parallel_workers
    if configured is None or configured <= 0:
        return num_roles
    return max(1, min(num_roles, configured))


def _seed_worker_files(case_dir: Path) -> None:
    for role, spec in WORKERS.items():
        path = case_dir / spec.target_file
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(PLACEHOLDER_MODULES[role], encoding="utf-8")


def _run_worker(
    *,
    case_dir: Path,
    task: str,
    planner_output: dict[str, object],
    spec: WorkerSpec,
    repair_context: str | None,
) -> WorkerDispatchResult:
    target_path = case_dir / spec.target_file
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = case_dir / "logs"
    schema_path = Path("code_agent/specs/worker_report.schema.json")
    prompt = _worker_prompt(
        case_dir=case_dir,
        task=task,
        planner_output=planner_output,
        asset_manifest=_load_asset_manifest(case_dir),
        deformable_config=_load_deformable_config(case_dir),
        genesis_context=_load_genesis_context(case_dir),
        spec=spec,
        repair_context=repair_context,
    )
    invocation_role = spec.role if repair_context is None else f"{spec.role}_repair"
    result = run_codex_exec(
        CodexExecRequest(
            role=invocation_role,
            prompt=prompt,
            cwd=DEFAULT_REPO_ROOT,
            sandbox=CONFIGS.codex.worker_sandbox,
            model=CONFIGS.codex.worker_model,
            output_schema_path=schema_path,
            output_jsonl_path=logs_dir / f"codex_{invocation_role}.jsonl",
            final_message_path=logs_dir / f"codex_{invocation_role}.final.json",
            timeout_sec=CONFIGS.codex.worker_timeout_sec,
        )
    )
    worker_report, error_message = _parse_worker_report(Path(result.final_message_path))
    target_source = target_path.read_text(encoding="utf-8", errors="replace") if target_path.exists() else ""
    ok = (
        result.success
        and worker_report is not None
        and worker_report.get("status") == "completed"
        and _changed_files_include_target(worker_report, spec.target_file)
        and spec.required_export in target_source
        and target_path.exists()
        and target_path.stat().st_size > 0
        and "NotImplementedError" not in target_source
    )
    return WorkerDispatchResult(
        role=spec.role,
        ok=ok,
        target_path=target_path,
        codex_result=result,
        worker_report=worker_report,
        error_message=error_message,
    )


def _worker_prompt(
    *,
    case_dir: Path,
    task: str,
    planner_output: dict[str, object],
    asset_manifest: dict[str, object],
    deformable_config: dict[str, object],
    genesis_context: str,
    spec: WorkerSpec,
    repair_context: str | None,
) -> str:
    mode = "repair the existing module source" if repair_context else "author the module source"
    layout_context = _load_layout_context(case_dir)
    return textwrap.dedent(
        f"""
        {WORKER_COMMON_RULES}

        Task prompt:
        {task}

        {"User-provided layout context:" if layout_context else ""}
        {layout_context}

        Planner output:
        {json.dumps(planner_output, indent=2)}

        Asset manifest:
        {json.dumps(asset_manifest, indent=2)}

        Effective FEM/IPC capability/config:
        {json.dumps(deformable_config, indent=2)}

        FEM/IPC generation policy:
        - If deformable_config["enabled"] is false, do not create FEM materials, FEM entities, or deformable behavior.
          If the assigned task fundamentally requires soft-body deformation, fail clearly in your worker report instead
          of writing a rigid-body substitute.
        - If deformable_config["enabled"] is true and the planner asks for soft-body behavior, use FEM+IPC only. Do not
          use MPM, PBD, SPH, or other non-rigid solvers.
        - If deformable_config["ipc_enabled"] is true, IPC contact/coupling may be used. For rigid-only scenes, keep
          the bodies rigid/articulated and use IPC only for contact/coupling through `gs.options.IPCCouplerOptions` and
          `gs.materials.Rigid(...)` coupling fields.
        - If deformable_config["ipc_enabled"] is false, do not instantiate `gs.options.IPCCouplerOptions`.
        - Do not pass deformable_config["ipc_contact_d_hat_adaptive"] to `gs.options.IPCCouplerOptions`; it is a
          code-agent runtime switch. The generated entrypoint resolves it into `ipc_contact_d_hat` before scene setup.
        - All FEM `E`/`nu`/`rho` material-range defaults, IPC, tet, and precision defaults must come from
          deformable_config, not hardcoded local constants. FEM elastic bodies must still pass explicit
          task-appropriate `E`, `nu`, `rho`, and `friction_mu` values. `friction_mu` is chosen by the worker per FEM
          material; do not read `deformable_config["fem_friction_mu"]`.

        Genesis documentation and local-code context:
        {genesis_context}

        Context roots:
        - Repository root: {DEFAULT_REPO_ROOT}
        - Case workspace root: {case_dir}
        - Generated source directory: {case_dir / "src"}
        - Contracts directory: {case_dir / "contracts"}
        - Assets directory: {case_dir / "assets"}
        - Reports directory: {case_dir / "reports"}
        - Artifacts directory: {case_dir / "artifacts"}

        Role: {spec.role} worker
        Responsibility: {spec.responsibility}
        Mode: {mode}
        Exact target file: {case_dir / spec.target_file}
        Allowed write paths: {case_dir / spec.target_file}
        Required export: `{spec.required_export}`

        {RIGID_API_GUIDE}

        {FEM_IPC_API_GUIDE}

        Role-specific instructions:
        {textwrap.dedent(spec.prompt_body).strip()}

        {"Failure context to fix:" if repair_context else ""}
        {repair_context or ""}

        Final response must be JSON matching worker_report.schema.json.
        Set status to `completed` only after you have written the full replacement module to the exact target file.
        Set `commands_run` to an empty list if you did not run commands.
        Include changed_files with exactly `{spec.target_file}` when completed.
        """
    ).strip()


def _load_planner_output(case_dir: Path) -> dict[str, object]:
    path = case_dir / "contracts" / "planner_output.json"
    return load_json_object(path) or {}


def _load_asset_manifest(case_dir: Path) -> dict[str, object]:
    path = case_dir / "assets" / "asset_manifest.json"
    if not path.exists():
        return {"assets": [], "assumptions": [], "unresolved_risks": []}
    payload = load_json_object(path)
    if payload is None:
        return {"assets": [], "assumptions": [], "unresolved_risks": [f"Invalid asset manifest JSON: {path}"]}
    return payload


def _load_deformable_config(case_dir: Path) -> dict[str, object]:
    path = case_dir / "contracts" / "deformable_config.json"
    if not path.exists():
        return {"enabled": False, "ipc_enabled": False, "unresolved_risks": [f"Missing deformable config contract: {path}"]}
    payload = load_json_object(path)
    if payload is None:
        return {"enabled": False, "ipc_enabled": False, "unresolved_risks": [f"Invalid deformable config JSON: {path}"]}
    return payload


def _load_genesis_context(case_dir: Path) -> str:
    context_md = case_dir / "contracts" / "genesis_context.md"
    context_json = case_dir / "contracts" / "genesis_context.json"
    docs_dir = "<see context JSON>"
    catalog_path = "<see context JSON>"
    if context_json.exists():
        payload = load_json_object(context_json) or {}
        if isinstance(payload, dict):
            docs_dir = str(payload.get("docs_dir") or docs_dir)
            catalog_path = str(payload.get("catalog_path") or catalog_path)
    return "\n".join(
        [
            "Genesis official-doc and local-source context is available on disk for on-demand reading.",
            "Read only the relevant docs/source for your module; do not treat the full pack as preloaded.",
            f"- Context index: {context_md}",
            f"- Machine-readable context JSON: {context_json}",
            f"- Cached official docs directory: {docs_dir}",
            f"- Selected official-doc catalog: {catalog_path}",
            "- Active non-rigid scope: FEM+IPC only. IPC may also be used for rigid/articulated contact when enabled.",
            "- For rigid/mesh cases, use rigid/mesh/rendering docs as needed.",
            "- Prefer local Genesis source and examples over online docs if they disagree.",
        ]
    )


def _load_layout_context(case_dir: Path) -> str:
    path = case_dir / "inputs" / "layout_context.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _parse_worker_report(path: Path) -> tuple[dict[str, object] | None, str | None]:
    if not path.exists():
        return None, f"missing final message: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"invalid worker JSON: {exc}"
    if not isinstance(data, dict):
        return None, "worker final message is not a JSON object"
    return data, None


def _changed_files_include_target(worker_report: dict[str, object], target_file: str) -> bool:
    changed_files = worker_report.get("changed_files")
    if not isinstance(changed_files, list):
        return False
    target = Path(target_file).as_posix()
    for item in changed_files:
        if not isinstance(item, str):
            continue
        item_path = Path(item).as_posix()
        if item_path == target or item_path.endswith(f"/{target}"):
            return True
    return False


def _normalize_owner(owner: str) -> WorkerRole | None:
    lowered = owner.strip().lower()
    if lowered in WORKERS:
        return lowered  # type: ignore[return-value]
    if lowered == "render":
        return "rendering"
    return None
