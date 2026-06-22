from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code_agent.opt.types import OptAgentRequest


def build_opt_prompt(request: OptAgentRequest) -> str:
    case_dir = request.case_dir.resolve()
    workspace = _workspace_summary(case_dir)
    request_text = json.dumps(_request_payload(request), indent=2)
    return f"""
You are the Opt Codex subagent for one generated Genesis simulation workspace.

Your job is to decide whether this generated case is limited by bounded continuous parameters and, if so, make the
smallest safe set of source/contract edits needed to expose optimizable parameters, run optimization, verify the
result, and report evidence back to Planner. You are not a fixed adapter and you must not rely on benchmark-specific
string recipes.

Planner-to-Opt request:
{request_text}

Case workspace summary:
{json.dumps(workspace, indent=2)}

Planner-dispatched SimDebug cards for Opt:
{request.simdebug_card_context or "No SimDebug cards were dispatched for this Opt pass."}

Core protocol:
- Inspect the generated case before editing. Read relevant `src/scene.py`, `src/body.py`, `src/action.py`,
  `src/main.py`, existing `contracts/*.json`, `reports/*.json`, and `artifacts/*/metrics.json`. You may read
  `src/rendering.py` only to understand evidence wiring, but it is outside the optimization/edit surface.
- Use the Planner-dispatched SimDebug cards as the source of Opt routing, parameter-selection, metric/objective design,
  physical-safety, visual-evidence, XML-scalar-patch, and task-semantics restrictions.
- Patch generated modules so they read `contracts/current_opt_params.json` when present, fall back to
  `contracts/default_opt_params.json`, and remain runnable without optimization.
- Create or revise `contracts/target_spec.json`, `contracts/opt_space.json`, and `contracts/default_opt_params.json`
  using the schemas in `code_agent/specs/opt_schema/`.
- Use the existing runner whenever possible:
  `uv run --no-sync python -m code_agent.cli run-opt --case-dir {case_dir} --backend {request.backend}`.
  Add `--max-trials`, `--timeout-sec`, `--steps`, `--duration-sec`, and `--render-fps` only when useful or requested.
  The runner reads `contracts/timing.json` for sim_dt, sim_substeps, render cadence, and render resolution.
- If you need a separate generated-code run or baseline render, use the local execution wrapper so Genesis/IPC gets the
  repository CUDA environment and cross-process execution lock:
  `uv run --no-sync python -m code_agent.utils.local_execution {case_dir} --main-file src/main.py --output-dir
  {case_dir}/reports/opt_agent_baseline --backend {request.backend} -- --backend {request.backend} --out-dir
  artifacts/opt_agent_baseline --render` plus the timing flags from the Planner-to-Opt request or
  `contracts/timing.json`.
- After optimization, inspect `reports/opt_report.json`, `reports/verification_report.json`, best params, metrics, and
  videos before returning success-like statuses.

Allowed edits from Planner:
{json.dumps(list(request.allowed_edits), indent=2)}

Forbidden changes from Planner:
{json.dumps(list(request.forbidden_changes), indent=2)}

Implementation rules:
- Use `uv run --no-sync ...` for Python commands. Do not run bare `python`, `python -m`, or `pytest`.
- Keep edits inside the case workspace unless you are only reading repository documentation/schemas.
- Do not edit `code_agent/opt/` or repository source during this Opt pass. Your edits should prepare the generated case,
  not alter the optimizer implementation.
- Follow the Planner-dispatched SimDebug cards when deciding whether a result is useful, needs more optimization,
  needs rewrite, or failed.
- If you cannot safely expose variables, return `needs_rewrite` with concrete owner guidance.

Required final response:
Return exactly one JSON object matching `code_agent/specs/opt_schema/opt_subagent_report.schema.json`.
Also write the same object, wrapped with `schema_version`, `request`, `result`, and any useful `codex_notes`, to
`reports/opt_subagent_report.json` inside the case workspace.

The final JSON fields are:
- `status`: one of `success`, `partial_success`, `needs_more_optimization`, `needs_rewrite`, `failed`
- `case_type`: concise task family string or null
- `edited_files`: relative paths you changed
- `optimized_variables`: variable names exposed in `contracts/opt_space.json`
- `baseline`: object with `success`, `score`, `metrics_path`, `video_path`, `params_path`, and `summary`; use null
  values for unavailable fields
- `best`: object with `success`, `score`, `metrics_path`, `video_path`, `params_path`, and `summary`; use null
  values for unavailable fields
- `diagnosis`: what happened and why
- `recommendation`: what Planner should do next
- `evidence`: concise path/value evidence
- `opt_report_path`: usually `reports/opt_report.json`, or null if optimization did not run
- `failures`: concrete blockers or errors
""".strip()


def _request_payload(request: OptAgentRequest) -> dict[str, object]:
    payload = asdict(request)
    payload["case_dir"] = str(request.case_dir.resolve())
    payload.pop("simdebug_card_context", None)
    return payload


def _workspace_summary(case_dir: Path) -> dict[str, object]:
    paths = {
        "src": _relative_files(case_dir, "src", 20),
        "assets_xml": _relative_files(case_dir, "assets/xml", 40),
        "contracts": _relative_files(case_dir, "contracts", 30),
        "reports": _relative_files(case_dir, "reports", 30),
        "artifacts": _relative_files(case_dir, "artifacts", 40),
    }
    key_json = {}
    for rel_path in (
        "contracts/planner_output.json",
        "contracts/timing.json",
        "contracts/deformable_config.json",
        "reports/opt_report.json",
        "reports/verification_report.json",
        "artifacts/metrics.json",
        "artifacts/render_stats.json",
    ):
        path = case_dir / rel_path
        if path.is_file():
            key_json[rel_path] = _short_text(path)
    return {
        "case_dir": str(case_dir),
        "files": paths,
        "key_json_previews": key_json,
    }


def _relative_files(case_dir: Path, relative_dir: str, limit: int) -> list[str]:
    root = case_dir / relative_dir
    if not root.is_dir():
        return []
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return [str(path.relative_to(case_dir)) for path in files[:limit]]


def _short_text(path: Path, max_chars: int = 4000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... <truncated>"
