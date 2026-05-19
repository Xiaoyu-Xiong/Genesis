from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from code_agent.prompts.common import PHYSICAL_CAUSALITY_CONTRACT, SOURCE_AWARE_REPAIR_GUIDE

if TYPE_CHECKING:
    from code_agent.opt.types import OptAgentRequest


def build_opt_prompt(request: OptAgentRequest) -> str:
    case_dir = request.case_dir.resolve()
    workspace = _workspace_summary(case_dir)
    request_text = json.dumps(_request_payload(request), indent=2)
    return f"""
You are the Opt Codex subagent for one generated Genesis simulation workspace.

Your job is to decide whether this generated case is parameter-limited and, if so, make the smallest safe set of
source/contract edits needed to expose optimizable parameters, run optimization, verify the result, and report evidence
back to Planner. You are not a fixed adapter and you must not rely on benchmark-specific string recipes.

Planner-to-Opt request:
{request_text}

Case workspace summary:
{json.dumps(workspace, indent=2)}

Core responsibilities:
- Inspect the generated case before editing. Read relevant `src/scene.py`, `src/body.py`, `src/action.py`,
  `src/rendering.py`, `src/main.py`, existing `contracts/*.json`, `reports/*.json`, and `artifacts/*/metrics.json`.
- Decide from the source and evidence whether optimization is appropriate. If the failure is structural, missing assets,
  missing control handles, impossible geometry, absent metrics, or invalid simulation code, return `needs_rewrite`
  instead of forcing a numeric optimization pass.
- If optimization is appropriate, identify a compact low-dimensional set of physically meaningful scalar parameters.
  You may expose action targets/timings/controller gains, body material/contact parameters, scene solver/contact
  parameters, or rendering variables only when they are relevant to the target. Do not optimize irrelevant constants.
- Patch generated modules so they read `contracts/current_opt_params.json` when present, fall back to
  `contracts/default_opt_params.json`, and remain runnable without optimization. Record loaded opt params and relevant
  measured quantities in `metrics.json`.
- Create or revise `contracts/target_spec.json`, `contracts/opt_space.json`, and `contracts/default_opt_params.json`
  using the schemas in `code_agent/specs/opt_schema/`.
- Use the existing runner whenever possible:
  `uv run python -m code_agent.cli run-opt --case-dir {case_dir} --backend {request.backend}`.
  Add `--max-trials`, `--timeout-sec`, `--steps`, `--duration-sec`, and `--render-fps` only when useful or requested.
- Render visual evidence when requested. If you need a separate baseline render, run the generated `src/main.py` from
  the case workspace with `uv run python src/main.py --backend {request.backend} --out-dir artifacts/opt_agent_baseline
  --render` plus requested timing flags.
- After optimization, inspect `reports/opt_report.json`, `reports/verification_report.json`, best params, metrics, and
  videos. Decide whether the outcome is `success`, `partial_success`, `needs_more_optimization`, `needs_rewrite`, or
  `failed`.

Hard safety constraints:
{PHYSICAL_CAUSALITY_CONTRACT}

Source-aware diagnosis guidance:
{SOURCE_AWARE_REPAIR_GUIDE}

Allowed edits from Planner:
{json.dumps(list(request.allowed_edits), indent=2)}

Forbidden changes from Planner:
{json.dumps(list(request.forbidden_changes), indent=2)}

Implementation rules:
- Use `uv run ...` for Python commands. Do not run bare `python`, `python -m`, or `pytest`.
- Keep edits inside the case workspace unless you are only reading repository documentation/schemas.
- Do not edit `code_agent/opt/` or repository source during this Opt pass. Your edits should prepare the generated case,
  not alter the optimizer implementation.
- Do not add hidden constraints, fake attachments, direct post-initialization object state writes, or task-object
  teleportation. Optimization must tune physical parameters or controls that the generated simulation uses honestly.
- Do not change the task semantics, required entities, or success target to make the case easier.
- Keep the search space small. Prefer 2-8 variables unless evidence justifies more.
- Prefer `scale: "log"` for positive variables spanning orders of magnitude, such as stiffness, damping, density,
  contact stiffness, tolerances, and controller gains.
- If you add metrics, make them observable quantities from the simulation state, not values copied from the target.
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
    return payload


def _workspace_summary(case_dir: Path) -> dict[str, object]:
    paths = {
        "src": _relative_files(case_dir, "src", 20),
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
