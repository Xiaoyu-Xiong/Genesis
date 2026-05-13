from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from code_agent.assets.mesh.workflow.steps import slugify_prompt
from code_agent.assets.xml.actuation import run_actuator_response_check
from code_agent.assets.xml.preview import render_xml_preview
from code_agent.assets.xml.validation import manifest_entry_from_xml_validation, validate_xml_asset
from code_agent.configs import CONFIGS
from code_agent.io_utils import dump_json
from code_agent.utils.codex import CodexExecRequest, run_codex_exec
from code_agent.utils.general_prompts import PHYSICAL_CAUSALITY_CONTRACT


def generate_xml_asset(
    *,
    task: str,
    output_dir: Path,
    logical_name: str | None = None,
    max_attempts: int = CONFIGS.xml_asset.max_generation_attempts,
) -> dict[str, Any]:
    """Run a standalone Codex XML worker and validate the generated MJCF asset."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logical_name = logical_name or slugify_prompt(task) or "xml_asset"
    xml_path = output_dir / f"{slugify_prompt(logical_name)}.xml"
    attempts: list[dict[str, Any]] = []
    previous_context: dict[str, Any] | None = None
    final_report: dict[str, Any] | None = None

    for attempt_index in range(1, max(1, max_attempts) + 1):
        prompt = _xml_worker_prompt(
            task=task,
            output_dir=output_dir,
            xml_path=xml_path,
            logical_name=logical_name,
            attempt_index=attempt_index,
            previous_context=previous_context,
        )
        result = run_codex_exec(
            CodexExecRequest(
                role=f"xml_asset_worker_attempt_{attempt_index}",
                prompt=prompt,
                cwd=Path.cwd(),
                sandbox=CONFIGS.codex.worker_sandbox,
                model=CONFIGS.codex.worker_model,
                output_schema_path=Path("code_agent/specs/xml_worker_report.schema.json"),
                output_jsonl_path=logs_dir / f"codex_xml_attempt_{attempt_index}.jsonl",
                final_message_path=logs_dir / f"codex_xml_attempt_{attempt_index}.final.json",
                timeout_sec=CONFIGS.codex.worker_timeout_sec,
            )
        )
        worker_report, worker_error = _load_worker_report(Path(result.final_message_path))
        resolved_xml_path = _resolve_generated_xml_path(worker_report, output_dir, preferred_path=xml_path)
        validation_report = validate_xml_asset(resolved_xml_path) if resolved_xml_path else _missing_xml_report(xml_path)
        preview_report = (
            render_xml_preview(resolved_xml_path, output_dir / "previews" / f"attempt_{attempt_index:02d}")
            if validation_report.get("mujoco_ok") and resolved_xml_path is not None
            else _skipped_preview_report(resolved_xml_path, "MuJoCo validation did not pass.")
        )
        actuator_response_report = (
            run_actuator_response_check(resolved_xml_path)
            if validation_report.get("mujoco_ok") and resolved_xml_path is not None
            else _skipped_actuator_response_report(resolved_xml_path, "MuJoCo validation did not pass.")
        )
        manifest_entry = manifest_entry_from_xml_validation(
            logical_name=logical_name,
            xml_path=resolved_xml_path or xml_path,
            validation_report=validation_report,
        )
        ok = (
            result.success
            and worker_report is not None
            and worker_report.get("status") == "completed"
            and resolved_xml_path is not None
            and validation_report.get("ok")
            and preview_report.get("ok")
            and actuator_response_report.get("ok")
        )
        attempt_report = {
            "attempt": attempt_index,
            "ok": ok,
            "codex": result.to_dict(),
            "worker_report": worker_report,
            "worker_error": worker_error,
            "xml_path": None if resolved_xml_path is None else str(resolved_xml_path),
            "validation_report": validation_report,
            "preview_report": preview_report,
            "actuator_response_report": actuator_response_report,
            "manifest_entry": manifest_entry,
        }
        attempts.append(attempt_report)
        dump_json(attempt_report, reports_dir / f"attempt_{attempt_index:02d}.json")

        if ok:
            final_report = _final_report(
                ok=True,
                status="xml_asset_generated",
                task=task,
                logical_name=logical_name,
                output_dir=output_dir,
                attempts=attempts,
                current_attempt=attempt_report,
            )
            break

        previous_context = _repair_context(
            worker_report=worker_report,
            worker_error=worker_error,
            validation_report=validation_report,
            preview_report=preview_report,
            actuator_response_report=actuator_response_report,
            xml_path=resolved_xml_path,
        )

    if final_report is None:
        final_report = _final_report(
            ok=False,
            status="xml_asset_generation_failed",
            task=task,
            logical_name=logical_name,
            output_dir=output_dir,
            attempts=attempts,
            current_attempt=attempts[-1] if attempts else None,
        )

    dump_json(final_report, reports_dir / "xml_asset_generation_report.json")
    if final_report.get("manifest_entry") is not None:
        dump_json(
            {
                "assets": [final_report["manifest_entry"]],
                "assumptions": [
                    "Standalone XML asset manifest for this output directory; episode runners may merge it into the "
                    "canonical Planner asset manifest."
                ],
                "unresolved_risks": [] if final_report.get("ok") else [logical_name],
            },
            output_dir / "asset_manifest.json",
        )
    return final_report


def _xml_worker_prompt(
    *,
    task: str,
    output_dir: Path,
    xml_path: Path,
    logical_name: str,
    attempt_index: int,
    previous_context: dict[str, Any] | None,
) -> str:
    repair_context = ""
    if previous_context is not None:
        repair_context = (
            "\nPrevious attempt failed. Repair the XML instead of hand-waving around the failure.\n"
            f"{json.dumps(previous_context, indent=2, ensure_ascii=False)}\n"
        )
    return textwrap.dedent(
        f"""
        You are the standalone XML/MJCF asset worker for the Genesis code-agent asset system.

        User asset request:
        {task}

        Write exactly one MJCF XML file for logical asset `{logical_name}` at:
        {xml_path}

        Output directory:
        {output_dir}

        Attempt number:
        {attempt_index}

        Scope rules:
        - Generate the XML from scratch; do not modify repository source code or main pipeline files.
        - The XML must contain exactly one <mujoco> root and exactly one articulated body tree directly under
          <worldbody>.
        - Include joints, inertials/geoms/sites/tendons/equality constraints only when they belong to this one
          articulated asset.
        - Include a complete <actuator> section for the articulated body. Actuators must be named, target valid
          joints/tendons/sites/bodies, and expose useful ctrlrange values when applicable.
        - Do not include scene-level ground, free props, projectiles, bins, ramps, arenas, task obstacles, cameras,
          lights, or global simulation staging.
        - Do not include <option> or other global simulation settings such as gravity, timestep, solver, integrator, or
          global contact options. Those belong to the Scene/Simulation harness, not the asset XML.
        - Do not use mesh or hfield assets. Use primitive MJCF geoms only.
        - Prefer named joints, named actuators, explicit joint ranges, collision-enabled primitive geoms, and sensible
          masses/inertias.
        - If the asset may be used as a fixed-base IPC external articulation, every parent and child link participating
          in a driven joint must have collision geometry. Do not leave a logical mount body empty. Add a tiny
          nonzero-volume primitive dummy collision geom to a mount parent when needed; it must not have both `contype`
          and `conaffinity` set to zero, and it should be placed away from the active contact region.
        - The result should satisfy the text prompt and also look physically coherent and visually understandable.
        - If the request implies grasping, gates, locks, buttons, hinges, sliders, latches, or tools, make the actuator
          interface explicit enough for an Action Worker to command it without reverse-engineering the XML.
        {PHYSICAL_CAUSALITY_CONTRACT}

        Final response requirements:
        - Return JSON matching code_agent/specs/xml_worker_report.schema.json.
        - `xml_path` must be the path to the generated file.
        - `changed_files` must include the XML path.
        - `actuator_contract.actuators` must describe every actuator with name, type/control mode, target, command
          range, neutral command, and suggested schedule semantics.
        {repair_context}
        """
    ).strip()


def _load_worker_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"Codex final message was not written: {path}"
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"Codex final message is not JSON: {exc}: {raw}"
    if not isinstance(data, dict):
        return None, "Codex final message must be a JSON object."
    return data, None


def _resolve_generated_xml_path(
    worker_report: dict[str, Any] | None,
    output_dir: Path,
    *,
    preferred_path: Path,
) -> Path | None:
    candidates: list[Path] = []
    if worker_report is not None and isinstance(worker_report.get("xml_path"), str):
        candidates.append(Path(worker_report["xml_path"]))
    candidates.append(preferred_path)
    candidates.extend(sorted(output_dir.glob("*.xml")))

    output_root = output_dir.resolve()
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        path = path.resolve()
        if not path.exists() or path.suffix.lower() != ".xml":
            continue
        try:
            path.relative_to(output_root)
        except ValueError:
            continue
        return path
    return None


def _missing_xml_report(expected_path: Path) -> dict[str, Any]:
    return {
        "ok": False,
        "xml_path": str(expected_path.resolve()),
        "parser_ok": False,
        "mujoco_ok": False,
        "model_summary": {},
        "errors": [f"No generated XML file was found at or under {expected_path.parent.resolve()}."],
        "warnings": [],
        "joints": [],
        "actuators": [],
        "sites": [],
        "tendons": [],
        "equalities": [],
        "bodies": [],
        "geoms": [],
        "base": {},
        "control_interface": {},
    }


def _skipped_preview_report(xml_path: Path | None, reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "xml_path": None if xml_path is None else str(xml_path),
        "views": [],
        "errors": [reason],
        "warnings": [],
    }


def _skipped_actuator_response_report(xml_path: Path | None, reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "xml_path": None if xml_path is None else str(xml_path),
        "errors": [reason],
        "warnings": [],
        "actuators": [],
    }


def _repair_context(
    *,
    worker_report: dict[str, Any] | None,
    worker_error: str | None,
    validation_report: dict[str, Any],
    preview_report: dict[str, Any],
    actuator_response_report: dict[str, Any],
    xml_path: Path | None,
) -> dict[str, Any]:
    context = {
        "worker_report": worker_report,
        "worker_error": worker_error,
        "validation_errors": validation_report.get("errors", []),
        "validation_warnings": validation_report.get("warnings", []),
        "preview_errors": preview_report.get("errors", []),
        "preview_warnings": preview_report.get("warnings", []),
        "actuator_response_errors": actuator_response_report.get("errors", []),
        "actuator_response_warnings": actuator_response_report.get("warnings", []),
    }
    if xml_path is not None and xml_path.exists():
        context["previous_xml_path"] = str(xml_path)
        context["previous_xml_content"] = xml_path.read_text(encoding="utf-8", errors="replace")
    return context


def _final_report(
    *,
    ok: bool,
    status: str,
    task: str,
    logical_name: str,
    output_dir: Path,
    attempts: list[dict[str, Any]],
    current_attempt: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest_entry = current_attempt.get("manifest_entry") if current_attempt else None
    return {
        "ok": ok,
        "status": status,
        "task": task,
        "logical_name": logical_name,
        "output_dir": str(output_dir),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "xml_path": current_attempt.get("xml_path") if current_attempt else None,
        "validation_report": current_attempt.get("validation_report") if current_attempt else None,
        "preview_report": current_attempt.get("preview_report") if current_attempt else None,
        "actuator_response_report": current_attempt.get("actuator_response_report") if current_attempt else None,
        "manifest_entry": manifest_entry,
    }
