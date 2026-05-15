from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import time
from pathlib import Path
from typing import Any

from code_agent.assets.inspection import inspect_generated_assets
from code_agent.assets.mesh.episode import generate_mesh_assets_for_episode, update_mesh_asset_metadata_for_episode
from code_agent.assets.xml.episode import generate_xml_assets_for_episode
from code_agent.io_utils import dump_json


class AssetActionHandler:
    """Planner action handlers for background asset generation jobs."""

    def __init__(self, session: Any):
        self.session = session
        self._asset_executors: dict[str, ThreadPoolExecutor] = {}
        self._asset_futures: dict[str, Future[dict[str, Any]]] = {}

    def start_mesh_assets(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._start_asset_job(
            action,
            kind="mesh",
            generator=generate_mesh_assets_for_episode,
            ready_status="mesh_assets_ready",
            running_status="mesh_assets_running",
            started_status="mesh_assets_started",
            report_path=self.session.case_dir / "reports" / "asset_generation_report.json",
            short_circuit_ready=False,
        )

    def wait_mesh_assets(self, action: dict[str, Any]) -> dict[str, Any]:
        _ = action
        if self._asset_job_ready("mesh"):
            return {
                "ok": True,
                "status": "mesh_assets_ready",
                "message": "Mesh asset manifest is ready.",
                "asset_manifest_path": self._asset_manifest_path_text(),
            }
        result = self.poll_asset_job(kind="mesh", wait=True)
        if result is not None:
            return result
        return {
            "ok": False,
            "status": "precondition_failed",
            "message": "No mesh asset generation job is running.",
        }

    def update_mesh_asset_metadata(self, action: dict[str, Any]) -> dict[str, Any]:
        future = self._asset_futures.get("mesh")
        if future is not None and not future.done():
            return {
                "ok": False,
                "status": "precondition_failed",
                "message": "Cannot update mesh metadata while mesh asset generation is running.",
                "asset_generation_report_path": self._asset_job_report_path_text("mesh"),
                "background": True,
            }
        planner_output, planner_update = self._planner_output_for_asset_action(action)
        if planner_output is None:
            return planner_update
        planner_output_updated = bool(planner_update.get("updated"))
        planner_output_path = planner_update.get("planner_output_path")
        asset_names = self._asset_names_from_action(action)
        result = update_mesh_asset_metadata_for_episode(
            case_dir=self.session.case_dir,
            task=self.session.config.task,
            planner_output=planner_output,
            asset_names=asset_names,
        )
        assets = self._ensure_assets_state()
        jobs = assets.setdefault("jobs", {})
        jobs["mesh"] = {
            "status": "metadata_update",
            "ok": False,
            "kind": "mesh",
            "asset_manifest_path": result.get("asset_manifest_path"),
            "asset_generation_report_path": result.get("asset_generation_report_path"),
            "selected_asset_names": result.get("selected_asset_names", []),
            "skipped_asset_names": result.get("skipped_asset_names", []),
            "num_assets": result.get("num_assets", 0),
            "schema_errors": [],
            "failure_classes": result.get("failure_classes", []),
            "message": result.get("message"),
            "planner_output_updated": planner_output_updated,
            "planner_output_path": planner_output_path,
            "updated_at_unix": time.time(),
            "background": False,
        }
        payload = self._finalize_asset_result("mesh", result)
        payload["metadata_updated_asset_names"] = result.get("metadata_updated_asset_names", [])
        return payload

    def start_xml_assets(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._start_asset_job(
            action,
            kind="xml",
            generator=generate_xml_assets_for_episode,
            ready_status="xml_assets_ready",
            running_status="xml_assets_running",
            started_status="xml_assets_started",
            report_path=self.session.case_dir / "reports" / "xml_asset_generation_report.json",
            short_circuit_ready=True,
        )

    def wait_xml_assets(self, action: dict[str, Any]) -> dict[str, Any]:
        _ = action
        if self._asset_job_ready("xml"):
            return {
                "ok": True,
                "status": "xml_assets_ready",
                "message": "XML asset manifest is ready.",
                "asset_manifest_path": self._asset_manifest_path_text(),
            }
        result = self.poll_asset_job(kind="xml", wait=True)
        if result is not None:
            return result
        return {
            "ok": False,
            "status": "precondition_failed",
            "message": "No XML asset generation job is running.",
        }

    def inspect_assets(self, action: dict[str, Any]) -> dict[str, Any]:
        asset_names = self._asset_names_from_action(action)
        report = inspect_generated_assets(self.session.case_dir, asset_names=asset_names)
        assets_summary = [
            {
                "logical_name": asset.get("logical_name"),
                "source_type": asset.get("source_type"),
                "runtime_path": asset.get("runtime_path"),
                "preview_paths": asset.get("preview_paths", []),
                "geometry": asset.get("geometry"),
                "errors": asset.get("errors", []),
                "warnings": asset.get("warnings", []),
            }
            for asset in report.get("assets", [])
            if isinstance(asset, dict)
        ]
        self.session.state["asset_inspection"] = {
            "status": report.get("status"),
            "ok": report.get("ok"),
            "report_path": report.get("report_path"),
            "output_dir": report.get("output_dir"),
            "selected_asset_names": report.get("selected_asset_names", []),
            "asset_error_count": report.get("asset_error_count", 0),
            "asset_warning_count": report.get("asset_warning_count", 0),
            "assets": assets_summary,
        }
        return {
            "ok": report.get("status") != "precondition_failed",
            "status": report.get("status", "asset_inspection_complete"),
            "message": (
                f"Inspected {len(assets_summary)} asset(s); "
                f"asset_errors={report.get('asset_error_count', 0)}, "
                f"asset_warnings={report.get('asset_warning_count', 0)}."
            ),
            "asset_inspection_report_path": report.get("report_path"),
            "preview_dir": report.get("output_dir"),
            "selected_asset_names": report.get("selected_asset_names", []),
            "asset_error_count": report.get("asset_error_count", 0),
            "asset_warning_count": report.get("asset_warning_count", 0),
            "assets": assets_summary,
            "errors": report.get("errors", []),
            "warnings": report.get("warnings", []),
        }

    def adopt_layout_asset_manifest(self) -> dict[str, Any] | None:
        """Register layout-declared reusable assets that were prepared before planning begins."""

        manifest_path = self.session.case_dir / "assets" / "layout_asset_manifest.json"
        if not manifest_path.exists():
            return None
        report_path = self.session.case_dir / "reports" / "layout_asset_report.json"
        manifest = self.session.load_json(manifest_path)
        if manifest is None:
            schema_errors = [f"layout asset manifest was not readable: {manifest_path}"]
            ok = False
            num_assets = 0
            unresolved = []
        else:
            schema_errors = self.session.validate_json_schema(
                manifest,
                Path("code_agent/specs/asset_manifest.schema.json"),
            )
            raw_assets = manifest.get("assets")
            entries = raw_assets if isinstance(raw_assets, list) else []
            unresolved = [
                str(entry.get("logical_name"))
                for entry in entries
                if isinstance(entry, dict) and entry.get("status") != "ready" and entry.get("logical_name")
            ]
            num_assets = len(entries)
            ok = not schema_errors and not unresolved
        assets = self._ensure_assets_state()
        jobs = assets.setdefault("jobs", {})
        jobs["layout"] = {
            "status": "ready" if ok else "failed",
            "ok": ok,
            "kind": "layout",
            "asset_manifest_path": str(manifest_path),
            "asset_generation_report_path": str(report_path) if report_path.exists() else None,
            "message": None if ok else "One or more reusable layout assets failed validation.",
            "failure_classes": [] if ok else ["layout_asset.validation_failed"],
            "selected_asset_names": [],
            "skipped_asset_names": [],
            "num_assets": num_assets,
            "schema_errors": schema_errors,
            "unresolved_assets": unresolved,
            "updated_at_unix": time.time(),
            "background": False,
        }
        combined_path, combined_errors = self._write_combined_asset_manifest()
        self._refresh_asset_state(combined_path=combined_path, combined_schema_errors=combined_errors)
        return {
            "ok": ok,
            "status": "layout_assets_ready" if ok else "layout_asset_validation_failed",
            "asset_manifest_path": str(combined_path),
            "partial_asset_manifest_path": str(manifest_path),
            "asset_generation_report_path": str(report_path) if report_path.exists() else None,
            "num_assets": num_assets,
            "schema_errors": schema_errors,
            "combined_schema_errors": combined_errors,
            "unresolved_assets": unresolved,
        }

    def poll_asset_jobs(self, *, wait: bool) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for kind in list(self._asset_futures):
            result = self.poll_asset_job(kind=kind, wait=wait)
            if result is not None:
                results.append(result)
        return results

    def poll_asset_job(self, *, kind: str, wait: bool) -> dict[str, Any] | None:
        future = self._asset_futures.get(kind)
        if future is None:
            return None
        if not wait and not future.done():
            return None
        try:
            result = future.result() if wait else future.result(timeout=0)
        except Exception as exc:
            self._asset_futures.pop(kind, None)
            payload = {
                "ok": False,
                "status": f"{kind}_asset_generation_failed",
                "asset_manifest_path": str(self._default_partial_manifest_path(kind)),
                "asset_generation_report_path": self._asset_job_report_path_text(kind),
                "selected_asset_names": [],
                "skipped_asset_names": [],
                "num_assets": 0,
                "schema_errors": [f"{type(exc).__name__}: {exc}"],
            }
            assets = self._ensure_assets_state()
            jobs = assets.setdefault("jobs", {})
            jobs[kind] = {
                "status": "failed",
                "ok": False,
                "kind": kind,
                "asset_manifest_path": payload["asset_manifest_path"],
                "asset_generation_report_path": payload["asset_generation_report_path"],
                "selected_asset_names": [],
                "skipped_asset_names": [],
                "num_assets": 0,
                "schema_errors": payload["schema_errors"],
                "updated_at_unix": time.time(),
                "background": False,
            }
            combined_path, combined_errors = self._write_combined_asset_manifest()
            self._refresh_asset_state(combined_path=combined_path, combined_schema_errors=combined_errors)
        else:
            self._asset_futures.pop(kind, None)
            payload = self._finalize_asset_result(kind, result)
        self._shutdown_asset_executor(kind)
        return payload

    def _start_asset_job(
        self,
        action: dict[str, Any],
        *,
        kind: str,
        generator: Any,
        ready_status: str,
        running_status: str,
        started_status: str,
        report_path: Path,
        short_circuit_ready: bool,
    ) -> dict[str, Any]:
        future = self._asset_futures.get(kind)
        if future is not None and not future.done():
            return {
                "ok": True,
                "status": running_status,
                "message": f"{kind} asset generation is already running in the background.",
                "asset_generation_report_path": self._asset_job_report_path_text(kind),
                "background": True,
            }
        planner_output, planner_update = self._planner_output_for_asset_action(action)
        if planner_output is None:
            return planner_update
        planner_output_updated = bool(planner_update.get("updated"))
        planner_output_path = planner_update.get("planner_output_path")
        asset_names = self._asset_names_from_action(action)
        if short_circuit_ready and self._asset_job_ready(kind) and not asset_names and not planner_output_updated:
            return {
                "ok": True,
                "status": ready_status,
                "message": f"{kind} asset manifest is already ready.",
                "asset_manifest_path": self._asset_manifest_path_text(),
                "asset_generation_report_path": self._asset_job_report_path_text(kind),
            }
        self._shutdown_asset_executor(kind)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{kind}_asset_job")
        self._asset_executors[kind] = executor
        self._asset_futures[kind] = executor.submit(
            generator,
            case_dir=self.session.case_dir,
            task=self.session.config.task,
            planner_output=planner_output,
            asset_names=asset_names,
        )
        manifest_path = self.session.case_dir / "assets" / "asset_manifest.json"
        assets = self._ensure_assets_state()
        jobs = assets.setdefault("jobs", {})
        jobs[kind] = {
            "status": "running",
            "ok": False,
            "kind": kind,
            "asset_manifest_path": None,
            "asset_generation_report_path": str(report_path),
            "selected_asset_names": asset_names or [],
            "skipped_asset_names": [],
            "num_assets": 0,
            "schema_errors": [],
            "failure_classes": [],
            "message": None,
            "planner_output_updated": planner_output_updated,
            "planner_output_path": planner_output_path,
            "started_at_unix": time.time(),
            "background": True,
        }
        assets.update(
            {
                "status": "running",
                "ok": False,
                "asset_manifest_path": str(manifest_path),
                "asset_generation_report_path": str(report_path),
                "selected_asset_names": self._collect_asset_names("selected_asset_names"),
                "skipped_asset_names": self._collect_asset_names("skipped_asset_names"),
                "num_assets": self._sum_asset_counts(),
                "schema_errors": [],
                "failure_classes": [],
                "message": None,
                "background": True,
                "updated_at_unix": time.time(),
            }
        )
        return {
            "ok": True,
            "status": started_status,
            "message": f"{kind} asset generation started in the background.",
            "asset_manifest_path": str(manifest_path),
            "asset_generation_report_path": str(report_path),
            "selected_asset_names": asset_names or [],
            "planner_output_updated": planner_output_updated,
            "planner_output_path": planner_output_path,
            "background": True,
        }

    def _planner_output_for_asset_action(self, action: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        action_planner_output = action.get("planner_output")
        if action_planner_output is not None:
            if not isinstance(action_planner_output, dict):
                return None, {
                    "ok": False,
                    "status": "invalid_action",
                    "message": "Asset actions require planner_output to be an object or null.",
                }
            accepted = self.session.accept_planner_output(
                action_planner_output,
                rationale=str(action.get("rationale") or ""),
            )
            if not accepted.get("ok"):
                return None, accepted
            return action_planner_output, {
                "ok": True,
                "updated": True,
                "planner_output_path": accepted.get("planner_output_path"),
                "timing": accepted.get("timing"),
            }

        planner_output = self.session.current_planner_output()
        if planner_output is None:
            return None, {"ok": False, "status": "precondition_failed", "message": "planner_output is missing."}
        return planner_output, {
            "ok": True,
            "updated": False,
            "planner_output_path": self.session.state.get("planner_output_path"),
        }

    def _finalize_asset_result(self, kind: str, result: dict[str, Any]) -> dict[str, Any]:
        manifest_path = Path(str(result.get("asset_manifest_path") or self._default_partial_manifest_path(kind)))
        manifest = self.session.load_json(manifest_path)
        schema_errors = []
        if manifest is None:
            schema_errors = [f"asset manifest was not written: {manifest_path}"]
        else:
            schema_errors = self.session.validate_json_schema(
                manifest,
                Path("code_agent/specs/asset_manifest.schema.json"),
            )
        ok = bool(result.get("ok")) and not schema_errors
        assets = self._ensure_assets_state()
        jobs = assets.setdefault("jobs", {})
        previous_job = jobs.get(kind) if isinstance(jobs.get(kind), dict) else {}
        jobs[kind] = {
            "status": "ready" if ok else "failed",
            "ok": ok,
            "kind": kind,
            "asset_manifest_path": str(manifest_path),
            "asset_generation_report_path": result.get("asset_generation_report_path"),
            "message": result.get("message"),
            "recommended_owner": result.get("recommended_owner"),
            "repair_summary": result.get("repair_summary"),
            "failure_classes": result.get("failure_classes", []),
            "selected_asset_names": result.get("selected_asset_names", []),
            "skipped_asset_names": result.get("skipped_asset_names", []),
            "num_assets": result.get("num_assets", 0),
            "schema_errors": schema_errors,
            "planner_output_updated": previous_job.get("planner_output_updated", False),
            "planner_output_path": previous_job.get("planner_output_path"),
            "updated_at_unix": time.time(),
            "background": False,
        }
        combined_path, combined_errors = self._write_combined_asset_manifest()
        self._refresh_asset_state(combined_path=combined_path, combined_schema_errors=combined_errors)
        return {
            "ok": ok,
            "status": result.get("status", f"{kind}_assets_generated"),
            "asset_manifest_path": str(combined_path),
            "partial_asset_manifest_path": str(manifest_path),
            "asset_generation_report_path": result.get("asset_generation_report_path"),
            "message": result.get("message"),
            "recommended_owner": result.get("recommended_owner"),
            "repair_summary": result.get("repair_summary"),
            "failure_classes": result.get("failure_classes", []),
            "num_assets": result.get("num_assets", 0),
            "selected_asset_names": result.get("selected_asset_names", []),
            "skipped_asset_names": result.get("skipped_asset_names", []),
            "schema_errors": schema_errors,
            "combined_schema_errors": combined_errors,
            "planner_output_updated": previous_job.get("planner_output_updated", False),
            "planner_output_path": previous_job.get("planner_output_path"),
        }

    def _ensure_assets_state(self) -> dict[str, Any]:
        assets = self.session.state.get("assets")
        if not isinstance(assets, dict):
            assets = {}
            self.session.state["assets"] = assets
        assets.setdefault("jobs", {})
        assets.setdefault("asset_manifest_path", str(self.session.case_dir / "assets" / "asset_manifest.json"))
        return assets

    def _asset_job_ready(self, kind: str) -> bool:
        assets = self._ensure_assets_state()
        jobs = assets.get("jobs")
        if not isinstance(jobs, dict):
            return False
        job = jobs.get(kind)
        if not isinstance(job, dict) or not job.get("ok"):
            return False
        manifest_path = job.get("asset_manifest_path")
        return isinstance(manifest_path, str) and self.session.load_json(Path(manifest_path)) is not None

    def _asset_job_report_path_text(self, kind: str) -> str | None:
        assets = self._ensure_assets_state()
        jobs = assets.get("jobs")
        if isinstance(jobs, dict):
            job = jobs.get(kind)
            if isinstance(job, dict) and isinstance(job.get("asset_generation_report_path"), str):
                return job["asset_generation_report_path"]
        if kind == "mesh":
            return str(self.session.case_dir / "reports" / "asset_generation_report.json")
        if kind == "xml":
            return str(self.session.case_dir / "reports" / "xml_asset_generation_report.json")
        return None

    def _asset_manifest_path_text(self) -> str | None:
        assets = self._ensure_assets_state()
        manifest_path = assets.get("asset_manifest_path")
        return manifest_path if isinstance(manifest_path, str) else None

    def _shutdown_asset_executor(self, kind: str) -> None:
        executor = self._asset_executors.pop(kind, None)
        if executor is not None:
            executor.shutdown(wait=False)

    def _default_partial_manifest_path(self, kind: str) -> Path:
        if kind == "xml":
            return self.session.case_dir / "assets" / "xml_asset_manifest.json"
        return self.session.case_dir / "assets" / "asset_manifest.json"

    def _write_combined_asset_manifest(self) -> tuple[Path, list[str]]:
        assets = self._ensure_assets_state()
        jobs = assets.get("jobs")
        canonical_path = self.session.case_dir / "assets" / "asset_manifest.json"
        entries: list[dict[str, Any]] = []
        assumptions: list[str] = []
        unresolved_risks: list[str] = []
        seen_entries: set[tuple[str, str, str]] = set()
        if isinstance(jobs, dict):
            for kind, job in jobs.items():
                if not isinstance(job, dict):
                    continue
                manifest_path = job.get("asset_manifest_path")
                manifest = self.session.load_json(Path(manifest_path)) if isinstance(manifest_path, str) else None
                if manifest is None:
                    if job.get("status") != "running":
                        unresolved_risks.append(f"{kind} asset manifest unavailable")
                    continue
                raw_entries = manifest.get("assets")
                if isinstance(raw_entries, list):
                    for entry in raw_entries:
                        if not isinstance(entry, dict):
                            continue
                        key = (
                            str(entry.get("logical_name", "")),
                            str(entry.get("source_type", "")),
                            str(entry.get("runtime_path", "")),
                        )
                        if key in seen_entries:
                            continue
                        seen_entries.add(key)
                        entries.append(entry)
                raw_assumptions = manifest.get("assumptions")
                if isinstance(raw_assumptions, list):
                    assumptions.extend(str(item) for item in raw_assumptions if item)
                raw_risks = manifest.get("unresolved_risks")
                if isinstance(raw_risks, list):
                    unresolved_risks.extend(str(item) for item in raw_risks if item)
                if not job.get("ok") and job.get("status") != "running":
                    unresolved_risks.extend(str(item) for item in job.get("schema_errors", []) if item)
        manifest = {
            "assets": entries,
            "assumptions": sorted(set(assumptions))
            or ["Combined Planner asset manifest assembled from completed asset jobs."],
            "unresolved_risks": sorted(set(unresolved_risks)),
        }
        dump_json(manifest, canonical_path)
        schema_errors = self.session.validate_json_schema(manifest, Path("code_agent/specs/asset_manifest.schema.json"))
        return canonical_path, schema_errors

    def _refresh_asset_state(self, *, combined_path: Path, combined_schema_errors: list[str]) -> bool:
        assets = self._ensure_assets_state()
        jobs = assets.get("jobs")
        job_values = [job for job in jobs.values() if isinstance(job, dict)] if isinstance(jobs, dict) else []
        has_running = any(job.get("status") == "running" for job in job_values) or bool(self._asset_futures)
        job_errors = [
            str(error)
            for job in job_values
            for error in (job.get("schema_errors", []) if isinstance(job.get("schema_errors"), list) else [])
        ]
        failure_classes = [
            str(failure_class)
            for job in job_values
            for failure_class in (
                job.get("failure_classes", []) if isinstance(job.get("failure_classes"), list) else []
            )
            if failure_class
        ]
        aggregate_ok = bool(job_values) and not has_running and all(bool(job.get("ok")) for job in job_values)
        aggregate_ok = aggregate_ok and not combined_schema_errors
        assets.update(
            {
                "status": "running" if has_running else "ready" if aggregate_ok else "failed",
                "ok": aggregate_ok,
                "asset_manifest_path": str(combined_path),
                "asset_generation_report_path": self._latest_asset_report_path(),
                "selected_asset_names": self._collect_asset_names("selected_asset_names"),
                "skipped_asset_names": self._collect_asset_names("skipped_asset_names"),
                "num_assets": self._sum_asset_counts(),
                "schema_errors": sorted(set(job_errors + combined_schema_errors)),
                "failure_classes": sorted(set(failure_classes)),
                "message": self._latest_asset_message(),
                "updated_at_unix": time.time(),
                "background": has_running,
            }
        )
        return aggregate_ok

    def _latest_asset_report_path(self) -> str | None:
        assets = self._ensure_assets_state()
        jobs = assets.get("jobs")
        if not isinstance(jobs, dict):
            return None
        latest_time = -1.0
        latest_path: str | None = None
        for job in jobs.values():
            if not isinstance(job, dict):
                continue
            report_path = job.get("asset_generation_report_path")
            updated = job.get("updated_at_unix") or job.get("started_at_unix") or 0.0
            if isinstance(report_path, str) and float(updated) >= latest_time:
                latest_time = float(updated)
                latest_path = report_path
        return latest_path

    def _latest_asset_message(self) -> str | None:
        assets = self._ensure_assets_state()
        jobs = assets.get("jobs")
        if not isinstance(jobs, dict):
            return None
        latest_time = -1.0
        latest_message: str | None = None
        for job in jobs.values():
            if not isinstance(job, dict):
                continue
            message = job.get("message")
            updated = job.get("updated_at_unix") or job.get("started_at_unix") or 0.0
            if isinstance(message, str) and message and float(updated) >= latest_time:
                latest_time = float(updated)
                latest_message = message
        return latest_message

    def _collect_asset_names(self, key: str) -> list[str]:
        assets = self._ensure_assets_state()
        jobs = assets.get("jobs")
        if not isinstance(jobs, dict):
            return []
        names: list[str] = []
        for job in jobs.values():
            if isinstance(job, dict) and isinstance(job.get(key), list):
                names.extend(str(item) for item in job[key] if item)
        return sorted(set(names))

    def _sum_asset_counts(self) -> int:
        assets = self._ensure_assets_state()
        jobs = assets.get("jobs")
        if not isinstance(jobs, dict):
            return 0
        total = 0
        for job in jobs.values():
            if isinstance(job, dict):
                total += int(job.get("num_assets") or 0)
        return total

    def _asset_names_from_action(self, action: dict[str, Any]) -> list[str] | None:
        raw_names = action.get("asset_names")
        if not isinstance(raw_names, list):
            return None
        names = [item for item in raw_names if isinstance(item, str) and item.strip()]
        return names or None
