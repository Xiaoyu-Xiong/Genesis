from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import time
from pathlib import Path
from typing import Any

from code_agent.assets.mesh.episode import generate_mesh_assets_for_episode
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

    def start_xml_assets(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._start_asset_job(
            action,
            kind="xml",
            generator=generate_xml_assets_for_episode,
            ready_status="xml_assets_ready",
            running_status="xml_assets_running",
            started_status="xml_assets_started",
            report_path=self.session.case_dir / "reports" / "xml_asset_generation_report.json",
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
        except Exception as exc:  # noqa: BLE001
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
    ) -> dict[str, Any]:
        planner_output = self.session.current_planner_output()
        if planner_output is None:
            return {"ok": False, "status": "precondition_failed", "message": "planner_output is missing."}
        future = self._asset_futures.get(kind)
        if future is not None and not future.done():
            return {
                "ok": True,
                "status": running_status,
                "message": f"{kind} asset generation is already running in the background.",
                "asset_generation_report_path": self._asset_job_report_path_text(kind),
                "background": True,
            }
        if self._asset_job_ready(kind):
            return {
                "ok": True,
                "status": ready_status,
                "message": f"{kind} asset manifest is already ready.",
                "asset_manifest_path": self._asset_manifest_path_text(),
                "asset_generation_report_path": self._asset_job_report_path_text(kind),
            }
        self._shutdown_asset_executor(kind)
        asset_names = self._asset_names_from_action(action)
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
            "background": True,
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
        jobs[kind] = {
            "status": "ready" if ok else "failed",
            "ok": ok,
            "kind": kind,
            "asset_manifest_path": str(manifest_path),
            "asset_generation_report_path": result.get("asset_generation_report_path"),
            "selected_asset_names": result.get("selected_asset_names", []),
            "skipped_asset_names": result.get("skipped_asset_names", []),
            "num_assets": result.get("num_assets", 0),
            "schema_errors": schema_errors,
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
            "num_assets": result.get("num_assets", 0),
            "selected_asset_names": result.get("selected_asset_names", []),
            "skipped_asset_names": result.get("skipped_asset_names", []),
            "schema_errors": schema_errors,
            "combined_schema_errors": combined_errors,
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
        return isinstance(manifest_path, str) and Path(manifest_path).exists()

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
