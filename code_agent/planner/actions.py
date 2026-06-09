from __future__ import annotations

from typing import Any

from code_agent.planner.action_handlers.asset_actions import AssetActionHandler
from code_agent.planner.action_handlers.runtime_actions import RuntimeActionHandler
from code_agent.planner.action_handlers.worker_actions import WorkerActionHandler


class EpisodeActionExecutor:
    """Route planner-selected actions to focused action handlers."""

    def __init__(self, session: Any):
        self.session = session
        self.assets = AssetActionHandler(session)
        self.workers = WorkerActionHandler(session)
        self.runtime = RuntimeActionHandler(session)

    def execute(self, action: dict[str, Any], turn: int) -> dict[str, Any]:
        name = action.get("action")
        self.assets.poll_asset_jobs(wait=False)
        try:
            if name == "write_plan":
                return self.runtime.write_plan(action)
            if name == "start_mesh_assets":
                return self.assets.start_mesh_assets(action)
            if name == "wait_mesh_assets":
                return self.assets.wait_mesh_assets(action)
            if name == "update_mesh_asset_metadata":
                return self.assets.update_mesh_asset_metadata(action)
            if name == "start_xml_assets":
                return self.assets.start_xml_assets(action)
            if name == "wait_xml_assets":
                return self.assets.wait_xml_assets(action)
            if name == "inspect_assets":
                return self.assets.inspect_assets(action)
            if name == "spawn_workers":
                return self.workers.spawn_workers(action)
            if name == "run_integrator":
                return self.runtime.run_integrator()
            if name == "run_execution":
                return self.runtime.run_execution(action)
            if name == "run_critic":
                return self.runtime.run_critic(action)
            if name == "run_opt":
                return self.runtime.run_opt(action)
            if name == "request_repair":
                return self.workers.request_repair(action)
            if name == "run_python":
                return self.runtime.run_command(
                    action,
                    turn,
                    label="python",
                    executable=("uv", "run", "--no-sync", "python"),
                )
            if name == "run_pytest":
                return self.runtime.run_command(
                    action,
                    turn,
                    label="pytest",
                    executable=("uv", "run", "--no-sync", "pytest"),
                )
            if name == "finish":
                return self.runtime.finish(action)
        except Exception as exc:
            return {"ok": False, "status": "error", "message": f"{type(exc).__name__}: {exc}"}
        return {"ok": False, "status": "invalid_action", "message": f"Unsupported action: {name!r}"}
